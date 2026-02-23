from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file, make_response, session
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from flask_caching import Cache
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import io
import pandas as pd
import hashlib
import json
import time
from functools import wraps
import logging

# Custom modules
from news_fetcher import fetch_news, NewsFetcher
#from categorizer import categorize_article, CATEGORIES
from recommender import build_recommender, get_recommendations
from models import mongo, User, init_db
import config
from summarizer import get_article_summary
from image_handler import get_image_with_fallback, extract_image_from_url
from openai import OpenAI
# --------------------------------------------------
# Setup Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App setup
# --------------------------------------------------

app = Flask(__name__)
app.config.from_object(config)



# Initialize cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Initialize MongoDB
mongo.init_app(app)
init_db(app)


llama_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=config.NVIDIA_API_KEY
)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# --------------------------------------------------
# Helper Functions & Decorators
# --------------------------------------------------

# Define categories list
CATEGORIES = ["Technology", "Business", "Sports", "Politics", "Entertainment", "Health", "Science", "Local", "General"]

LLAMA_CATEGORY_PROMPT = """
You are a news classifier.

Choose ONLY ONE category from this list:
Technology, Business, Sports, Politics, Entertainment, Health, Science, Local, General

Rules:
- Respond with ONLY the category name
- No explanations
- No punctuation

Article:
Title: {title}
Description: {description}
"""

def ai_categorize_article(title, description, llama_client):
    prompt = LLAMA_CATEGORY_PROMPT.format(
        title=title[:200],
        description=description[:500]
    )

    response = llama_client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    category = response.choices[0].message.content.strip()

    return category if category in CATEGORIES else "General"


def needs_ai_category(category):
    return category in ("General", None, "")




def get_processed_articles(force_refresh=False):
    cache_key = "processed_articles"

    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    raw = get_cached_news(force_refresh=force_refresh, limit=100)
    processed = process_articles(raw, enrich_images=True)
    processed.sort(key=local_boost_sort)

    cache.set(cache_key, processed, timeout=300)
    return processed


def get_article_id(article):
    """Generate unique ID for article"""
    return hashlib.md5(article["url"].encode()).hexdigest()

def cache_news(timeout=300):
    """Decorator to cache news data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            force_refresh = kwargs.get('force_refresh', False) or request.args.get('refresh') == 'true'
            cache_key = f"news_data_{request.args.get('category', 'all')}_{request.args.get('search', '')}"
            
            if not force_refresh:
                cached = cache.get(cache_key)
                if cached:
                    logger.info(f"Retrieved news from cache with key: {cache_key}")
                    return cached
            
            data = f(*args, **kwargs)
            return data
        return decorated_function
    return decorator
def get_cached_news(force_refresh=False, limit=100):
    cache_key = f"raw_news_{limit}"

    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    news = fetch_news(force_refresh=force_refresh, limit=limit)
    cache.set(cache_key, news, timeout=300)  # 5 min
    return news

def get_default_image():
    """Get default image URL based on category"""
    # You can replace these with your own image URLs
    default_images = {
        "General": "https://images.unsplash.com/photo-1495020689067-958852a7765e?w=800&auto=format&fit=crop",
        "Technology": "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w-800&auto=format&fit=crop",
        "Sports": "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w-800&auto=format&fit=crop",
        "Entertainment": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w-800&auto=format&fit=crop",
        "Business": "https://images.unsplash.com/photo-1444653614773-995cb1ef9efa?w-800&auto=format&fit=crop",
        "Health": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w-800&auto=format&fit=crop",
        "Science": "https://images.unsplash.com/photo-1532094349884-543bc11b234d?w-800&auto=format&fit=crop",
        "Local": "https://images.unsplash.com/photo-1518398046578-8cca57782e17?w-800&auto=format&fit=crop"
    }
    return default_images

def enhance_article_images(article):
    """Enhance article with proper images"""
    # Priority order for image sources
    if article.get('urlToImage'):
        article['image'] = article['urlToImage']
    elif article.get('image'):
        article['image'] = article['image']
    else:
        # Try to extract image from article URL
        try:
            extracted_image = extract_image_from_url(article['url'])
            if extracted_image:
                article['image'] = extracted_image
            else:
                # Use category-based default image
                default_images = get_default_image()
                category = article.get('category', 'General')
                article['image'] = default_images.get(category, default_images['General'])
        except:
            default_images = get_default_image()
            category = article.get('category', 'General')
            article['image'] = default_images.get(category, default_images['General'])
    
    # Ensure image URL is HTTPS
    if article.get('image') and article['image'].startswith('http://'):
        article['image'] = article['image'].replace('http://', 'https://')
    
    # Add placeholder image if still missing
    if not article.get('image'):
        article['image'] = "https://via.placeholder.com/800x450.png?text=News+Image"
    
    return article

def process_articles(articles, enrich_images=True):
    """
    Process and enhance articles with:
    - Strong Local dominance
    - Rule-based + AI categorization
    - Image guarantees
    - Date normalization
    - Preview & reading time
    """
    processed = []

    for art in articles:
        try:
            # --------------------------------------------------
            # ID
            # --------------------------------------------------
            if not art.get("url"):
                continue

            art["id"] = get_article_id(art)

            # --------------------------------------------------
            # CATEGORY (Local → Rules → AI fallback)
            # --------------------------------------------------
            source_name = art.get("source", {}).get("name", "").lower()
            raw_cat = art.get("category")

            # 1️⃣ Force Local for Daijiworld
            if "daijiworld" in source_name:
                category = "Local"

            # 2️⃣ API explicitly says local
            elif raw_cat and raw_cat.lower() in ("local", "india", "indian"):
                category = "Local"

            # 3️⃣ Rule-based categorization
            else:
                category = categorize_article(
                    art.get("title", ""),
                    art.get("description", ""),
                    raw_cat
                )

                # 4️⃣ AI fallback only if needed
                if needs_ai_category(category):
                    cache_key = f"ai_cat_{art['id']}"
                    cached_cat = cache.get(cache_key)

                    if cached_cat:
                        category = cached_cat
                    else:
                        try:
                            category = ai_categorize_article(
                                art.get("title", ""),
                                art.get("description", ""),
                                llama_client
                            )
                            cache.set(cache_key, category, timeout=86400 * 30)
                        except Exception as e:
                            logger.warning(f"AI category failed: {e}")

            art["category"] = category

            # --------------------------------------------------
            # DESCRIPTION
            # --------------------------------------------------
            desc = (
                art.get("description")
                or art.get("content")
                or art.get("title")
                or "No description available."
            ).strip()

            art["description"] = desc[:300] + "..." if len(desc) > 300 else desc

            # --------------------------------------------------
            # PREVIEW
            # --------------------------------------------------
            art["preview"] = (
                art["description"][:150] + "..."
                if len(art["description"]) > 150
                else art["description"]
            )

            # --------------------------------------------------
            # IMAGES
            # --------------------------------------------------
            if enrich_images:
                art = enhance_article_images(art)

            if not art.get("image"):
                art["image"] = "https://via.placeholder.com/800x450.png?text=News"

            # --------------------------------------------------
            # SOURCE ICON
            # --------------------------------------------------
            source_display = art.get("source", {}).get("name", "Unknown")

            source_icons = {
                "Daijiworld": "fas fa-newspaper",
                "Times of India": "fas fa-landmark",
                "BBC News": "fas fa-globe",
                "CNN": "fas fa-tv",
                "Reuters": "fas fa-broadcast-tower",
                "Al Jazeera": "fas fa-satellite-dish",
            }

            art["source_icon"] = source_icons.get(
                source_display, "fas fa-newspaper"
            )

            # --------------------------------------------------
            # DATE HANDLING
            # --------------------------------------------------
            pub_date = art.get("publishedAt")
            art["_published_dt"] = (
                pub_date if isinstance(pub_date, datetime) else datetime.min
            )

            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(
                            pub_date.replace("Z", "+00:00")
                        )
                    art["time_ago"] = get_time_ago(pub_date)
                    art["formatted_date"] = pub_date.strftime(
                        "%b %d, %Y · %I:%M %p"
                    )
                except Exception:
                    art["time_ago"] = "Recently"
                    art["formatted_date"] = "Recent"
            else:
                art["time_ago"] = "Recently"
                art["formatted_date"] = "Recent"

            # --------------------------------------------------
            # READING TIME
            # --------------------------------------------------
            word_count = len(art["description"].split())
            art["reading_time"] = max(1, word_count // 200)

            processed.append(art)

        except Exception as e:
            logger.error(f"process_articles error: {e}", exc_info=True)
            continue

    return processed

def get_time_ago(pub_date):
    """Get human-readable time difference"""
    now = datetime.utcnow()
    diff = now - pub_date
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

# --------------------------------------------------
# HOME / INDEX (MERGED)
# --------------------------------------------------
def local_boost_sort(article):
    if not isinstance(article, dict):
        return (1, "")
    return (
    0 if article.get("category") == "Local" else 1,
    article.get("_published_dt", datetime.min)
)


@app.route("/")
@app.route("/home")
def index():
    start_time = time.time()
    
    page = int(request.args.get("page", 1))
    category = request.args.get("category", "All")
    search = request.args.get("search", "").strip().lower()
    refresh = request.args.get("refresh", "false") == "true"
    sort_by = request.args.get("sort", "relevance")  # relevance, newest, oldest
    
    error_message = None
    articles = []
    recommended = []
    
    try:
        # Fetch news articles
        logger.info(f"Fetching news with refresh={refresh}")
        articles = get_processed_articles(force_refresh=refresh)

# Increase limit
        
        if not isinstance(articles, list):
            articles = []
            error_message = "Invalid news data received."
            logger.error("News data is not a list")
        else:
            logger.info(f"Fetched {len(articles)} raw articles")
            
            # Process and enhance articles

            from collections import Counter
            logger.info("Category distribution: %s", Counter(a["category"] for a in articles))

            logger.info(f"Processed {len(articles)} articles")
            
            # Count sources
            source_counter = Counter(a["source"]["name"] for a in articles)
            logger.info(f"Sources: {dict(source_counter)}")
            
            # Cache summaries in background
            cache_summaries_background(articles)
            
            # Build recommender
            build_recommender(articles)
            
            # Get recommendations for authenticated users
            if current_user.is_authenticated:
                recommended = get_personalized_recommendations(articles, current_user)
                logger.info(f"Generated {len(recommended)} recommendations")
                
                # Sort articles based on user preferences
                articles = sort_by_user_preferences(articles, current_user)
                articles.sort(key=local_boost_sort)

            
    except Exception as e:
        logger.error(f"Failed to fetch news: {str(e)}", exc_info=True)
        error_message = f"Failed to fetch news: {str(e)}"
        # Try to get cached articles as fallback
        try:
            cached_articles = cache.get("cached_articles_fallback")
            if cached_articles:
                articles = cached_articles
                logger.info("Using cached articles as fallback")
        except:
            pass
    
    # Filtering
    filtered = filter_articles(articles, category, search, sort_by)
    
    # Pagination
    per_page = 12  # Show more articles per page
    total = len(filtered)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated = filtered[start_idx:end_idx]
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page if total else 1
    
    # Get trending articles (most saved)
    trending = get_trending_articles(articles)[:5]
    
    # Prepare response
    response_data = {
        "articles": paginated,
        "recommended": recommended,
        "trending": trending,
        "categories": ["All"] + CATEGORIES,
        "current_category": category,
        "search": search,
        "sort_by": sort_by,
        "page": page,
        "total_pages": total_pages,
        "total_articles": total,
        "has_prev": page > 1,
        "has_next": end_idx < total,
        "error": error_message,
        "load_time": round(time.time() - start_time, 2)
    }
    
    # Cache processed articles for fallback
    cache.set("cached_articles_fallback", articles, timeout=600)
    
    # Return JSON for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(response_data)
    
    return render_template("index.html", **response_data)

def cache_summaries_background(articles):
    """Cache article summaries in background"""
    try:
        for art in articles[:20]:  # Cache first 20 articles
            cache_key = f"summary_{art['id']}"
            if not cache.get(cache_key):
                # Store minimal summary for preview
                cache.set(cache_key, {
                    'preview': art.get('preview', ''),
                    'title': art.get('title', '')
                }, timeout=3600)
    except Exception as e:
        logger.error(f"Background caching failed: {e}")

def get_personalized_recommendations(articles, user):
    """Get personalized recommendations for user"""
    try:
        id_to_index = {get_article_id(a): i for i, a in enumerate(articles)}
        saved_indices = [
            id_to_index[a_id]
            for a_id in user.saved_articles
            if a_id in id_to_index
        ]
        
        if saved_indices:
            rec_indices = get_recommendations(saved_indices, top_n=10)

            recommended = [articles[i] for i in rec_indices if i < len(articles)]
            
            # Remove duplicates and articles already saved
            seen = set()
            unique_recommended = []
            for art in recommended:
                art_id = art['id']
                if art_id not in seen and art_id not in user.saved_articles:
                    seen.add(art_id)
                    unique_recommended.append(art)
            
            return unique_recommended[:6]  # Return top 6 recommendations
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
    
    return []

def sort_by_user_preferences(articles, user):
    """Sort articles based on user preferences"""
    if not user.preferences:
        return articles
    
    preferred_categories = set(user.preferences)
    
    def sort_key(a):
        category = a.get("category", "").lower()
        if category in preferred_categories:
            return (0, -len(a.get("description", "")))  # Preferred categories first
        else:
            return (1, -len(a.get("description", "")))  # Others after
    
    return sorted(articles, key=sort_key)

def filter_articles(articles, category, search, sort_by):
    """Filter and sort articles"""
    filtered = articles
    
    # Filter by category
    if category != "All":
        filtered = [
            a for a in filtered
            if a.get("category", "").strip().lower() == category.strip().lower()

        ]
    
    # Filter by search
    if search:
        filtered = [
            a for a in filtered
            if search in (a.get("title") or "").lower()
            or search in (a.get("description") or "").lower()
            or search in (a.get("category") or "").lower()
        ]
    
    # Sort articles
    if sort_by == "newest":
        filtered.sort(key=lambda x: x.get("_published_dt", datetime.min), reverse=True)
    elif sort_by == "oldest":
        filtered.sort(key=lambda x: x.get("_published_dt", datetime.min))
    elif sort_by == "relevance":
        # Keep current order (already sorted by preferences if logged in)
        pass
    
    return filtered

def get_trending_articles(articles, days=7):
    """Get trending articles based on saves in last N days"""
    try:
        # Get recent saves from database
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_saves = mongo.db.user_saves.find({
            "saved_at": {"$gte": cutoff}
        })
        
        # Count saves per article
        save_counter = defaultdict(int)
        for save in recent_saves:
            save_counter[save.get('article_id')] += 1
        
        # Sort articles by save count
        article_map = {art['id']: art for art in articles}
        trending = []
        for art_id, count in sorted(save_counter.items(), key=lambda x: x[1], reverse=True):
            if art_id in article_map:
                art = article_map[art_id].copy()
                art['save_count'] = count
                trending.append(art)
        
        return trending[:10]  # Return top 10 trending
    except Exception as e:
        logger.error(f"Trending articles error: {e}")
        return []

# --------------------------------------------------
# ARTICLE DETAILS
# --------------------------------------------------
@app.route("/article/<article_id>")
def article(article_id):
    start_time = time.time()
    
    try:
        # Try to get from cache first
        cache_key = f"article_{article_id}"
        cached_article = cache.get(cache_key)
        
        if cached_article:
            article_data = cached_article
            logger.info(f"Retrieved article {article_id} from cache")
        else:
            # Fetch fresh articles
            articles = get_cached_news(limit=50)
            article_map = {get_article_id(a): a for a in articles}
            article_data = article_map.get(article_id)
            
            if article_data:
                # Process and enhance article
                article_data = process_articles([article_data])[0]
                # Cache for future requests
                cache.set(cache_key, article_data, timeout=600)
            else:
                return render_template("404.html", message="Article not found"), 404
        
        # Check if article is saved by current user
        is_saved = False
        if current_user.is_authenticated:
            is_saved = article_id in current_user.saved_articles
        
        # Get summary
        summary = None
        cache_key = f"summary_full_{article_id}"
        cached_summary = cache.get(cache_key)
        
        if cached_summary:
            summary = cached_summary
        else:
            try:
                summary = get_article_summary(
                    article_data["url"],
                    article_data["title"],
                    article_data.get("description")
                )
                if summary:
                    cache.set(cache_key, summary, timeout=3600)
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                summary = article_data.get("description", "")
        
        # Get related articles
        related = get_related_articles(article_id, article_data.get("category", ""))
        
        # Record article view for analytics
        if current_user.is_authenticated:
            record_article_view(article_id, current_user.id)
        
        load_time = round(time.time() - start_time, 2)
        
        return render_template(
            "article.html",
            article=article_data,
            summary=summary,
            related=related[:4],
            is_saved=is_saved,
            load_time=load_time,
            config=config
        )
        
    except Exception as e:
        logger.error(f"Article view error: {e}", exc_info=True)
        return render_template("error.html", message="Failed to load article"), 500

def get_related_articles(article_id, category, limit=4):
    """Get related articles based on category"""
    try:
        articles = get_cached_news(limit=50)
        processed = get_processed_articles()
        
        # Filter by same category, exclude current article
        related = [
            art for art in processed
            if art['id'] != article_id
            and art.get('category') == category
        ]
        
        # If not enough same-category articles, add from other categories
        if len(related) < limit:
            other_articles = [
                art for art in processed
                if art['id'] != article_id and art not in related
            ]
            related.extend(other_articles[:limit - len(related)])
        
        return related[:limit]
    except Exception as e:
        logger.error(f"Related articles error: {e}")
        return []

def record_article_view(article_id, user_id):
    """Record article view for analytics"""
    try:
        mongo.db.article_views.insert_one({
            "article_id": article_id,
            "user_id": user_id,
            "viewed_at": datetime.utcnow(),
            "ip_address": request.remote_addr
        })
    except Exception as e:
        logger.error(f"Failed to record article view: {e}")

# --------------------------------------------------
# SAVE / UNSAVE ARTICLES
# --------------------------------------------------

@app.route("/save/<article_id>", methods=["POST"])
@login_required
def save_article(article_id):
    try:
        current_user.save_article(article_id)
        
        # Update trending cache
        cache_key = f"trending_{hashlib.md5(article_id.encode()).hexdigest()}"
        cache.delete(cache_key)
        
        flash("Article saved!", "success")
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True, "saved": True})
            
    except Exception as e:
        logger.error(f"Save article error: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": str(e)}), 500
        flash("Failed to save article", "error")
    
    return redirect(request.referrer or url_for("index"))

@app.route("/unsave/<article_id>", methods=["POST"])
@login_required
def unsave_article(article_id):
    try:
        current_user.unsave_article(article_id)
        flash("Article removed.", "info")
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True, "saved": False})
            
    except Exception as e:
        logger.error(f"Unsave article error: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": str(e)}), 500
        flash("Failed to remove article", "error")
    
    return redirect(request.referrer or url_for("saved"))

@app.route("/saved")
@login_required
def saved():
    try:
        articles = fetch_news(limit=100)
        article_map = {get_article_id(a): a for a in articles}
        
        saved_articles = []
        for a_id in current_user.saved_articles:
            if a_id in article_map:
                art = article_map[a_id].copy()
                art["id"] = a_id
                # Process the article
                processed = process_articles([art])
                if processed:
                    saved_articles.append(processed[0])
        
        # Sort by save date (most recent first)
        saved_articles.sort(key=lambda x: x.get('saved_at', datetime.min), reverse=True)
        
        return render_template("saved.html", articles=saved_articles)
        
    except Exception as e:
        logger.error(f"Saved articles error: {e}")
        flash("Failed to load saved articles", "error")
        return redirect(url_for("index"))

# --------------------------------------------------
# PROFILE / PREFERENCES
# --------------------------------------------------

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        try:
            cats = request.form.getlist("categories")
            current_user.update_preferences(cats)
            
            # Clear recommendation cache for this user
            cache.delete(f"recommendations_{current_user.id}")
            
            flash("Preferences updated!", "success")
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"success": True})
                
        except Exception as e:
            logger.error(f"Update preferences error: {e}")
            flash("Failed to update preferences", "error")
    
    # Get user stats
    stats = {
        "articles_saved": len(current_user.saved_articles),
        "categories_preferred": len(current_user.preferences or []),
        "member_since": current_user.created_at.strftime("%B %d, %Y") if hasattr(current_user, 'created_at') else "Recently"
    }
    
    return render_template(
        "profile.html",
        preferences=current_user.preferences,
        categories=CATEGORIES,
        stats=stats
    )

# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    
    if request.method == "POST":
        try:
            user = User.validate(
                request.form["username"],
                request.form["password"]
            )
            if user:
                login_user(user, remember=True)
                next_page = request.args.get('next')
                
                # Clear any user-specific cache
                cache.delete(f"recommendations_{user.id}")
                
                flash(f"Welcome back, {user.username}!", "success")
                return redirect(next_page or url_for("index"))
            else:
                flash("Invalid credentials", "error")
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash("Login failed. Please try again.", "error")
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form.get("email", "")

        if mongo.db.users.find_one({"username": username}):
            flash("Username already taken", "error")
        elif email and mongo.db.users.find_one({"email": email}):
            flash("Email already registered", "error")
        else:
            try:
                User.create(username, password, email)
                flash("Registered successfully! Please login.", "success")
                return redirect(url_for("login"))
            except Exception as e:
                logger.error(f"Registration error: {e}")
                flash("Registration failed. Please try again.", "error")

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash(f"Goodbye, {username}. Come back soon!", "info")
    return redirect(url_for("index"))

# --------------------------------------------------
# EXPORT CSV
# --------------------------------------------------

@app.route("/export_csv")
@login_required
def export_csv():
    try:
        articles = fetch_news(limit=200)
        processed = process_articles(articles)
        
        # Create DataFrame
        df_data = []
        for art in processed:
            df_data.append({
                "Title": art.get("title", ""),
                "Description": art.get("description", ""),
                "Category": art.get("category", ""),
                "Source": art.get("source", {}).get("name", ""),
                "URL": art.get("url", ""),
                "Published Date": art.get("publishedAt", ""),
                "Reading Time (min)": art.get("reading_time", 0)
            })
        
        df = pd.DataFrame(df_data)
        
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"news_export_{timestamp}.csv"
        )
        
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        flash("Failed to export CSV", "error")
        return redirect(url_for("index"))

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.route("/api/news")
def api_news():
    """API endpoint for news"""
    try:
        limit = int(request.args.get("limit", 20))
        category = request.args.get("category", "all")
        page = int(request.args.get("page", 1))
        
        articles = fetch_news(limit=limit * 2)
        processed = process_articles(articles)
        
        # Filter by category
        if category != "all":
            processed = [a for a in processed if a.get("category", "").lower() == category.lower()]
        
        # Paginate
        per_page = limit
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated = processed[start_idx:end_idx]
        
        return jsonify({
            "success": True,
            "data": paginated,
            "meta": {
                "page": page,
                "per_page": per_page,
                "total": len(processed),
                "has_more": end_idx < len(processed)
            }
        })
    except Exception as e:
        logger.error(f"API news error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/search")
def api_search():
    """API endpoint for search"""
    try:
        query = request.args.get("q", "").strip().lower()
        if not query or len(query) < 2:
            return jsonify({"success": True, "data": []})
        
        articles = fetch_news(limit=100)
        processed = process_articles(articles)
        
        results = []
        for art in processed:
            if (query in (art.get("title") or "").lower() or
                query in (art.get("description") or "").lower() or
                query in (art.get("category") or "").lower()):
                results.append(art)
        
        return jsonify({
            "success": True,
            "data": results[:20],  # Limit results
            "count": len(results)
        })
    except Exception as e:
        logger.error(f"API search error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('403.html'), 403

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "News Aggregator"
    })

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting News Aggregator Application...")
    app.run(
        host=config.HOST or '0.0.0.0',
        port=config.PORT or 5000,
        debug=config.DEBUG,
        threaded=True
    )