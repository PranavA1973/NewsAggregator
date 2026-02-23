import requests
import hashlib
from models import mongo
from config import NVIDIA_API_KEY, NVIDIA_MODEL
from extractor import extract_article_text
from robots import is_scraping_allowed

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def get_article_summary(url, title, description):
    cache_key = hashlib.md5(url.encode()).hexdigest()

    # -----------------------------
    # Cache check
    # -----------------------------
    cached = mongo.db.summaries.find_one({"key": cache_key})
    if cached:
        return cached["summary"]

    # -----------------------------
    # robots.txt check
    # -----------------------------
    if not is_scraping_allowed(url):
        summary = "Summary unavailable due to site restrictions."
        _cache_summary(cache_key, url, summary)
        return summary

    # -----------------------------
    # Extract content safely
    # -----------------------------
    article_text = ""

    try:
        extracted = extract_article_text(url)
        if isinstance(extracted, str):
            article_text = extracted.strip()
    except Exception as e:
        print(f"[EXTRACT ERROR] {e}")

    # -----------------------------
    # Fallback to description/title
    # -----------------------------
    if len(article_text) < 150:
        article_text = (description or title or "").strip()

    # -----------------------------
    # Still too short → no LLM
    # -----------------------------
    if len(article_text) < 100:
        summary = article_text or "Not enough content to summarize."
        _cache_summary(cache_key, url, summary)
        return summary

    # -----------------------------
    # LLM request
    # -----------------------------
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "Summarize the following news article in 2–4 factual sentences. "
                "Do not speculate or add opinions.\n\n"
                f"Title: {title}\n\n"
                f"{article_text[:3000]}"
            )
        }],
        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=25
        )
        response.raise_for_status()

        data = response.json()
        summary = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not summary:
            summary = article_text[:200]

        _cache_summary(cache_key, url, summary)
        return summary

    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
        summary = article_text[:200] or "Summary temporarily unavailable."
        _cache_summary(cache_key, url, summary)
        return summary


def _cache_summary(cache_key, url, summary):
    """Cache summary safely"""
    try:
        mongo.db.summaries.update_one(
            {"key": cache_key},
            {"$set": {"summary": summary, "url": url}},
            upsert=True
        )
    except Exception as e:
        print(f"[CACHE ERROR] {e}")
