from datetime import datetime, timedelta
import logging

from scrapers.daijiworld import DaijiworldScraper
from scrapers.newsapi import NewsAPIFetcher
from scrapers.newsdata import NewsDataFetcher
from scrapers.worldnews import WorldNewsFetcher

logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self):
        self.scrapers = {
            "daijiworld": DaijiworldScraper(),
            "newsapi": NewsAPIFetcher(),
            "newsdata": NewsDataFetcher(),
            "worldnews": WorldNewsFetcher(),
        }

        self.cache_duration = timedelta(minutes=10)
        self.last_fetch = None
        self.cached_articles = []

    def fetch_all_news(self, force_refresh=False, limit=120):
        # Use cache if valid
        if not force_refresh and self.cached_articles and self.last_fetch:
            if datetime.utcnow() - self.last_fetch < self.cache_duration:
                logger.info("Using cached news")
                return self.cached_articles[:limit]

        all_articles = []

        # Fetch equally from each scraper
        per_source_limit = max(1, limit // len(self.scrapers))

        for name, scraper in self.scrapers.items():
            try:
                articles = scraper.fetch_news(limit=per_source_limit)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        # Deduplicate by URL (first come = first kept)
        unique = {}
        for art in all_articles:
            if art.get("url") and art["url"] not in unique:
                unique[art["url"]] = art

        self.cached_articles = list(unique.values())[:limit]
        self.last_fetch = datetime.utcnow()

        logger.info(
            f"Fetched {len(self.cached_articles)} total articles "
            f"from {len(self.scrapers)} sources"
        )

        return self.cached_articles


# singleton
_fetcher = NewsFetcher()

def fetch_news(force_refresh=False, limit=120):
    return _fetcher.fetch_all_news(force_refresh=force_refresh, limit=limit)
