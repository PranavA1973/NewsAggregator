import urllib.robotparser as robotparser
from urllib.parse import urlparse

def is_scraping_allowed(url, user_agent="*"):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False  # Fail-safe: do NOT scrape
