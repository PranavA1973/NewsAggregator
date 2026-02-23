import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)

def extract_image_from_url(url, timeout=5):
    """Extract main image from article URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try different image selectors
        selectors = [
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'meta[property="twitter:image"]',
            'img.article-image',
            'img.wp-post-image',
            'img.attachment-full',
            'img[src*="article"]',
            'img[src*="news"]',
            'img:not([src*="icon"]):not([src*="logo"])'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                src = elem.get('content') or elem.get('src')
                if src:
                    full_url = urljoin(url, src)
                    # Validate it's an image URL
                    if any(ext in full_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                        return full_url
        
        # Try to find any image in the main content
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=lambda x: x and ('content' in x or 'article' in x))
        if main_content:
            images = main_content.find_all('img')
            for img in images:
                src = img.get('src')
                if src:
                    full_url = urljoin(url, src)
                    if any(ext in full_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                        return full_url
        
    except Exception as e:
        logger.debug(f"Failed to extract image from {url}: {e}")
    
    return None

def get_image_with_fallback(article, default_images):
    """Get image with multiple fallback strategies"""
    # Try multiple image sources in order
    image_sources = [
        article.get('urlToImage'),
        article.get('image'),
        extract_image_from_url(article.get('url')),
        default_images.get(article.get('category', 'General')),
        default_images.get('General')
    ]
    
    for img in image_sources:
        if img and isinstance(img, str) and img.strip():
            # Ensure HTTPS
            if img.startswith('http://'):
                img = img.replace('http://', 'https://')
            return img
    
    return "https://via.placeholder.com/800x450.png?text=News+Image"