import requests
# from readability import Document
from bs4 import BeautifulSoup

def extract_article_text(url):
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    doc = Document(response.text)
    soup = BeautifulSoup(doc.summary(html_partial=True), "html.parser")

    text = soup.get_text(separator="\n").strip()
    return text[:6000]
