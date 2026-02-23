import os

BASE_DIR = "ai_news_aggregator_web"

folders = [
    BASE_DIR,
    f"{BASE_DIR}/templates",
    f"{BASE_DIR}/cache"
]

files = [
    f"{BASE_DIR}/app.py",
    f"{BASE_DIR}/requirements.txt",
    f"{BASE_DIR}/config.py",
    f"{BASE_DIR}/news_fetcher.py",
    f"{BASE_DIR}/recommender.py",
    f"{BASE_DIR}/models.py",
    f"{BASE_DIR}/templates/base.html",
    f"{BASE_DIR}/templates/index.html",
    f"{BASE_DIR}/templates/article.html",
    f"{BASE_DIR}/templates/login.html",
    f"{BASE_DIR}/templates/register.html",
    f"{BASE_DIR}/templates/profile.html",
    f"{BASE_DIR}/templates/saved.html",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write("")

print("✅ Project structure created successfully!")
