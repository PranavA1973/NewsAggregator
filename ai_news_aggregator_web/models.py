from flask_pymongo import PyMongo
from flask_login import UserMixin
from bson import ObjectId
import bcrypt

mongo = PyMongo()

# --------------------------------------------------
# Database initialization
# --------------------------------------------------

def init_db(app):
    with app.app_context():
        mongo.db.users.create_index("username", unique=True)
        mongo.db.summaries.create_index("key", unique=True)

# --------------------------------------------------
# User Model
# --------------------------------------------------

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.preferences = user_data.get("preferences", [])
        self.saved_articles = user_data.get("saved_articles", [])

    @staticmethod
    def get(user_id):
        try:
            user_data = mongo.db.users.find_one(
                {"_id": ObjectId(user_id)}
            )
            return User(user_data) if user_data else None
        except Exception:
            return None

    @staticmethod
    def create(username, password,email=None):
        hashed = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        result = mongo.db.users.insert_one({
            "username": username,
            "password": hashed,
            "preferences": [],
            "saved_articles": []  # stores ARTICLE IDs (hashes)
        })

        return str(result.inserted_id)

    @staticmethod
    def validate(username, password):
        user = mongo.db.users.find_one({"username": username})
        if user and bcrypt.checkpw(
            password.encode("utf-8"),
            user["password"]
        ):
            return User(user)
        return None

    def update_preferences(self, categories):
        mongo.db.users.update_one(
            {"_id": ObjectId(self.id)},
            {"$set": {"preferences": categories}}
        )
        self.preferences = categories

    # ----------------------------------------------
    # Saved Articles (ID-based)
    # ----------------------------------------------

    def save_article(self, article_id):
        if article_id not in self.saved_articles:
            mongo.db.users.update_one(
                {"_id": ObjectId(self.id)},
                {"$addToSet": {"saved_articles": article_id}}
            )
            self.saved_articles.append(article_id)

    def unsave_article(self, article_id):
        mongo.db.users.update_one(
            {"_id": ObjectId(self.id)},
            {"$pull": {"saved_articles": article_id}}
        )
        self.saved_articles = [
            a for a in self.saved_articles if a != article_id
        ]
