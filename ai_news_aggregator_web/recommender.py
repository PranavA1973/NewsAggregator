# recommender.py - FINAL FIX: NO MORE np.matrix ERRORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = None
valid_indices = []  # original article indices that made it into the matrix

def build_recommender(articles):
    global tfidf_matrix, valid_indices
    valid_texts = []
    valid_indices = []

    for i, art in enumerate(articles):
        text = f"{art.get('title', '')} {art.get('description', '')}".strip().lower()
        if len(text) > 20:  # meaningful content
            valid_texts.append(text)
            valid_indices.append(i)

    if len(valid_texts) < 2:
        tfidf_matrix = None
        valid_indices = []
        return

    try:
        # Fit and transform — this returns a sparse matrix
        sparse_matrix = vectorizer.fit_transform(valid_texts)
        
        # CRITICAL: Convert to dense ndarray immediately
        tfidf_matrix = np.asarray(sparse_matrix.toarray(), dtype=np.float32)
        
        print(f"Recommender built with {len(valid_texts)} valid articles")
    except Exception as e:
        print(f"Recommender build failed: {e}")
        tfidf_matrix = None
        valid_indices = []

def get_recommendations(saved_indices, top_n=5):
    global tfidf_matrix, valid_indices

    if (tfidf_matrix is None or 
        tfidf_matrix.shape[0] < 2 or 
        len(valid_indices) < 2 or 
        not saved_indices):
        return []

    # Only use saved articles that exist in our valid matrix
    valid_saved = [i for i in saved_indices if i in valid_indices]
    if not valid_saved:
        return []

    # Map to row indices in tfidf_matrix
    matrix_rows = [valid_indices.index(i) for i in valid_saved]

    try:
        # Use dense array — safe for cosine_similarity
        saved_vectors = tfidf_matrix[matrix_rows]
        user_profile = np.mean(saved_vectors, axis=0).reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()

        # Hide already saved articles
        similarities[matrix_rows] = -1

        # Get top indices
        top_matrix_idx = np.argsort(similarities)[-top_n:][::-1]
        
        # Convert back to original article indices
        recommendations = []
        for idx in top_matrix_idx:
            if similarities[idx] > 0.05:  # minimum relevance
                recommendations.append(valid_indices[idx])
            if len(recommendations) >= top_n:
                break

        return recommendations

    except Exception as e:
        print(f"Recommendation error: {e}")
        return []