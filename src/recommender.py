from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def recommend_by_content(df, selected_title, top_n=5):
    df = df.dropna(subset=["title", "genres", "imdbAverageRating"])
    df = df.drop_duplicates(subset=["title"])

    # Join genre list back to string for vectorizer
    df["genre_str"] = df["genres"].apply(lambda x: " ".join(x))

    # Vectorize genres
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genre_str"])

    # Find index of selected movie
    idx = df[df["title"] == selected_title].index[0]

    # Compute cosine similarity
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get top N similar movies
    top_indices = sim_scores.argsort()[-top_n-1:-1][::-1]  # Exclude the movie itself
    recommendations = df.iloc[top_indices][["title", "genres", "imdbAverageRating"]]

    return recommendations.reset_index(drop=True)
