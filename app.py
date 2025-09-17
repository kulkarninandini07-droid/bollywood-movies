import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bollywood_movies.csv")  # Make sure this matches your CSV filename

movies = load_data()

# Fill missing values
for col in ['Genre', 'Lead Star', 'Director']:
    if col in movies.columns:
        movies[col] = movies[col].fillna('')

# ----------------------------
# Feature Engineering
# ----------------------------
def combine_features(row):
    return (row['Genre'] + " ") * 1 + (row['Director'] + " ") * 2 + (row['Lead Star'] + " ") * 3

movies['content'] = movies.apply(combine_features, axis=1)

# ----------------------------
# TF-IDF + Similarity
# ----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

indices = pd.Series(movies.index, index=movies['Movie Name']).drop_duplicates()

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend_movies(title, num_recommendations=10):
    title = title.strip()
    if title not in indices:
        close_matches = get_close_matches(title, indices.index, n=1, cutoff=0.6)
        if close_matches:
            title = close_matches[0]
        else:
            return []
    idx = indices[title]
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix, tfidf_matrix)[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Movie Name'].iloc[movie_indices]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Bollywood Movie Recommender")

selected_movie = st.selectbox("Choose a movie:", movies['Movie Name'].values)

if st.button("Recommend"):
    recs = recommend_movies(selected_movie, 10)
    if len(recs) > 0:
        st.subheader("Recommended Movies:")
        for movie in recs:
            st.write(movie)
    else:
        st.warning("No similar movies found.")
