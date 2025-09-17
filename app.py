# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches
import requests

# --- TMDB API function to fetch poster ---
def get_poster(movie_name):
    api_key = "febc261adb0704290fbf6328907726cb"  # Replace with your TMDB API key
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
    data = requests.get(url).json()
    if data['results']:
        poster_path = data['results'][0]['poster_path']
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/150x220?text=No+Image"

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Data for repository.csv")  # Update with your CSV filename
    # Fill missing values
    for col in ['Genre', 'Lead Star', 'Director']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    # Combine features
    df['content'] = df.apply(lambda row: (row['Genre'] + " ") + (row['Director'] + " ") * 2 + (row['Lead Star'] + " ") * 3, axis=1)
    return df

movies = load_data()

# --- Create indices for fast lookup ---
indices = pd.Series(movies.index, index=movies['Movie Name']).drop_duplicates()

# --- TF-IDF vectorization ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# --- Recommendation function ---
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
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Movie Name'].iloc[movie_indices]

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Bollywood Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

selected_movie = st.selectbox("Choose a movie:", movies['Movie Name'].values)
num_rec = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Recommend"):
    recs = recommend_movies(selected_movie, num_rec)
    if len(recs) > 0:
        st.subheader("Recommended Movies:")
        cols = st.columns(2)
        for i, movie in enumerate(recs):
            col = cols[i % 2]
            with col:
                poster_url = get_poster(movie)
                st.image(poster_url, width=150)
                st.write(movie)
    else:
        st.warning("No similar movies found.")




