# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("bollywood_movies_full.csv")
    # Fill missing values
    for col in ['Genre', 'Lead Star', 'Director', 'Plot']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    # Combine features for content-based similarity
    df['content'] = df['Genre'] + " " + df['Director'] + " " + df['Lead Star'] + " " + df['Plot']
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
        # Fuzzy matching
        close_matches = get_close_matches(title, indices.index, n=1, cutoff=0.6)
        if close_matches:
            title = close_matches[0]
        else:
            return pd.DataFrame()  # No match
    idx = indices[title]
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix, tfidf_matrix)[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Bollywood Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

# Searchable input
selected_movie = st.text_input("Enter a movie name:", "")
num_rec = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Recommend"):
    if selected_movie.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recs = recommend_movies(selected_movie, num_rec)
        if not recs.empty:
            st.subheader("Recommended Movies:")
            # Display in 3-column grid
            cols = st.columns(3)
            for i, (_, row) in enumerate(recs.iterrows()):
                col = cols[i % 3]
                with col:
                    # Poster
                    poster_url = row['Poster URL'] if row['Poster URL'] else "https://via.placeholder.com/150x220?text=No+Image"
                    st.image(poster_url, width=150)
                    # Movie info
                    st.markdown(f"**{row['Movie Name']}**")
                    st.markdown(f"⭐ Rating: {row['Rating']}  |  📅 Year: {row['Release Year']}")
                    st.markdown(f"🎭 Genre: {row['Genre']}")
        else:
            st.warning("No similar movies found. Check spelling or try another movie.")
