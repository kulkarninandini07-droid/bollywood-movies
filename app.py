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
    return pd.read_csv("bollywood_movies.csv")  # make sure your CSV is in the repo

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
# Movie posters (static demo)
# ----------------------------
movie_posters = {
    "3 Idiots": "https://upload.wikimedia.org/wikipedia/en/d/df/3_idiots_poster.jpg",
    "Lagaan": "https://upload.wikimedia.org/wikipedia/en/d/db/Lagaan_movie_poster.jpg",
    "PK": "https://upload.wikimedia.org/wikipedia/en/9/9d/PK_Poster.jpg",
    "Taare Jameen Par": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoIFyKAL4ldG54LGrepUXMraXASXkEmk4b79Uv_tPy35dtC1fmCTz3VBAfC9wR-ZugQjiB"
    "Dangal": "https://upload.wikimedia.org/wikipedia/en/9/9d/Dangal_Poster.jpg",
    "Bajrangi Bhaijaan": "https://upload.wikimedia.org/wikipedia/en/6/60/Bajrangi_Bhaijaan_Poster.jpg",
    "Sultan": "https://upload.wikimedia.org/wikipedia/en/f/f4/Sultan_Poster.jpg",
    "Padmaavat": "https://upload.wikimedia.org/wikipedia/en/3/3b/Padmaavat_poster.jpg",
    "Andhadhun": "https://upload.wikimedia.org/wikipedia/en/3/3f/Andhadhun_poster.jpg",
    "Bahubali 2": "https://upload.wikimedia.org/wikipedia/en/7/7e/Baahubali_The_Conclusion.jpg",
    "Tanhaji": "https://upload.wikimedia.org/wikipedia/en/0/0f/Tanhaji_poster.jpg",
    "Chhichhore": "https://upload.wikimedia.org/wikipedia/en/3/33/Chhichhore_poster.jpg",
    "Queen": "https://upload.wikimedia.org/wikipedia/en/0/0c/Queen_2013_film_poster.jpg",
    "Barfi": "https://upload.wikimedia.org/wikipedia/en/1/15/Barfi%21_poster.jpg",
    "Raazi": "https://upload.wikimedia.org/wikipedia/en/3/3f/Raazi_film_poster.jpg",
    "Secret Superstar": "https://upload.wikimedia.org/wikipedia/en/7/79/Secret_Superstar_poster.jpg",
    "Gully Boy": "https://upload.wikimedia.org/wikipedia/en/3/33/Gully_Boy_film_poster.jpg",
    "Zindagi Na Milegi Dobara": "https://upload.wikimedia.org/wikipedia/en/c/c0/Zindagi_Na_Milegi_Dobara.jpg",
    "Yeh Jawaani Hai Deewani": "https://upload.wikimedia.org/wikipedia/en/4/45/Yeh_Jawaani_Hai_Deewani.jpg",
    "Kabir Singh": "https://upload.wikimedia.org/wikipedia/en/f/f9/Kabir_Singh_film_poster.jpg",
    "Jodhaa Akbar": "https://upload.wikimedia.org/wikipedia/en/0/01/Jodhaa_Akbar_film.jpg",
}

    # add more movies here with URLs

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
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['Movie Name'].iloc[movie_indices]

# ----------------------------
# Streamlit UI
# ----------------------------
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
                poster_url = movie_posters.get(movie, "")
                if poster_url:
                    st.image(poster_url, width=150)
                st.write(movie)
    else:
        st.warning("No similar movies found.")

