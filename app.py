# app.py
import streamlit as st
import pandas as pd

# --- Load dataset ---
@st.cache_data
# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("bollywood_movies_full.csv")  # Your CSV
    # Fill missing values
    df['Lead Star'] = df['Lead Star'].fillna('')
    df['Genre'] = df['Genre'].fillna('')
    return df  # No need for Region filter

movies = load_data()

# --- Recommendation function ---
def recommend_movies(movie_name, num_recommendations=10):
    movie_name = movie_name.strip()
    if movie_name not in movies['Movie Name'].values:
        st.warning("Movie not found! Check spelling or try another movie.")
        return pd.DataFrame()
    
    # Get selected movie info
    selected_movie = movies[movies['Movie Name'] == movie_name].iloc[0]
    lead_star = selected_movie['Lead Star']
    genre = selected_movie['Genre'].split(',')[0]  # pick main genre
    
    # Filter movies with same lead actor and similar genre
    recs = movies[
        (movies['Lead Star'] == lead_star) & 
        (movies['Genre'].str.contains(genre)) & 
        (movies['Movie Name'] != movie_name)  # exclude the selected movie
    ]
    
    return recs.head(num_recommendations)

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Bollywood Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

movie_input = st.text_input("Enter a Bollywood movie name:")
num_rec = st.slider("Number of recommendations:", 5, 15, 10)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_input, num_rec)
    if not recommendations.empty:
        st.subheader(f"Movies starring the same lead actor as '{movie_input}':")
        cols = st.columns(3)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            col = cols[i % 3]
            with col:
                poster = row['Poster URL'] if row['Poster URL'] else "https://via.placeholder.com/150x220?text=No+Image"
                st.image(poster, width=150)
                st.markdown(f"**{row['Movie Name']}**")
                st.markdown(f"⭐ Rating: {row['Rating']} | 📅 Year: {row['Release Year']}")
                st.markdown(f"🎭 Genre: {row['Genre']}")
    else:
        st.info("No similar movies found for this lead actor and genre.")

