import streamlit as st
import pandas as pd
from difflib import get_close_matches

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("bollywood_movies.csv")
    # Fill missing values
    if 'Lead Star' not in df.columns:
        df['Lead Star'] = ''   # fallback if dataset has no lead star
    if 'Genres' not in df.columns:
        df['Genres'] = ''      # fallback if dataset has no genres
    if 'Overview' not in df.columns:
        df['Overview'] = ''    # fallback if dataset has no overview
    return df

movies = load_data()

# --- Find closest movie match ---
def find_movie(title, movies):
    all_titles = movies['Movie Name'].astype(str).str.lower().tolist()
    matches = get_close_matches(title.lower(), all_titles, n=1, cutoff=0.6)
    if matches:
        return matches[0]  # best fuzzy match
    return None

# --- Recommendation function ---
def recommend(movie_name, top_n=10):
    movie_key = find_movie(movie_name, movies)
    if not movie_key:
        return None, "Movie not found! Check spelling or try another."

    matches = movies[movies['Movie Name'].str.lower() == movie_key]
    if matches.empty:
        return None, "Movie not found in dataset."
    
    movie = matches.iloc[0]
    lead_star = movie.get('Lead Star', '')
    genres = movie.get('Genres', '')

    # filter by same lead star
    same_star = movies[movies['Lead Star'] == lead_star] if lead_star else pd.DataFrame()

    # filter by same genre (safe contains)
    if isinstance(genres, str) and genres.strip():
        first_genre = genres.split(",")[0].strip()
        same_genre = movies[movies['Genres'].astype(str).str.contains(first_genre, case=False, na=False, regex=False)]
    else:
        same_genre = pd.DataFrame()

    # combine
    recommendations = pd.concat([same_star, same_genre]).drop_duplicates()

    # remove input movie itself
    recommendations = recommendations[movies['Movie Name'].str.lower() != movie_key]

    if recommendations.empty:
        return None, "No similar movies found for this lead actor or genre."

    return recommendations.head(top_n), None

# --- Streamlit UI ---
st.set_page_config(page_title="Bollywood Recommender", layout="wide")

st.title("🎬 Bollywood Movie Recommender")
st.write("Get recommendations based on **Lead Star** and **Genre**")

movie_input = st.text_input("Enter a Bollywood movie name:")

if st.button("Recommend"):
    if movie_input:
        results, error = recommend(movie_input)
        if error:
            st.warning(error)
        else:
            st.success(f"Recommendations for **{movie_input.title()}**:")
            cols = st.columns(2)  # two-column layout
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.subheader(row['Movie Name'])
                    if 'Poster URL' in row and pd.notna(row['Poster URL']):
                        st.image(row['Poster URL'], width=200)
                    st.write(f"⭐ Lead Star: {row['Lead Star']}")
                    st.write(f"🎭 Genre: {row['Genres']}")
                    if 'Overview' in row and isinstance(row['Overview'], str):
                        st.caption(row['Overview'])
    else:
        st.info("Please enter a movie name.")



