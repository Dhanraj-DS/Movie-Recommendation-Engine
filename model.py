import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movie_df = pd.read_csv("Movies Data Set.csv")


movie_df['combined_features'] = movie_df['genre'] + ' ' + movie_df['act'] + ' ' + movie_df['Plot']
tfid = TfidfVectorizer(stop_words='english')
vectors = tfid.fit_transform(movie_df['combined_features'])

indexed = pd.Series(index=movie_df["Movie Title"], data=movie_df.index)

dis = linear_kernel(vectors[0], vectors)
scores = pd.DataFrame(dis).T

x = scores.sort_values(by=0, ascending=False)

def movie_recommendation(name, n):
    indices = movie_df[movie_df['Movie Title'].str.lower() == name.lower()].index
    if len(indices) == 0:
        return []  # Return empty list if not found
    idx = indices[0]
    dis = linear_kernel(vectors[idx], vectors)
    scores = pd.DataFrame(dis).T
    scores.columns = ["score"]
    sorted_scores = scores.sort_values(by="score", ascending=False)

    recommendation = []
    for i in range(n):
        movie_idx = sorted_scores.index[i+1]
        s = {
            "Movie_title": movie_df["Movie Title"][movie_idx],
            "genre": movie_df["genre"][movie_idx],
            "Rating": movie_df["Ratinng"][movie_idx],
            "Actors": movie_df["act"][movie_idx]
        }
        recommendation.append(s)
    return recommendation



#streamlit UI
import streamlit as st
st.title("Movie Recommendation System")
movie_name = st.text_input('Enter your favorite movie:')
num_recs = st.slider('How many recommendations?', 1, 30, 5)
tmdb_api_key = st.text_input('TMDB API Key (optional, for posters):', type='password')

if movie_name:
    st.write(f"Recommendations for '{movie_name}':")
    recommendations = get_recommendations(movie_name, num_recs)
    if recommendations.empty:
        st.error("Movie not found! Please try another title.")
    else:
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"**Genres**: {row['genres']}")
            st.write(f"{row['overview']}")
            if tmdb_api_key:
                poster_url = fetch_poster(row['title'], tmdb_api_key)
                if poster_url:
                    st.image(poster_url, width=200)