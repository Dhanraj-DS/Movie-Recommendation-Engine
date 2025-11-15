import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load Dataset
movie_df = pd.read_csv("Movies Data Set.csv")

# Combine text features
movie_df['combined_features'] = (
    movie_df['genre'] + ' ' + movie_df['act'] + ' ' + movie_df['Plot']
)

tfid = TfidfVectorizer(stop_words='english')
vectors = tfid.fit_transform(movie_df['combined_features'])

# Index mapping
indexed = pd.Series(index=movie_df["Movie Title"], data=movie_df.index)

# Recommendation function
def movie_recommendation(name, n):
    indices = movie_df[movie_df['Movie Title'].str.lower() == name.lower()].index
    if len(indices) == 0:
        return []

    idx = indices[0]

    dis = linear_kernel(vectors[idx], vectors)
    scores = pd.DataFrame(dis).T
    scores.columns = ["score"]
    sorted_scores = scores.sort_values(by="score", ascending=False)

    recommendation = []
    count = min(n, len(sorted_scores) - 1)

    for i in range(1, count + 1):
        movie_idx = sorted_scores.index[i]
        s = {
            "Movie_title": movie_df["Movie Title"][movie_idx],
            "genre": movie_df["genre"][movie_idx],
            "Ratinng": movie_df["Ratinng"][movie_idx],   # FIXED SPELLING
            "Actors": movie_df["act"][movie_idx]
        }
        recommendation.append(s)
    return recommendation

# Streamlit UI
st.title("Movie Recommendation System")

movie_name = st.text_input('Enter your favorite movie:')
num_recs = st.slider('How many recommendations?', 1, 30, 5)

if movie_name:
    recommendation = movie_recommendation(movie_name, num_recs)

    if len(recommendation) == 0:
        st.error("Movie not found or no recommendations available.")
    else:
        for movie in recommendation:
            st.write(f"### 🎬 {movie['Movie_title']}")
            st.write(f"- **Ratinng:** {movie['Ratinng']}")
            st.write(f"- **Genre:** {movie['genre']}")
            st.write(f"- **Actors:** {movie['Actors']}")
            st.write("---")