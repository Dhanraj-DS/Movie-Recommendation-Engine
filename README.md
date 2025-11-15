# 🎬 Movie Recommendation System (NLP + Machine Learning)

## 🚀 Overview
This project is a **Content-Based Movie Recommendation System** built using **TF-IDF vectorization**, **cosine similarity**, and **Streamlit**.  
It analyzes movie metadata (genre, actors, and plot summaries) to recommend movies similar to a user-selected title.

This system demonstrates how modern streaming platforms apply NLP to improve content discovery.

---

## 🧠 How It Works

### 1. Data Preprocessing
- Loaded a movie dataset containing:
  - `Movie Title`
  - `genre`
  - `act`
  - `Plot`
  - `Ratinng`
- Combined text columns into a unified feature string.

### 2. Feature Engineering
- Applied **TF-IDF Vectorizer** to convert text to numerical vectors.
- Calculated similarity between movies using **linear kernel (cosine similarity)**.

### 3. Recommendation Logic
- Locate user-selected movie.
- Rank all movies by similarity score.
- Return **Top-N recommendations** with genre, rating, and actor info.

### 4. User Interface (Streamlit)
- Provides simple input box for movie name.
- Slider to choose number of recommendations.
- Displays results in clean, readable layout.

---

## 🛠 Tech Stack

### 🔹 Languages & Frameworks
- Python 3
- Streamlit

### 🔹 Libraries
- Pandas  
- Scikit-Learn  
- NumPy  

### 🔹 ML & NLP
- TF-IDF Vectorizer  
- Content-Based Filtering  
- Cosine Similarity  

---

## 📸 Screenshots  
(Add after running your Streamlit app)

---

## 📂 Project Structure

