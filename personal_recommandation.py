# File: recommender_system/project_pipeline.py

import pandas as pd
import numpy as np
from flask import Flask, request, render_template, flash, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import csr_matrix
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Week 1: Load and preprocess dataset
def load_movielens_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    data = ratings.merge(movies, on='movieId')
    return data

def preprocess_data(data):
    data = data.dropna()
    data['rating'] = data['rating'].astype(float)
    return data

def build_user_item_matrix(data):
    return data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Week 2: Collaborative filtering models
def user_based_cf(matrix):
    similarity = cosine_similarity(matrix)
    return pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

def item_based_cf(matrix):
    similarity = cosine_similarity(matrix.T)
    return pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)

def evaluate_collaborative(matrix, predicted):
    true_vals = matrix.values.flatten()
    pred_vals = predicted.values.flatten()
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    mae = mean_absolute_error(true_vals, pred_vals)
    return rmse, mae

# Matrix Factorization: SVD, NMF
def apply_svd(matrix):
    svd = TruncatedSVD(n_components=20)
    reduced = svd.fit_transform(matrix)
    approx = svd.inverse_transform(reduced)
    return pd.DataFrame(approx, index=matrix.index, columns=matrix.columns)

def apply_nmf(matrix):
    nmf = NMF(n_components=20, init='random', random_state=42)
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    approx = np.dot(W, H)
    return pd.DataFrame(approx, index=matrix.index, columns=matrix.columns)

# Week 3: Content-Based Filtering
def content_based_recommender(data):
    movie_metadata = data.drop_duplicates("movieId")[["movieId", "title"]]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_metadata['title'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cosine_sim, index=movie_metadata['movieId'], columns=movie_metadata['movieId'])

# Week 3: Hybrid Model
def hybrid_recommender(cf_matrix, content_matrix, alpha=0.5):
    return alpha * cf_matrix + (1 - alpha) * content_matrix

# Week 4: Recommendation Function
def get_top_k_recommendations(user_id, hybrid_matrix, k=10):
    if user_id not in hybrid_matrix.index:
        return []
    return hybrid_matrix.loc[user_id].sort_values(ascending=False).head(k).index.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        if user_id not in app.hybrid_matrix.index:
            flash(f"User ID {user_id} not found. Please try a different ID.", "error")
            return redirect(url_for('home'))

        top_movies = get_top_k_recommendations(user_id, app.hybrid_matrix, 10)
        movie_titles = app.movies[app.movies['movieId'].isin(top_movies)]['title'].tolist()
        return render_template('index.html', recommendations=movie_titles, user_id=user_id)

    except ValueError:
        flash("Invalid input. Please enter a numeric user ID.", "error")
        return redirect(url_for('home'))

@app.before_request
def init_model():
    raw_data = load_movielens_data()
    cleaned = preprocess_data(raw_data)
    user_item = build_user_item_matrix(cleaned)

    svd_pred = apply_svd(user_item)
    content_sim = content_based_recommender(cleaned)

    app.hybrid_matrix = hybrid_recommender(svd_pred, content_sim)
    app.movies = pd.read_csv('movies.csv')

if __name__ == '__main__':
    app.run(debug=True)
