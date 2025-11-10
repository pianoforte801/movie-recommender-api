from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS   # 👈 new import

app = Flask(__name__)
CORS(app)                     # 👈 enable all origins


# Load the data once when the server starts
print("Loading data...")
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Create user–movie matrix
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

print("Data loaded — server ready.")

@app.route('/')
def home():
    return jsonify({
        "message": "Movie Recommender API (Memory-Optimized)",
        "usage": "/recommend?movie=Toy Story&num=10"
    })

@app.route('/recommend')
def recommend():
    movie_title = request.args.get('movie', '')
    num = int(request.args.get('num', 10))

    if not movie_title:
        return jsonify({"error": "Please provide a movie title"}), 400

    # Find movie
    movie_match = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
    if movie_match.empty:
        return jsonify({"error": f"Movie '{movie_title}' not found"}), 404

    movie_id = movie_match.iloc[0]['movieId']
    movie_name = movie_match.iloc[0]['title']

    # Check if movie exists in matrix
    if movie_id not in user_movie_matrix.columns:
        return jsonify({"error": f"Movie '{movie_name}' has no ratings data"}), 404

    # ✅ Compute similarity on demand — saves 700 MB of memory
    movie_vector = user_movie_matrix[movie_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(movie_vector, user_movie_matrix.T)[0]

    # Get top N similar movies
    similar_indices = similarity_scores.argsort()[::-1][1:num + 1]
    similar_movie_ids = user_movie_matrix.columns[similar_indices]

    # Get details
    recommendations = movies[movies['movieId'].isin(similar_movie_ids)][['title', 'genres']].to_dict('records')

    return jsonify({
        "query": movie_name,
        "recommendations": recommendations
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
