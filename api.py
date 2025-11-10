from flask import Flask,request,jsonify 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

#load the data once when the server starts
print("loading data")
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_movie_matrix=ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

movie_similarity=cosine_similarity(user_movie_matrix)

print("data loaded server is ready")

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
    
    # Check if movie exists in our matrix
    if movie_id not in user_movie_matrix.columns:
        return jsonify({"error": f"Movie '{movie_name}' has no ratings data"}), 404
    
    # Get recommendations
    movie_idx = user_movie_matrix.columns.get_loc(movie_id)
    similarity_scores = movie_similarity[movie_idx]
    similar_indices = similarity_scores.argsort()[::-1][1:num+1]
    similar_movie_ids = user_movie_matrix.columns[similar_indices]
    
    recommendations = movies[movies['movieId'].isin(similar_movie_ids)][['title', 'genres']].to_dict('records')
    
    return jsonify({
        "query": movie_name,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)