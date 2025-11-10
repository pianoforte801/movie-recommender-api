import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

#create user-movie matrix (rows=users columns=movies , values=ratings)
user_movie_matrix=ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'

).fillna(0) #fill missing ratings with 0

print("user-movie matrix shape:",user_movie_matrix.shape)
print("\n first 5 users x first 5 movies:")
print(user_movie_matrix.iloc[:5,:5])

#calculate similarity btw all movies
movie_similarity=cosine_similarity(user_movie_matrix.T)
print("\n movie similarity matrix shape:", movie_similarity.shape )

def recommend_movies(movie_title,num_recommendations=10):
    '''
    given a movie title recommend similar movies
    '''

    #find the movie id
    movie_match=movies[movies['title'].str.contains(movie_title,case=False,na=False)]

    if movie_match.empty:
        return f"movie {movie_title} not found!"
    movie_id=movie_match.iloc[0]['movieId']
    movie_name=movie_match.iloc[0]['title']

    #get the index of this movie in our similarity matrix
    movie_idx=user_movie_matrix.columns.get_loc(movie_id)

    #get similarity scored for this movie with all others
    similarity_scores=movie_similarity[movie_idx]

    #get indices of most similar movies(sorted)
    similar_indices=similarity_scores.argsort()[::-1][1:num_recommendations+1]

    #get the movie ids
    similar_movie_ids=user_movie_matrix.columns[similar_indices]

    #get movie details
    recommendations=movies[movies['movieId'].isin(similar_movie_ids)][['title','genres']]

    print(f"\n because you liked {movie_name} :")
    print("="*60)
    for idx,row in recommendations.iterrows():
        print(f"-> {row['title']}")
        print(f"     Genres: {row['genres']}\n")
    return recommendations

recommend_movies("Toy story")