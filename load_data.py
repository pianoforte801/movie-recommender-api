import pandas as pd

# Load the data from subfolder
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Explore
print("Movies shape:", movies.shape)
print("\nFirst 5 movies:")
print(movies.head())

print("\nRatings shape:", ratings.shape)
print("\nFirst 5 ratings:")
print(ratings.head())

print("\n data explaoration")
print(f"unique users: {ratings['userId'].nunique()}")#count unique values
print(f"unique movies rating: {ratings['movieId'].nunique()}")#count unique movies
print(f"average rating: {ratings['rating'].mean():.2f}") #mean calculation
print(f"most rated movie id: {ratings['movieId'].value_counts().head(1)}") #frequency distribution

#find what that movie is
most_rated_id=ratings['movieId'].value_counts().index[0]
most_rated_movie=movies[movies['movieId']==most_rated_id]
print(f"most rated movie:\n{most_rated_movie}")
