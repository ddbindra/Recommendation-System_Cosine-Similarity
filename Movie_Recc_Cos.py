import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

user_cols = ['user_id','age','sex','occupation','zipcode']
users = pd.read_csv('./ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1',parse_dates=True)
rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./ml-100k/u.data', sep='\t', names=rating_cols, encoding='latin-1')
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('./ml-100k/u.item', sep='|', names=movie_cols, usecols=range(5),encoding='latin-1')
movie_rating = pd.merge(movies,ratings)
df = pd.merge(movie_rating,users)
df.drop(df.columns[[3,4,7]],axis=1,inplace=True)
ratings.drop("unix_timestamp",axis=1,inplace=True)
movies.drop(movies.columns[[3,4]],axis=1,inplace=True)
movie_stats = df.groupby('title').agg({'rating': [np.size, np.mean]})
min_50=movie_stats['rating']['size'] >= 50
rating_matrix = ratings.pivot_table(index=['movie_id'],columns=['user_id'],values='rating').reset_index(drop=True)
rating_matrix.fillna(0,inplace=True)
movie_similarity = 1 - pairwise_distances( ratings_matrix, metric="cosine" )
np.fill_diagonal(movie_similarity,0)
rating_matrix = pd.DataFrame( movie_similarity )
try:
    user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
    inp = movies[movies['title'] == user_inp].index.tolist()
    inp = inp[0]

    movies['similarity'] = ratings_matrix.iloc[inp]
    movies.columns = ['movie_id', 'title', 'release_date', 'similarity']
    print("Recommended movies based on your choice of ", user_inp, ": \n",
          movies.sort_values(["similarity"], ascending=False)[1:10])

except:
    print("Sorry, the movie is not in the database!")