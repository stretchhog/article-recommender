import pandas as pd
import numpy as np

__author__ = 'Stretchhog'


def load_data():
	users = pd.read_table('../../data/ml-1m/users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip'])
	ratings = pd.read_table('../../data/ml-1m/ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
	movies = pd.read_table('../../data/ml-1m/movies.dat', sep='::', engine='python', header=None, names=['movie_id', 'title', 'genres'])
	return users, ratings, movies
