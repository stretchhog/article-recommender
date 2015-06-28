import pandas as pd
import numpy as np
from service.LearningService import LearningService

__author__ = 'Stretchhog'


def load_data():
	users = pd.read_table('../../data/ml-1m/users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip'])
	ratings = pd.read_table('../../data/ml-1m/ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
	movies = pd.read_table('../../data/ml-1m/movies.dat', sep='::', engine='python', header=None, names=['movie_id', 'title', 'genres'])
	return users, ratings, movies

service = LearningService()
u, r, m = load_data()

msk = np.random.rand(len(r)) < 0.8
r_train = r[msk]
r_test = r[~msk]

for index, x in r_train.iterrows():
	prediction = service.predict(x['user_id'], x['movie_id'])
	service.save_reading_features(x['user_id'], x['movie_id'], x['timestamp'])
	service.feedback(x['user_id'], x['movie_id'], prediction, x['rating'])
