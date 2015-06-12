import os
import pickle
from recommender.persistence.DataStore import DataStore

__author__ = 'Stretchhog'


class PickleDS(DataStore):
	def save(self, name, data):
		pickle.dump(data, open(name, "wb"))

	def load(self, name):
		if os.path.exists(name):
			return pickle.load(open(name, "rb"))
		else:
			return None
