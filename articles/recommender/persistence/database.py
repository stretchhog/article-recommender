import datetime
from pymongo import MongoClient

__author__ = 'Stretchhog'

client = MongoClient()


class Database(object):
	def __init__(self):
		db = client['recommender-db']
		self.models = db['models']
		self.models.delete_many({"type": "NaiveBayes"})

	def save_model(self, model):
		new_model = {
			"type": model.get_type(),
			"model": model,
			"date": datetime.datetime.utcnow()
		}
		return self.models.find_one_and_replace({"type": type}, new_model).inserted_id

	def load_model(self, model_type):
		return self.models.find_one({"type": model_type})
