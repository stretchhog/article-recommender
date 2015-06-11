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
			"type": type(model),
			"model": model.get_model()
			"date": datetime.datetime.utcnow()
		}
		return self.models.insert_one(new_model).inserted_id

found_models = models.find({"type": "NaiveBayes"})

for model in found_models:
	print(model)
