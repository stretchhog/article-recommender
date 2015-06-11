import datetime
from pymongo import MongoClient

__author__ = 'Stretchhog'

client = MongoClient()

db = client['recommender-db']
models = db['models']

models.delete_many({"type": "NaiveBayes"})

new_model = {"type": "NaiveBayes",
             "model": "model",
             "data": datetime.datetime.utcnow()
             }

new_model_id = models.insert_one(new_model).inserted_id
print(new_model_id)

found_models = models.find({"type": "NaiveBayes"})

for model in found_models:
	print(model)
