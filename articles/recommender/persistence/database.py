import datetime
import subprocess
from pymongo import MongoClient

__author__ = 'Stretchhog'

client = MongoClient()


class Database(object):
	def __init__(self):
		subprocess.Popen(['C:/tools/mongodb/bin/mongod'])
		# subprocess.Popen(['C:/tools/mongodb/bin/mongod', '----dbpath C:\\dropbox\\projects\\mongodb'])
		db = client['recommender-db']
		self.models = db['models']
		self.models.delete_many({"type": "NaiveBayes"})

	def save(self, name, data):
		self.models.find_one_and_replace({"name": name}, data)

	def load(self, name):
		return self.models.find_one({"name": name})
