from enum import Enum
from recommender.features.FeatureManager import ArticleFeature

from recommender.model.Model import Model

__author__ = 'Stretchhog'


class Ensemble(Model):
	def get_type(self):
		raise NotImplementedError("ensemble does not have a type")

	def __init__(self, mode, models):
		if mode is Mode.MAJORITY_AVG and len(models) % 2 == 0:
			raise AttributeError('An even number of models cannot always reach a majority vote!')

		self.mode = mode
		self.models = models

	def train(self, x, y):
		for model in self.models:
			model.train(x, y)

	def score(self, doc, x):
		scores = []
		for model in self.models:
			scores.append(model.score(doc, x))

		likelihood = 0
		if self.mode is Mode.MAJORITY_AVG:
			pass
		elif self.mode is Mode.GLOBAL_AVG:
			likelihood = sum(scores) / len(scores)
		return likelihood

	def persist(self):
		for model in self.models:
			mid = self.db.save_model(model)
			print(mid)


class Mode(Enum):
	MAJORITY_AVG = 1
	GLOBAL_AVG = 2
