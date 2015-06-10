from enum import Enum

from recommender.model.Model import Model

__author__ = 'Stretchhog'


class Ensemble(Model):
	def __init__(self, mode, models):
		self.models = list()
		if mode is Mode.MAJORITY_AVG and len(models) % 2 == 0:
			raise AttributeError('An even number of models cannot always reach a majority vote!')

		self.mode = mode
		for model in models:
			self.__register(model)

	def __register(self, model):
		"""

		:type model: Model
		"""
		self.models.append(model)

	def train(self, x, y):
		for model in self.models:
			model.train(x, y)

	def score(self, x):
		scores = []
		for model in self.models:
			scores.append(model.score(x))

		likelihood = 0
		if self.mode is Mode.MAJORITY_AVG:
			pass
		elif self.mode is Mode.GLOBAL_AVG:
			likelihood = sum(scores) / len(scores)
		return likelihood


class Mode(Enum):
	MAJORITY_AVG = 1
	GLOBAL_AVG = 2
