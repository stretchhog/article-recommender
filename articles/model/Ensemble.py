from articles.model.Model import Model

__author__ = 'Stretchhog'


class Ensemble(Model):
	def __init__(self):
		self.models = list()

	def register(self, model):
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
		return sum(scores) / len(scores)

