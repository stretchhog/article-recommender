__author__ = 'Stretchhog'


class Ensemble:
	def __init__(self):
		self.models = list()

	def register(self, model):
		"""

		:type model: Model
		"""
		self.models.append(model)
