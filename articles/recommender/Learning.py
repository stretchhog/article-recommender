import uuid
from recommender.persistence.PickleDS import PickleDS
from recommender.pipeline import Pipeline

__author__ = 'Stretchhog'


class Learning(object):
	def __init__(self):
		self.prediction_cache = {}
		self.predictive_model = Pipeline(PickleDS())

	def predict(self, document):
		likelihood = self.predictive_model.score(document)
		prediction_id = uuid.uuid4
		self.prediction_cache[prediction_id] = (likelihood, document)
		return prediction_id, likelihood

	def feedback(self, prediction_id, label):
		if prediction_id in self.prediction_cache:
			self.predictive_model.train(self.prediction_cache[prediction_id][1], label)
