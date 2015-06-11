from sklearn import svm

from recommender.model.Model import Model

__author__ = 'Stretchhog'


class SupportVectorMachines(Model):
	def __init__(self):
		self.model = svm.SVC(kernel='linear', probability=True)

	def score(self, x):
		return self.model.predict_proba(x)[0][0]

	def train(self, x, y):
		self.model = self.model.fit(x, y)

	def get_model(self):
		return self.model
