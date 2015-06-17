from itertools import chain
import numpy as np
from sklearn import svm

from recommender.model.Model import Model

__author__ = 'Stretchhog'


class SupportVectorMachines(Model):
	def __init__(self):
		self.model = svm.SVC(kernel='linear', probability=True)

	def score(self, doc, x, y=None):
		all_features = [feature.get_score_for_svm(value) for feature, value in zip(x, doc)]
		concat = np.concatenate((all_features[0], all_features[1]), axis=1)
		for i in range(2, len(all_features)):
			concat = np.concatenate((concat, all_features[i]), axis=1)
		return self.model.predict_proba(concat)[0][0]

	def train(self, x, y):
		all_features = [feature.get_train_for_svm() for feature in x]
		concat = np.concatenate((all_features[0], all_features[1]), axis=1)
		for i in range(2, len(all_features)):
			concat = np.concatenate((concat, all_features[i]), axis=1)
		self.model = self.model.fit(concat, y)

	def get_type(self):
		return "SVM"
