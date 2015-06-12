import numpy as np

from recommender.model.Model import Model

__author__ = 'Stretchhog'


class NaiveBayes(Model):
	def __init__(self):
		self.data_cache = []
		self.y = []
		self.x = []

	def train(self, x, y):
		self.data_cache = [np.histogram(column, 100) for column in x.T]
		self.y = y
		self.x = x

	def score(self, x):
		prob_neg, prob_pos = self.__get_priors()
		for i in range(1, x.shape[1]):
			if x[0, i] == 0.:
				continue
			index = np.searchsorted(self.data_cache[i][0], x[0, i])
			pos, neg = self.__labels_for_range(self.x[:, i], self.data_cache[i][1][index], self.data_cache[i][1][index + 1] if index < len( self.data_cache[i][1]) else None)
			prob_pos *= pos / (pos + neg)
			prob_neg *= neg / (pos + neg)

		return prob_pos / (prob_neg + prob_pos)

	def __get_priors(self):
		total = len(self.y)
		all_pos = np.count_nonzero(self.y)
		all_neg = total - all_pos
		return (all_neg / total), (all_pos / total)

	def __labels_for_range(self, data, lo, hi):
		if hi is not None:
			documents_in_range = np.logical_and(lo < data, data <= hi)
		else:
			documents_in_range = lo < self.x

		pos = np.count_nonzero(self.y[documents_in_range])
		neg = data.shape[0] - pos
		return pos, neg

	def get_type(self):
		return "NaiveBayes"
