import numpy as np
from articles.model.Model import Model

__author__ = 'Stretchhog'


class NaiveBayes(Model):
	def __init__(self):
		self.bins = []
		self.edges = []
		self.y = []
		self.x = []

	def train(self, x, y):
		self.bins, self.edges = [np.histogram(column, 100) for column in x.T]
		self.y = y
		self.x = x

	def score(self, x):
		prob_neg, prob_pos = self.get_priors()
		for i in range(1, len(x)):
			if x[i] is None:
				continue
			index = np.searchsorted(self.bins[i], x[i])
			pos, neg = self.labels_for_range(self.edges[index], self.edges[index + 1] if index < len(self.edges) else None)
			prob_pos *= pos / (pos + neg)
			prob_neg *= neg / (pos + neg)

		return prob_pos / (prob_neg + prob_pos)

	def get_priors(self):
		total = len(self.y)
		all_pos = np.count_nonzero(self.y)
		all_neg = total - all_pos
		return (all_neg / total), (all_pos / total)

	def labels_for_range(self, lo, hi):
		if hi is not None:
			documents_in_range = lo < self.x <= hi
		else:
			documents_in_range = lo < self.x

		pos = np.count_nonzero(self.y[documents_in_range])
		neg = pos - self.x.shape[0]
		return pos, neg
