from bisect import bisect_left
from model.Model import Model
import numpy as np

__author__ = 'Stretchhog'


class NaiveBayes(Model):
	def __init__(self, feature_manager):
		"""

		:type feature_manager: FeatureManager.FeatureManager
		"""
		self.feature_manager = feature_manager

	def train(self, document):
		pass

	def score(self, document_features):
		binned, edges = [np.histogram(column, 100) for column in self.feature_manager.get_x().T]
		prob_neg, prob_pos = self.get_priors()
		for x in range(1, len(document_features)):
			index = np.searchsorted(binned[x], document_features[x])
			pos, neg = self.labels_for_range(edges[index], edges[index + 1] if index < len(edges) else None)
			prob_pos *= pos / (pos + neg)
			prob_neg *= neg / (pos + neg)

		post_pos = prob_pos / (prob_neg + prob_pos)
		post_neg = prob_neg / (prob_neg + prob_pos)

	def get_priors(self):
		total = len(self.feature_manager.get_y())
		all_pos = np.count_nonzero(self.feature_manager.get_y())
		all_neg = total - all_pos
		return (all_neg / total), (all_pos / total)

	def labels_for_range(self, lo, hi):
		if hi is not None:
			documents_in_range = lo < self.feature_manager.get_x() <= hi
		else:
			documents_in_range = lo < self.feature_manager.get_x()

		pos = np.count_nonzero(self.feature_manager.get_y()[documents_in_range])
		neg = pos - self.feature_manager.feature_dimensions()[0]
		return pos, neg
