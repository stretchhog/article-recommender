import numpy as np
from collections import Counter
from recommender.features.FeatureManager import ArticleFeature
from recommender.model.Model import Model

__author__ = 'Stretchhog'


class NaiveBayes(Model):
	def __init__(self):
		self.bins = []
		self.y = []
		self.x = []

	def train(self, x, y):
		self.bins = [f.get_train_for_nb() for f in x]
		self.y = y
		self.x = x

	def __bin_x(self, x, y):
		bins = [[], [], [], [], []]
		bins[ArticleFeature.TOPIC.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.TOPIC.value[0]], y)
		bins[ArticleFeature.ORIGIN.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.ORIGIN.value[0]], y)
		bins[ArticleFeature.AUTHOR.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.AUTHOR.value[0]], y)
		bins[ArticleFeature.SENTIMENT.value[0]] = np.histogram(x[ArticleFeature.SENTIMENT.value[0]], 100)
		bins[ArticleFeature.TFIDF.value[0]] = [np.histogram(column, 100) for column in
		                                       x[ArticleFeature.TFIDF.value[0]].T]
		return bins

	def __bin_categoric_feature(self, x, y):
		bin = {}
		for value, label in zip(x, y):
			if value in bin:
				if label:
					bin[value]['+'] += 1
				else:
					bin[value]['-'] += 1
			else:
				bin[value] = {}
				bin[value]['+'] = 0
				bin[value]['-'] = 0
				if label:
					bin[value]['+'] = 1
				else:
					bin[value]['-'] = 1
		return bin

	def score(self, doc, x):
		prob_neg, prob_pos = self.__get_priors()

		pos = []
		neg = []
		for i in range(0, len(doc)):
			_pos, _neg = x[i].get_score_for_nb(doc[i], self.bins[i])
			pos += _pos
			neg += _neg
		for p, n in zip(pos, neg):
			prob_pos *= self.__ratio(p, n)
			prob_neg *= self.__ratio(n, p)
		return prob_pos / (prob_neg + prob_pos)

	def __ratio(self, a, b):
		return a + 1 / (a + b + 2)

	def __get_priors(self):
		total = len(self.y)
		all_pos = np.count_nonzero(self.y)
		all_neg = total - all_pos
		return self.__ratio(all_neg, all_pos), self.__ratio(all_pos, all_neg)

	def __labels_for_range(self, data, lo, hi):
		if hi is not None:
			documents_in_range = np.logical_and(lo < data, data <= hi)
		else:
			documents_in_range = lo < self.x

		pos = np.count_nonzero(self.y[documents_in_range])
		neg = data.shape[0] - pos
		return pos, neg
