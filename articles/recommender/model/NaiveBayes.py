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
		self.bins = self.__bin_x(x, y)
		self.y = y
		self.x = x

	def __bin_x(self, x, y):
		bins = [[], [], [], [], []]
		bins[ArticleFeature.TOPIC.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.TOPIC.value[0]], y)
		bins[ArticleFeature.ORIGIN.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.ORIGIN.value[0]], y)
		bins[ArticleFeature.AUTHOR.value[0]] = self.__bin_categoric_feature(x[ArticleFeature.AUTHOR.value[0]], y)
		bins[ArticleFeature.SENTIMENT.value[0]] = np.histogram(x[ArticleFeature.SENTIMENT.value[0]], 100)
		bins[ArticleFeature.TFIDF.value[0]] = [np.histogram(column, 100) for column in x[ArticleFeature.TFIDF.value[0]].T]
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

	def score(self, x):
		prob_neg, prob_pos = self.__get_priors()

		pos, neg = self.__get_probs_for_categoric(self.bins[ArticleFeature.ORIGIN.value[0]], x[ArticleFeature.ORIGIN.value[0]])
		prob_pos *= self.__ratio(pos, neg)
		prob_neg *= self.__ratio(neg, pos)

		pos, neg = self.__get_probs_for_categoric(self.bins[ArticleFeature.AUTHOR.value[0]], x[ArticleFeature.AUTHOR.value[0]])
		prob_pos *= self.__ratio(pos, neg)
		prob_neg *= self.__ratio(neg, pos)

		pos, neg = self.__get_probs_for_categoric(self.bins[ArticleFeature.TOPIC.value[0]], x[ArticleFeature.TOPIC.value[0]])
		prob_pos *= self.__ratio(pos, neg)
		prob_neg *= self.__ratio(neg, pos)

		sentiment_index = ArticleFeature.SENTIMENT.value[0]
		pos, neg = self.__get_probs_for_numeric(self.bins[sentiment_index], x[sentiment_index], self.x[sentiment_index])
		prob_pos *= self.__ratio(pos, neg)
		prob_neg *= self.__ratio(neg, pos)

		for i in range(1, x.shape[1]):
			if x[0, i] == 0.:
				continue
			index = np.searchsorted(self.data_cache[i][0], x[0, i])
			pos, neg = self.__labels_for_range(self.x[:, i], self.data_cache[i][1][index], self.data_cache[i][1][index + 1] if index < len(self.data_cache[i][1]) else None)
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

	def __get_probs_for_categoric(self, bins, value):
		if value not in bins:
			return None, None
		return bins[value]['+'], bins[value]['-']

	def __get_probs_for_numeric(self, bins, value, x):
		index = np.searchsorted(bins[1], value)
		pos, neg = self.__labels_for_range(x, bins[1][index], bins[1][index + 1] if index < len(bins[1]) else None)

	def __ratio(self, x, y):
		if x is not None and y is not None:
			return x / (x + y)
		return 1
