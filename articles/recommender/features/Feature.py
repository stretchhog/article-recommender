import abc
import numpy as np

__author__ = 'Stretchhog'


class Feature(object):
	@abc.abstractmethod
	def update(self, value):
		return

	@abc.abstractmethod
	def get_train_for_nb(self):
		return

	@abc.abstractmethod
	def get_score_for_nb(self, value, bins):
		return

	@abc.abstractmethod
	def get_train_for_svm(self):
		return

	@abc.abstractmethod
	def get_score_for_svm(self, values):
		return


class CategoricFeature(Feature):
	def __init__(self, feature_manager):
		self.values = []
		self.feature_manager = feature_manager
		self.uniques = []

	def update(self, value):
		self.values.append(value)

	def get_train_for_nb(self):
		bins = {}
		for value, label in zip(self.values, self.feature_manager.y):
			if value in bins:
				if label:
					bins[value]['+'] += 1
				else:
					bins[value]['-'] += 1
			else:
				bins[value] = {}
				bins[value]['+'] = 0
				bins[value]['-'] = 0
				if label:
					bins[value]['+'] = 1
				else:
					bins[value]['-'] = 1
		return bins

	def get_score_for_nb(self, value, bins):
		if value not in bin:
			return None, None
		return bins[value]['+'], bins[value]['-']

	def get_train_for_svm(self):
		self.uniques = list(set(self.values))
		features = np.zeros((len(self.values), len(self.uniques)))
		i = 0
		for value in self.values:
			features[i][self.uniques.index(value)] = 1
			i += 1
		return features

	def get_score_for_svm(self, value):
		features = np.zeros((1, len(self.uniques)))
		if value in self.uniques:
			features[0][self.values.index(value)] = 1
		return features


class NumericFeature(Feature):
	def __init__(self, feature_manager):
		self.values = np.zeros((1, 1))
		self.feature_manager = feature_manager

	def update(self, value):
		if self.values.shape == (1, 1):
			self.values = value
		else:
			self.values = np.vstack((self.values, value))

	def get_train_for_nb(self):
		return np.histogram(self.values, 100)

	def get_score_for_nb(self, value, bins):
		index = np.searchsorted(bins[1], value)
		pos, neg = labels_for_range(self.values, self.feature_manager.y, bins[1][index],
		                            bins[1][index + 1] if index < len(bins[1]) else None)
		return pos, neg

	def get_train_for_svm(self):
		return self.values

	def get_score_for_svm(self, value):
		return value


def labels_for_range(x, y, lo, hi):
	if hi is not None:
		documents_in_range = np.logical_and(lo < x, x <= hi)
	else:
		documents_in_range = lo < x

	pos = np.count_nonzero(y[documents_in_range])
	neg = x.shape[0] - pos
	return pos, neg


class TFIDFFeature(Feature):
	def __init__(self, feature_manager, tfidf):
		self.feature_manager = feature_manager
		self.tfidf = tfidf

	def update(self, document):
		self.tfidf.update_tfidf(document)

	def get_train_for_nb(self):
		return [np.histogram(column, 100) for column in self.tfidf.features.x.T]

	def get_score_for_nb(self, values, bins):
		pos = []
		neg = []
		for i in range(1, values.shape[1]):
			if values[0, i] == 0.:
				continue
			index = np.searchsorted(bins[i][0], values[0, i])
			_pos, _neg = labels_for_range(self.tfidf.get_tfidf()[:, i], self.feature_manager.y,
			                              bins[i][1][index], bins[i][1][index + 1] if index < len(bins[i][1]) else None)
			pos.append(_pos)
			neg.append(_neg)
		return pos, neg

	def get_train_for_svm(self):
		return self.tfidf.get_tfidf()

	def get_score_for_svm(self, document):
		return self.tfidf.single_doc_tfidf(document)
