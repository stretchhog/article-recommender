import numpy as np

from recommender.Features import Features

__author__ = 'tvancann'


class FeatureManager(object):
	def __init__(self):
		self.features = Features()
		self.y = []

	def add_document(self, row, column=None):
		if column is not None:
			self.features.add_column(column)

		self.features.add_row(row)

	def feature_dimensions(self):
		return self.features.number_of_documents(), self.features.number_of_features()

	def add_label(self, label):
		self.y = np.hstack((self.y, label))
