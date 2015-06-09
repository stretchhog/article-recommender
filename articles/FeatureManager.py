import numpy as np
from Features import Features

__author__ = 'tvancann'


class FeatureManager(object):
	def __init__(self):
		self.features = Features()
		self.number_of_features = 0
		self.y = np.zeros((1, 1))

	def add_document(self, row, label, column=None):
		"""

		:type column: numpy.ndarray
		"""
		if column is not None:
			self.number_of_features = len(column)
			self.features.add_column(column)

		self.features.add_row(row)
		self.y = np.hstack((self.y, label))

	def feature_dimensions(self):
		"""
		Returns the dimensions of the data

		:return: number of documents, number of features
		"""
		return self.features.number_of_documents(), self.features.number_of_features()

	def clean_features(self):
		if sum(self.features.x[0, :]) == 0:
			self.features.remove_first_row()

	def get_features(self):
		return self.features

	def get_x(self):
		return self.features.x

	def get_y(self):
		return self.y
