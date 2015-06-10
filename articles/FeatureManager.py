import numpy as np
from Features import Features

__author__ = 'tvancann'


class FeatureManager(object):
	def __init__(self):
		self.features = Features()
		self.number_of_features = 0
		self.y = []

	def add_document(self, row, column=None):
		"""

		:type column: numpy.ndarray
		"""
		if column is not None:
			self.features.add_column(column)

		self.features.add_row(row)
		self.number_of_features = self.features.x.shape[1]

	def feature_dimensions(self):
		return self.features.number_of_documents(), self.number_of_features

	def get_x(self):
		return self.features.x

	def get_y(self):
		return self.y

	def add_label(self, label):
		self.y = np.hstack((self.y, label))

