import numpy as np

__author__ = 'Stretchhog'


class Features:
	def __init__(self):
		self.matrix = np.zeros((1, 1))

	def remove_first_row(self):
		self.matrix = np.delete(self.matrix, 0, 0)

	def add_row(self, row):
		self.matrix = np.hstack((self.matrix, row))

	def add_column(self, column):
		self.matrix = np.vstack((self.matrix, column))

	def number_of_features(self):
		return self.matrix.shape[1]

	def number_of_documents(self):
		return self.matrix.shape[0]

	def get(self):
		return self.matrix
