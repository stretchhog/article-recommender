import numpy as np

__author__ = 'Stretchhog'


class Features(object):
	def __init__(self):
		self.x = np.zeros((1, 1))

	def remove_first_row(self):
		self.x = np.delete(self.x, 0, 0)

	def add_row(self, row):
		self.x = np.hstack((self.x, row))

	def add_column(self, column):
		self.x = np.vstack((self.x, column))

	def number_of_features(self):
		return self.x.shape[1]

	def number_of_documents(self):
		return self.x.shape[0]
