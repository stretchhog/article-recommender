import numpy as np

__author__ = 'Stretchhog'


class Features(object):
	def __init__(self):
		self.x = np.zeros((1, 1))
		self.first_row = True

	def remove_first_row(self):
		self.x = np.delete(self.x, 0, 0)

	def add_row(self, row):
		if self.first_row:
			self.x = row
			self.first_row = False
		else:
			self.x = np.vstack((self.x, row))

	def add_column(self, column):
		if not self.first_row:
			self.x = np.hstack((self.x, column))

	def number_of_features(self):
		return self.x.shape[1]

	def number_of_documents(self):
		return self.x.shape[0]
