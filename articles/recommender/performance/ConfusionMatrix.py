import numpy as np

__author__ = 'Stretchhog'

# for mutlt-star ratings model

class ConfusionMatrix(object):
	def __init__(self):
		self.matrix = np.zeros((5, 5))

	def update(self, actual, predicted):
		self.matrix[actual][predicted] += 1
