__author__ = 'Stretchhog'
import abc


class Model(object):
	@abc.abstractmethod
	def train(self, x, y):
		return

	@abc.abstractmethod
	def score(self, x):
		return

	@abc.abstractmethod
	def get_type(self):
		return

