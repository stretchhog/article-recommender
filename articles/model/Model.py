__author__ = 'Stretchhog'
import abc


class Model(object):
	@abc.abstractmethod
	def train(self, document):
		return

	@abc.abstractmethod
	def score(self, document):
		return
