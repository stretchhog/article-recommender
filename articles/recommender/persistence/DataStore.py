import abc

__author__ = 'Stretchhog'


class DataStore(object):
	@abc.abstractmethod
	def save(self, name, data):
		return

	@abc.abstractmethod
	def load(self, name):
		return
