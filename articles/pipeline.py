from articles.FeatureManager import FeatureManager
from articles.feature_extraction.TFIDF import TFIDF
from articles.feature_extraction.Tokenization import Tokenization
from articles.model.Ensemble import Ensemble
from articles.model.NaiveBayes import NaiveBayes

__author__ = 'Stretchhog'


class Pipeline(object):
	def __init__(self):
		self.feature_manager = FeatureManager()
		self.tfidf = TFIDF(self.feature_manager)
		self.tokenization = Tokenization()
		self.ensemble = Ensemble()
		self.ensemble.register(NaiveBayes(self.feature_manager))
		self.cache = []

	def score(self, document):
		features = self.tokenization.tokenize(document)
		return self.ensemble.score(x, new)

	def train(self, document):
		self.cache.append(document)
		if len(self.cache) >= 5:
			for doc in self.cache:
				tokens = self.tokenization.tokenize(doc)
				self.tfidf.update_tfidf(tokens)
		self.ensemble.train(self.feature_manager.get_x(), self.feature_manager.get_y())
