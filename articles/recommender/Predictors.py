from recommender.features.FeatureManager import FeatureManager
from recommender.model.Ensemble import Ensemble, Mode
from recommender.model.NaiveBayes import NaiveBayes
from recommender.model.SupportVectorMachines import SupportVectorMachines
from recommender.persistence.PickleDS import PickleDS

__author__ = 'Stretchhog'


class DocumentPredictor(object):
	def __init__(self, ds):
		self.feature_manager = FeatureManager()
		self.ensemble = Ensemble(Mode.GLOBAL_AVG, [NaiveBayes(), SupportVectorMachines()])
		self.document_cache = []
		self.ds = ds
		self.restore("foo")

	def score(self, document):
		doc, x, y = self.feature_manager.get_document_data(document)
		return self.ensemble.score(doc, x, y)

	def train(self, document, label):
		self.document_cache.append((document, label))
		if len(self.document_cache) >= 2:
			for doc, label in self.document_cache:
				self.update_knowledge(doc, label)
			self.ensemble.train(self.feature_manager.x, self.feature_manager.y)
			self.persist()

	def update_knowledge(self, doc, label):
		self.feature_manager.add_document(doc, label)

	def persist(self):
		name = "foo"
		data = self.feature_manager.get_for_persistence()
		self.ds.save(name, data)

	def restore(self, name):
		data = self.ds.load(name)
		if data is not None:
			self.feature_manager.restore(data)


class UserPredictor(object):
	def __init__(self):
		pass


class RatingPredictor(object):
	def __init__(self):
		pass
