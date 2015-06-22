import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from recommender.features.FeatureManager import FeatureManager
from recommender.model.Ensemble import Ensemble, Mode
from recommender.model.NaiveBayes import NaiveBayes
from recommender.model.SupportVectorMachines import SupportVectorMachines
from recommender.persistence.PickleDS import PickleDS

__author__ = 'Stretchhog'


class Predictor(object):
	def __init__(self, user_id):
		self.user_id = user_id

		self.document_model = ContentPredictor(PickleDS())
		self.user_model = CollaborativePredictor()
		self.rating_model = RatingPredictor()

	def predict(self, user_features, document_features):
		doc_likelihood = self.document_model.score(document_features)
		user_likihood = self.user_model.score(user_features)
		rating = self.rating_model.score(doc_likelihood, user_likihood)
		return rating

	def feedback(self, doc_features, user_features, rating, label):
		self.document_model.update(doc_features, label)
		self.user_model.update(user_features, label)
		self.rating_model.update(rating)


class ContentPredictor(object):
	def __init__(self, ds):
		self.feature_manager = FeatureManager()
		self.ensemble = Ensemble(Mode.GLOBAL_AVG, [NaiveBayes(), SupportVectorMachines()])
		self.document_cache = []
		self.ds = ds
		self.__restore("foo")

	def score(self, document):
		doc, x, y = self.feature_manager.get_document_data(document)
		return self.ensemble.score(doc, x, y)

	def update(self, document, label):
		self.document_cache.append((document, label))
		if len(self.document_cache) >= 2:
			for doc, label in self.document_cache:
				self.__update_knowledge(doc, label)
			self.ensemble.train(self.feature_manager.x, self.feature_manager.y)
			self.__persist()

	def __update_knowledge(self, doc, label):
		self.feature_manager.add_document(doc, label)

	def __persist(self):
		name = "foo"
		data = self.feature_manager.get_for_persistence()
		self.ds.save(name, data)

	def __restore(self, name):
		data = self.ds.load(name)
		if data is not None:
			self.feature_manager.restore(data)


class CollaborativePredictor(object):
	def __init__(self):
		pass

	def score(self, user_features):
		pass

	def update(self, user_features, label):
		pass


class RatingPredictor(object):
	def __init__(self):
		self.x = np.zeros((0, 2))
		self.y = np.zeros((0, 1))
		self.knn = KNeighborsClassifier(n_neighbors=5)
		self.count = 0

	def score(self, doc_likelihood, user_likelihood):
		weights = self.knn.predict_proba(np.array([doc_likelihood, user_likelihood]))
		rating = np.sum(self.knn.classes_ * weights)
		return rating

	def update(self, doc_prob, user_prob, rating):
		array = np.array([doc_prob, user_prob])
		self.x = np.vstack((self.x, array))
		self.y = np.vstack((self.y, rating))
		self.knn.fit(self.x, self.y.ravel())


p = RatingPredictor()

p.update(0.98, 0.92, 5)
p.update(0.9, 0.9, 5)
p.update(0.78, 0.8, 4)
p.update(0.69, 0.75, 4)
p.update(0.5, 0.5, 3)
p.update(0.49, 0.53, 3)
p.update(0.3, 0.3, 2)
p.update(0.26, 0.29, 2)
p.update(0.01, 0.1, 1)
p.update(0.14, 0.09, 1)
p.update(0.10, 0.19, 1)
print('rating:', p.score(0.5, 0.4), 'stars')
