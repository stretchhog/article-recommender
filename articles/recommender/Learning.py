import uuid
from recommender.persistence.PickleDS import PickleDS
from recommender.Predictors import DocumentPredictor, UserPredictor, RatingPredictor

__author__ = 'Stretchhog'


class Learning(object):
	def __init__(self):
		self.prediction_cache = {}
		self.document_model = DocumentPredictor(PickleDS())
		self.user_model = UserPredictor()
		self.rating_model = RatingPredictor()

	def predict(self, user_id, doc_id, user_features, doc_features):
		doc_likelihood = self.document_model.score(doc_id, doc_features)
		user_likihood = self.user_model.score(user_id, user_features)
		rating = self.rating_model.score(doc_likelihood, user_likihood)
		prediction_id = uuid.uuid4
		self.prediction_cache[prediction_id] = Prediction(rating, doc_id, doc_likelihood, doc_features, user_id,
		                                                  user_likihood, user_features)
		return prediction_id, rating

	def feedback(self, prediction_id, rating):
		if prediction_id in self.prediction_cache:
			prediction = self.prediction_cache[prediction_id]
			label = True if rating == prediction.rating else label = False
			self.document_model.train(prediction.doc_id, prediction.doc_features, label)
			self.user_model.train(prediction.user_id, prediction.user_features, label)
			self.rating_model.train(rating)


class Prediction(object):
	def __init__(self, rating, doc_id, doc_likelihood, doc_features, user_id, user_likelihood, user_features):
		self.rating = rating
		self.doc_id = doc_id
		self.doc_likelihood = doc_likelihood
		self.doc_features = doc_features
		self.user_id = user_id
		self.user_likelihood = user_likelihood
		self.user_features = user_features
