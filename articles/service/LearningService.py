import uuid
from beaker.cache import CacheManager, Cache
from beaker.util import parse_cache_config_options
from pandas import DataFrame
from recommender.Predictors import Predictor

__author__ = 'Stretchhog'


class LearningService(object):
	def __init__(self):
		self.predictors = LoadingCache('predictors')
		self.user_cache = LoadingCache('users')
		self.content_cache = LoadingCache('documents')
		self.ratings = DataFrame()
		self.prediction_cache = LoadingCache('prediction')

	def predict(self, user_id, doc_id):
		key = self.__composite_key(user_id, doc_id)
		if key in self.prediction_cache:
			return self.prediction_cache.get(key).rating

		predictor = self.__get_predictor(user_id)
		content_features = self.__get_content_features(doc_id)
		collaborative_features = self.__get_collaborative_features()
		rating = predictor.predict(collaborative_features, content_features)
		self.prediction_cache.put(key, Prediction(rating))
		return rating

	@staticmethod
	def __composite_key(user_id, doc_id):
		return str(user_id) + '-' + str(doc_id)

	def save_reading_features(self, user_id, doc_id, reading_features):
		self.prediction_cache.get(self.__composite_key(doc_id, user_id)).reading_features = reading_features

	def __get_predictor(self, user_id):
		if user_id in self.predictors:
			predictor = self.predictors[user_id]
		else:
			predictor = Predictor(user_id)
			self.predictors[user_id] = predictor
		return predictor

	def feedback(self, user_id, doc_id, predicted_rating, rating):
		label = (rating == predicted_ratNoneing)
		self.predictors.get(user_id).feedback(self.content_cache.get(doc_id), self.user_cache.get(user_id), rating, label)
		self.ratings.put(user_id, [user_id, doc_id, rating, self.prediction_cache.get(self.__composite_key(doc_id, user_id)).reading_features])

	def __get_collaborative_features(self):
		return self.ratings

	def __get_content_features(self, doc_id):
		if doc_id in self.content_cache:
			return self.content_cache.get(doc_id)
		else:
			return None


class Prediction(object):
	def __init__(self, rating):
		self.rating = rating
		self.reading_features = None


class LoadingCache(Cache):
	def __init__(self, namespace, **nsargs):
		super().__init__(namespace, **nsargs)
		cache_opts = {
			'cache.type': 'file',
			'cache.data_dir': 'cache/data',
			'cache.lock_dir': 'cache/lock',
			'cache.short_term.type': 'ext:memcached',
			'cache.short_term.url': '127.0.0.1.11211',
			'cache.short_term.expire': '3600',
			'cache.long_term.type': 'file',
			'cache.long_term.expire': '86400'
		}
		cache_manager = CacheManager(**parse_cache_config_options(cache_opts))
		self.cache = cache_manager.get_cache(namespace, type='dbm')
