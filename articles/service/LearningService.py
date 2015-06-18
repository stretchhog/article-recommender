import uuid
from beaker.cache import CacheManager, Cache
from beaker.util import parse_cache_config_options
from recommender.Predictors import Predictor

__author__ = 'Stretchhog'


class LearningService(object):
	def __init__(self):
		self.prediction_cache = LoadingCache('prediction')
		self.document_cache = LoadingCache('documents')
		self.user_cache = LoadingCache('users')
		self.predictors = LoadingCache('predictors')

	def predict(self, user_id, doc_id, reading_features):
		if self.predictors.has_key(user_id):
			predictor = Predictor(user_id)
			self.predictors[user_id] = predictor
		else:
			predictor = self.predictors[user_id]

		rating = predictor.predict(reading_features)
		prediction_id = uuid.uuid4
		self.prediction_cache.put(prediction_id, Prediction(rating, doc_id, user_id))
		return prediction_id, rating

	def feedback(self, prediction_id, rating):
		if self.prediction_cache.has_key(prediction_id):
			prediction = self.prediction_cache.get(prediction_id)
			label = (rating == prediction.rating)
			self.predictors.get(prediction.user_id).feedback(
				self.document_cache.get(prediction.doc_id),
				self.user_cache.get(prediction.user_id),
				rating,
				label
			)
			self.prediction_cache.remove_value(prediction_id)


class Prediction(object):
	def __init__(self, rating, doc_id, user_id):
		self.rating = rating
		self.doc_id = doc_id
		self.user_id = user_id


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
