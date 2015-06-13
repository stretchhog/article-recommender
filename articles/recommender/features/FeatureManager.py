from enum import Enum
import numpy as np
from recommender.feature_extraction.Sentiment import Sentiment
from recommender.feature_extraction.TFIDF import TFIDF

__author__ = 'tvancann'


class FeatureManager(object):
	def __init__(self):
		self.x = [[], [], [], [], []]
		self.tfidf = TFIDF()
		self.sentiment = Sentiment()
		self.y = []

	def add_document(self, data, label):
		self.x[ArticleFeature.TOPIC[0]].append(data[ArticleFeature.TOPIC[1]])
		self.x[ArticleFeature.ORIGIN[0]].append(data[ArticleFeature.ORIGIN[1]])
		self.x[ArticleFeature.WRITER[0]].append(data[ArticleFeature.WRITER[1]])
		document = data['document']
		self.x[ArticleFeature.SENTIMENT[0]].append(self.sentiment.get_sentiment(document))
		self.tfidf.update_tfidf(document)
		self.y = np.hstack((self.y, label))

	def get_data(self):
		self.gather_features()
		return self.x, self.y

	def gather_features(self):
		self.x[ArticleFeature.TFIDF[0]] = self.tfidf.get_tfidf()

	def restore(self, data):
		self.x[ArticleFeature.TOPIC[0]] = data[ArticleFeature.TOPIC[1]]
		self.x[ArticleFeature.ORIGIN[0]] = data[ArticleFeature.ORIGIN[1]]
		self.x[ArticleFeature.WRITER[0]] = data[ArticleFeature.WRITER[1]]
		self.x[ArticleFeature.SENTIMENT[0]] = data[ArticleFeature.SENTIMENT[1]]
		self.y = data['y']
		self.tfidf.vocabulary = data['tfidf']['vocabulary']
		self.tfidf.word_index = data['tfidf']['word_index']

	def get_for_persistence(self):
		data = {
			"tfidf": {
				"vocabulary": self.tfidf.vocabulary,
				"word_index": self.tfidf.word_index
			},
			"feature_manager": {
				ArticleFeature.TOPIC(1): self.x[ArticleFeature.TOPIC[0]],
				ArticleFeature.ORIGIN(1): self.x[ArticleFeature.ORIGIN[0]],
				ArticleFeature.WRITER(1): self.x[ArticleFeature.WRITER[0]],
				ArticleFeature.SENTIMENT(1): self.x[ArticleFeature.SENTIMENT[0]],
				"y": self.y
			}
		}
		return data


class ArticleFeature(Enum):
	TOPIC = (0, "topic")
	ORIGIN = (1, "origin")
	WRITER = (2, "author")
	SENTIMENT = (3, "sentiment")
	TFIDF = (4, "tfidf")


class UserFeatureName(Enum):
	TIME = 1
	WEEKDAY = 2
	LOCATION = 3
