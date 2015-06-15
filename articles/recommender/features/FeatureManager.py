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
		self.x[ArticleFeature.TOPIC.value[0]].append(data[ArticleFeature.TOPIC.value[1]])
		self.x[ArticleFeature.ORIGIN.value[0]].append(data[ArticleFeature.ORIGIN.value[1]])
		self.x[ArticleFeature.AUTHOR.value[0]].append(data[ArticleFeature.AUTHOR.value[1]])
		document = data['article']
		self.x[ArticleFeature.SENTIMENT.value[0]].append(self.sentiment.get_sentiment(document))
		self.tfidf.update_tfidf(document)
		self.y = np.hstack((self.y, label))

	def get_training_data(self):
		self.gather_features()
		return self.x, self.y

	def get_document_data(self, data):
		features = [[], [], [], [], []]
		features[ArticleFeature.TOPIC.value[0]] = data[ArticleFeature.TOPIC.value[1]]
		features[ArticleFeature.ORIGIN.value[0]] = data[ArticleFeature.ORIGIN.value[1]]
		features[ArticleFeature.AUTHOR.value[0]] = data[ArticleFeature.AUTHOR.value[1]]
		document = data['article']
		features[ArticleFeature.SENTIMENT.value[0]] = self.sentiment.get_sentiment(document)
		features[ArticleFeature.TFIDF.value[0]] = self.tfidf.single_doc_tfidf(document)
		return features

	def gather_features(self):
		self.x[ArticleFeature.TFIDF.value[0]] = self.tfidf.get_tfidf()

	def restore(self, data):
		self.x[ArticleFeature.TOPIC.value[0]] = data['feature_manager'][ArticleFeature.TOPIC.value[1]]
		self.x[ArticleFeature.ORIGIN.value[0]] = data['feature_manager'][ArticleFeature.ORIGIN.value[1]]
		self.x[ArticleFeature.AUTHOR.value[0]] = data['feature_manager'][ArticleFeature.AUTHOR.value[1]]
		self.x[ArticleFeature.SENTIMENT.value[0]] = data['feature_manager'][ArticleFeature.SENTIMENT.value[1]]
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
				ArticleFeature.TOPIC.value[1]: self.x[ArticleFeature.TOPIC.value[0]],
				ArticleFeature.ORIGIN.value[1]: self.x[ArticleFeature.ORIGIN.value[0]],
				ArticleFeature.AUTHOR.value[1]: self.x[ArticleFeature.AUTHOR.value[0]],
				ArticleFeature.SENTIMENT.value[1]: self.x[ArticleFeature.SENTIMENT.value[0]],
			},
			"y": self.y
		}
		return data


class ArticleFeature(Enum):
	TOPIC = (0, "topic")
	ORIGIN = (1, "origin")
	AUTHOR = (2, "author")
	SENTIMENT = (3, "sentiment")
	TFIDF = (4, "tfidf")


class UserFeatureName(Enum):
	TIME = 1
	WEEKDAY = 2
	LOCATION = 3
