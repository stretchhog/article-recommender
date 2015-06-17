from enum import Enum
import numpy as np
from recommender.feature_extraction.Sentiment import Sentiment
from recommender.feature_extraction.TFIDF import TFIDF
from recommender.features.Feature import CategoricFeature, NumericFeature, TFIDFFeature

__author__ = 'tvancann'


class FeatureManager(object):
	def __init__(self):
		self.tfidf = TFIDF()
		self.x = [CategoricFeature(),
		          CategoricFeature(),
		          CategoricFeature(),
		          NumericFeature(),
		          TFIDFFeature(self.tfidf)]
		self.sentiment = Sentiment()
		self.y = np.zeros((0, 1))

	def add_document(self, data, label):
		self.x[ArticleFeature.TOPIC.value[0]].update(data[ArticleFeature.TOPIC.value[1]])
		self.x[ArticleFeature.ORIGIN.value[0]].update(data[ArticleFeature.ORIGIN.value[1]])
		self.x[ArticleFeature.AUTHOR.value[0]].update(data[ArticleFeature.AUTHOR.value[1]])
		document = data['article']
		self.x[ArticleFeature.SENTIMENT.value[0]].update(self.sentiment.get_sentiment(document))
		self.x[ArticleFeature.TFIDF.value[0]].update(document)

		if self.y.shape is (0, 0):
			self.y = np.array(label)
		else:
			self.y = np.vstack((self.y, label))

	def get_document_data(self, data):
		features = [[], [], [], [], []]
		features[ArticleFeature.TOPIC.value[0]] = data[ArticleFeature.TOPIC.value[1]]
		features[ArticleFeature.ORIGIN.value[0]] = data[ArticleFeature.ORIGIN.value[1]]
		features[ArticleFeature.AUTHOR.value[0]] = data[ArticleFeature.AUTHOR.value[1]]
		document = data['article']
		features[ArticleFeature.SENTIMENT.value[0]] = self.sentiment.get_sentiment(document)
		features[ArticleFeature.TFIDF.value[0]] = self.tfidf.single_doc_tfidf(document)
		return features, self.x, self.y

	def restore(self, data):
		self.x[ArticleFeature.TOPIC.value[0]] = data['feature_manager'][ArticleFeature.TOPIC.value[1]]
		self.x[ArticleFeature.ORIGIN.value[0]] = data['feature_manager'][ArticleFeature.ORIGIN.value[1]]
		self.x[ArticleFeature.AUTHOR.value[0]] = data['feature_manager'][ArticleFeature.AUTHOR.value[1]]
		self.x[ArticleFeature.SENTIMENT.value[0]] = data['feature_manager'][ArticleFeature.SENTIMENT.value[1]]
		self.y = data['y']
		self.tfidf.vocabulary = data['tfidf']['vocabulary']
		self.tfidf.word_index = data['tfidf']['word_index']
		self.tfidf.features.x = data['tfidf']['x']
		self.tfidf.features.first_row = False

	def get_for_persistence(self):
		data = {
			'tfidf': {
				'vocabulary': self.tfidf.vocabulary,
				'word_index': self.tfidf.word_index,
				'x': self.tfidf.features.x
			},
			'feature_manager': {
				ArticleFeature.TOPIC.value[1]: self.x[ArticleFeature.TOPIC.value[0]],
				ArticleFeature.ORIGIN.value[1]: self.x[ArticleFeature.ORIGIN.value[0]],
				ArticleFeature.AUTHOR.value[1]: self.x[ArticleFeature.AUTHOR.value[0]],
				ArticleFeature.SENTIMENT.value[1]: self.x[ArticleFeature.SENTIMENT.value[0]],
			},
			'y': self.y
		}
		return data


class ArticleFeature(Enum):
	TOPIC = (0, 'topic')
	ORIGIN = (1, 'origin')
	AUTHOR = (2, 'author')
	SENTIMENT = (3, 'sentiment')
	TFIDF = (4, 'tfidf')


class UserFeatureName(Enum):
	TIME = 1
	WEEKDAY = 2
	LOCATION = 3
