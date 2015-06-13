import numpy as np
from recommender.feature_extraction.Tokenization import Tokenization


class TFIDF(object):
	def __init__(self):
		self.features = TFIDFFeatures()
		self.tokenizer = Tokenization()
		self.vocabulary = {}
		self.word_index = 0

	def update_tfidf(self, document):
		tokens = self.tokenizer.tokenize(document)
		self.__add_to_vocabulary(tokens)
		tf_array = self.__create_document_vector(tokens)
		self.features.add_data(tf_array, self.__get_column_difference_matrix(tf_array))

	def get_tfidf(self):
		features = self.features.x
		tf = features / features.sum(axis=0)
		idf = self.__calculate_idf(features)
		return np.multiply(tf, idf)

	def single_doc_tfidf(self, document):
		tokens = self.tokenizer.tokenize(document)
		tf = self.__create_document_vector(tokens)
		idf = self.__calculate_idf(self.features.x)
		return np.multiply(tf, idf)

	def __calculate_idf(self, features):
		number_of_documents, _ = self.features.x.shape
		idf = np.log10(number_of_documents / (features != 0).sum(axis=0))
		return idf

	def __get_column_difference_matrix(self, tf_array):
		number_of_documents, number_of_features = self.features.x.shape
		return np.zeros((number_of_documents, tf_array.shape[1] - number_of_features))

	def __create_document_vector(self, words):
		local_dictionary = {}
		tf_array = np.zeros((1, len(self.vocabulary)))

		for word in words:
			if word in local_dictionary:
				local_dictionary[word] += 1
			else:
				local_dictionary[word] = 1
			if word in self.vocabulary:
				tf_array[0, self.vocabulary[word]] = local_dictionary[word]

		return tf_array

	def __add_to_vocabulary(self, words):
		new_words = filter(lambda word: word not in self.vocabulary, words)
		uniques = set(new_words)
		for new_word in uniques:
			self.vocabulary[new_word] = self.word_index
			self.word_index += 1


class TFIDFFeatures(object):
	def __init__(self):
		self.x = np.zeros((1, 1))
		self.first_row = True

	def remove_first_row(self):
		self.x = np.delete(self.x, 0, 0)

	def add_data(self, row, column):
		if column is not None:
			self.add_column(column)
		self.add_row(row)

	def add_row(self, row):
		if self.x.shape == (1, 1):
			self.x = row
			self.first_row = False
		else:
			self.first_row = False
			self.x = np.vstack((self.x, row))

	def add_column(self, column):
		if not self.first_row:
			self.x = np.hstack((self.x, column))

	def number_of_features(self):
		return self.x.shape[1]

	def number_of_documents(self):
		return self.x.shape[0]
