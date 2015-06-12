import numpy as np


class TFIDF(object):
	def __init__(self, feature_manager):
		self.feature_manager = feature_manager
		self.vocabulary = {}
		self.word_index = 0

	def update_tfidf(self, tokens):
		self.__add_to_vocabulary(tokens)
		tf_array = self.__create_document_vector(tokens)
		self.feature_manager.add_document(tf_array, self.__get_column_difference_matrix(tf_array))

	def get_tfidf(self):
		features = self.feature_manager.features.x
		tf = features / features.sum(axis=0)
		idf = self.__calculate_idf(features)
		return np.multiply(tf, idf)

	def __calculate_idf(self, features):
		number_of_documents, _ = self.feature_manager.feature_dimensions()
		idf = np.log10(number_of_documents / (features != 0).sum(axis=0))
		return idf

	def single_doc_tfidf(self, tokens):
		tf = self.__create_document_vector(tokens)
		idf = self.__calculate_idf(self.feature_manager.features.x)
		return np.multiply(tf, idf)

	def __get_column_difference_matrix(self, tf_array):
		number_of_documents, number_of_features = self.feature_manager.feature_dimensions()
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
