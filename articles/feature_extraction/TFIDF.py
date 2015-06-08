import numpy as np


class TFIDF:
	def __init__(self, feature_manager):
		"""

		:type feature_manager: FeatureManager.FeatureManager
		"""
		self.features_manager = feature_manager
		self.vocabulary = {}
		self.word_index = 0

	def update_tfidf(self, tokens):
		self.__add_to_vocabulary(tokens)
		tf_array = self.__create_document_vector(tokens)
		self.features_manager.add_document(tf_array, self.__get_column_difference_matrix(tf_array))

	def get_tfidf(self):
		self.features_manager.clean_features()
		features = self.features_manager.get_features()
		tf = features / features.sum(axis=1)
		number_of_documents, _ = self.features_manager.feature_dimensions()
		idf = number_of_documents / (features != 0).sum(axis=0)
		return np.multiply(tf, idf)

	def __get_column_difference_matrix(self, tf_array):
		number_of_documents, number_of_features = self.features_manager.feature_dimensions()
		return np.zeros((number_of_documents, len(tf_array) - number_of_features))

	def __create_document_vector(self, words):
		local_dictionary = {}
		tf_array = np.zeros(len(self.vocabulary))

		for word in words:
			if word in local_dictionary:
				local_dictionary[word] += 1
			else:
				local_dictionary[word] = 1
			tf_array[self.vocabulary[word]] = local_dictionary[word]
		return tf_array

	def __add_to_vocabulary(self, words):
		new_words = filter(lambda word: word not in self.vocabulary, words)
		for new_word in new_words:
			self.vocabulary[new_word] = self.word_index
			self.word_index += 1
