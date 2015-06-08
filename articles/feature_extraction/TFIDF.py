import numpy as np


class TFIDF:
	def __init__(self, features):
		self.features = features
		self.vocabulary = {}
		self.word_index = 0
		self.feature_start = self.features.number_of_features()

	def update_tfidf(self, tokens):
		self.__add_to_vocabulary(tokens)
		tf_array = self.__create_document_vector(tokens)
		self.features.add_column(self.__get_column_difference_matrix(tf_array))
		self.features.add_row(tf_array)

	def get_tfidf(self):
		if sum(self.features.get()[0, :]) == 0:
			self.features.remove_first_row()
		tf = self.features.get() / self.features.get().sum(axis=1)
		non_zeros = self.features.get() != 0
		idf = self.features.number_of_documents() / non_zeros.sum(axis=0)
		return np.multiply(tf, idf)

	def __get_column_difference_matrix(self, tf_array):
		return np.zeros((self.features.number_of_documents(), len(tf_array) - self.features.number_of_features()))

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
