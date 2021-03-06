import itertools

import nltk
from nltk.corpus import stopwords


class Tokenization(object):
	stop = stopwords.words('english')

	def __init__(self):
		self.minimum_word_length = 3

	def tokenize(self, document):
		sentences = self.__split_sentences(document)
		sentences = self.__extract_tokens(sentences)
		sentences = self.__pos_tagging(sentences)
		sentences = self.__remove_stopwords(sentences)
		sentences = self.__remove_short_words(sentences)
		flat = self.__flatten_sentences(sentences)
		tokens = self.__remove_pos_tags(flat)
		return tokens

	def __split_sentences(self, document):
		return nltk.sent_tokenize(document)

	def __extract_tokens(self, sentences):
		return [nltk.word_tokenize(sentence) for sentence in sentences]

	def __pos_tagging(self, sentences):
		return [nltk.pos_tag(sentence) for sentence in sentences]

	def __remove_stopwords(self, sentences):
		return [[token for token in sentence if token[0] not in Tokenization.stop] for sentence in sentences]

	def __remove_short_words(self, sentences):
		return [[token for token in sentence if len(token[0]) >= self.minimum_word_length] for sentence in sentences]

	def __flatten_sentences(self, sentences):
		return list(itertools.chain.from_iterable(sentences))

	def __remove_pos_tags(self, tokens):
		return [token[0] for token in tokens]
