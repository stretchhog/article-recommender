import unittest
from articles.feature_extraction import Tokenization


class TestTokenization(unittest.TestCase):
	def test_empty_string(self):
		tokenization = Tokenization.Tokenization("")
		tokens = tokenization.tokenize()
		self.assertEqual(tokens, [''])

	def test_remove_punctuation(self):
		tokenization = Tokenization.Tokenization("this. is*^/ a ,test\';")
		tokens = tokenization.tokenize()
		self.assertEqual(tokens, ['this', 'is', 'a', 'test'])

	def test_remove_stopwords(self):
		tokenization = Tokenization.Tokenization("this is the test")
		tokens = tokenization.tokenize()
		self.assertEqual(tokens, ['this', 'test'])


if __name__ == '__main__':
	unittest.main()
