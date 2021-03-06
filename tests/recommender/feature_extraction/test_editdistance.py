import unittest

from recommender.utils import editdistance


class TestLevenshtein(unittest.TestCase):
	def test_empty_string(self):
		distance = editdistance.levenshtein("", "")
		self.assertEqual(distance, 0)

	def test_first_empty_string(self):
		distance = editdistance.levenshtein("", "string")
		self.assertEqual(distance, 6)

	def test_second_empty_string(self):
		distance = editdistance.levenshtein("string", "")
		self.assertEqual(distance, 6)

	def test_same_string(self):
		distance = editdistance.levenshtein("string", "string")
		self.assertEqual(distance, 0)

	def test_one_insertion(self):
		distance = editdistance.levenshtein("string", "stringi")
		self.assertEqual(distance, 1)

	def test_one_deletion(self):
		distance = editdistance.levenshtein("strng", "string")
		self.assertEqual(distance, 1)

	def test_one_substitution(self):
		distance = editdistance.levenshtein("string", "streng")
		self.assertEqual(distance, 2)

	def test_full(self):
		distance = editdistance.levenshtein("execution", "intention")
		self.assertEqual(distance, 8)


if __name__ == '__main__':
	unittest.main()
