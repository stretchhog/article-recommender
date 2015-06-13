from recommender.features import FeatureManager
from recommender.feature_extraction import TFIDF

__author__ = 'tvancann'

import unittest


class TestTFIDF(unittest.TestCase):
	def test_add_to_voc(self):
		tfidf = TFIDF(FeatureManager())
		res = tfidf.single_doc_tfidf([])
		self.assertEqual(res.shape, (1, 0))

	def test_update_tfidf(self):
		tfidf = TFIDF(FeatureManager())
		tfidf.update_tfidf(["hello", "my", "name", "is", "Tim"])
		tfidf.update_tfidf(["hello", "my", "cat", "is", "called", "Isis"])
		self.assertEqual(tfidf.feature_manager.feature_dimensions()[1], 8)
		self.assertEqual(tfidf.feature_manager.feature_dimensions()[0], 2)

		res = tfidf.get_tfidf()


if __name__ == '__main__':
	unittest.main()
