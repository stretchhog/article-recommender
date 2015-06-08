from Features import Features

__author__ = 'tvancann'


class FeatureManager:
	def __init__(self):
		self.features = Features()

	def add_document(self, row, column):
		self.features.add_column(column)
		self.features.add_row(row)

	def feature_dimensions(self):
		return self.features.number_of_documents(), self.features.number_of_features()

	def clean_features(self):
		if sum(self.features.get()[0, :]) == 0:
			self.features.remove_first_row()

	def get_features(self):
		return self.features
