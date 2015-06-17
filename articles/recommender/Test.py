from recommender.Recommender import Pipeline
from recommender.persistence.PickleDS import PickleDS

__author__ = 'Stretchhog'


def read_train_data():
	pass


def read_score_data():
	pass


pipeline = Pipeline(PickleDS())

train_data = read_train_data()
for d, label in train_data:
	pipeline.train(d, label)

score_data = read_score_data()
for d, label in score_data:
	print(pipeline.score(d))
