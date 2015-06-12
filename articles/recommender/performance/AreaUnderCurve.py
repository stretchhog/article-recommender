__author__ = 'Stretchhog'

# for predictive model
class AreaUnderCurve(object):
	def __init__(self):
		self.BINS = 100
		self.RESOLUTION = 1.0 / self.BINS
		self.MIN_AUC = 0.5

		self.pos = [0] * self.BINS
		self.neg = [0] * self.BINS

	def update(self, score, label):
		if not (0. <= score & score <= 1.):
			raise AttributeError("The score should be always in [0, 1] interval")
		idx = min(int(score / self.RESOLUTION), self.BINS - 1)
		if label:
			self.pos[idx] += 1
		else:
			self.neg[idx] += 1

	def get_auc(self):
		cumPos = self.pos[0]
		cumNeg = self.neg[0]
		sum = cumNeg * cumPos
		for i in range(len(self.pos)):
			prevCumNeg = cumNeg
			cumPos += self.pos[i]
			cumNeg += self.neg[i]
			sum += self.pos[i] * (cumNeg + prevCumNeg)

		if cumPos == 0 | cumNeg == 0:
			return self.MIN_AUC

		auc = self.MIN_AUC * sum / (cumPos * cumNeg)
		return 1 - auc if auc < self.MIN_AUC else auc
