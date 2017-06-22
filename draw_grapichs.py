import numpy as np
import argparse
from face_feature_extractor import FaceFeatureExtractor
from scores_extractor import ScoresExtractor
from models_trainer import ModelsTrainer
import json
from collections import OrderedDict
import glob
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

scores_extractor = ScoresExtractor( glob.glob(os.path.realpath('./scores/*.txt')))
scores = scores_extractor.extract_scores_for_histogram()
# scores = [round(val) for val in scores]
scores = np.array(sorted(scores))

print("Scores Mean: ", scores.mean())
print("Standard deviation: ", scores.std())

fit = stats.norm.pdf(scores, scores.mean(), scores.std())



plt.ylabel('Number of rates');

plt.plot(scores, fit, '-o')
plt.hist(scores,normed=True, bins=10)
plt.savefig('score_reports/graphics/scores_histogram.png')
# plt.show()
plt.clf()

plt.boxplot(scores)
plt.savefig('score_reports/graphics/scores_boxplot.png')

