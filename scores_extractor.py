import json
from collections import defaultdict
import numpy as np
from sklearn import preprocessing

class ScoresExtractor:
    def __init__(self, file_names):
        self.filenames = file_names
        self.number_of_scores = len(file_names)

    def extract_scores_for_histogram(self):
        scores = []
        for filename in self.filenames:
            with open(filename) as json_file:
                next_scores = json.load(json_file)
                for key, value in next_scores.items():
                    scores.append(value)

        return scores
    def extract_average_scores(self):
        sum_of_scores = defaultdict(lambda: 0, {})
        average_scores = {}
        for filename in self.filenames:
            with open(filename) as json_file:
                next_scores = json.load(json_file)
                for key, value in next_scores.items():
                    sum_of_scores[key] += value

        for key, value in sum_of_scores.items():
            average_scores[key] = value / self.number_of_scores
        return average_scores

    def get_z_scaled_sum_of_scores(self):
        sum_of_scores = defaultdict(lambda: 0, {})


        for filename in self.filenames:
            with open(filename) as json_file:
                next_scores = json.load(json_file)
                next_scores_row_labels = np.array([item[0] for item in next_scores.items()])
                next_scores_row = np.array([item[1] for item in next_scores.items()])
                avr = next_scores_row.mean()
                std = next_scores_row.std()

                new_scores = [((score - avr) / std) for score in next_scores_row]

                for index, key in enumerate(next_scores_row_labels):
                    sum_of_scores[key] += new_scores[index]

        return sum_of_scores

    def extract_z_scaled(self):
        sum_of_scores = self.get_z_scaled_sum_of_scores()

        sum_of_scores_row_labels = np.array([item[0] for item in sum_of_scores.items()])
        sum_of_scores_row = np.array([item[1] for item in sum_of_scores.items()])

        min_max_scaler = preprocessing.MinMaxScaler(feature_range = (1, 10))
        sum_of_scores_row_scaled = min_max_scaler.fit_transform(sum_of_scores_row.transpose())

        for index, key in enumerate(sum_of_scores_row_labels):
            sum_of_scores[key] = sum_of_scores_row_scaled[index]

        return sum_of_scores

    def get_z_scaled_average(self):
        sum_of_scores = self.get_z_scaled_sum_of_scores()
        average_scores = {}
        for key, value in sum_of_scores.items():
            average_scores[key] = value / self.number_of_scores

        return average_scores