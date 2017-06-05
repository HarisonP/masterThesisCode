import json
from collections import defaultdict

class ScoresExtractor:
    def __init__(self, file_names):
        self.filenames = file_names
        self.number_of_scores = len(file_names)

    def extract_average_scores(self):
        sum_of_scores = defaultdict(lambda: 0, {})
        average_scores = {}
        print(len(self.filenames))
        for filename in self.filenames:
            with open(filename) as json_file:
                next_scores = json.load(json_file)
                for key, value in next_scores.items():
                    sum_of_scores[key] += value

        for key, value in sum_of_scores.items():
            average_scores[key] = value / self.number_of_scores
        return average_scores
