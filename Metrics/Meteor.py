import csv
from nltk.translate.meteor_score import meteor_score
import numpy as np

DATA_FILE = '../Datasets/Paralex_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence = row['text1'].strip()
        second_sentence = row['text2'].strip()
        pairs.append((first_sentence, second_sentence))


scores_meteor = []
for first_sentence, second_sentence in pairs:
    score_meteor = meteor_score([first_sentence], second_sentence)
    scores_meteor.append(score_meteor)


print(np.mean(scores_meteor))
print(np.std(scores_meteor))
