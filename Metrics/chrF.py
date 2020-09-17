import csv
from nltk.translate.chrf_score import sentence_chrf
import numpy as np

DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence_tokens = row['text1'].strip().split()
        second_sentence_tokens = row['text2'].strip().split()
        pairs.append((first_sentence_tokens, second_sentence_tokens))


scores_chrf = []
for first_sentence_tokens, second_sentence_tokens in pairs:
    score_chrf = sentence_chrf(first_sentence_tokens, second_sentence_tokens)
    scores_chrf.append(score_chrf)


print(np.mean(scores_chrf))
print(np.std(scores_chrf))