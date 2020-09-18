import csv
import numpy as np

DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence_tokens = row['text1'].strip().split()
        second_sentence_tokens = row['text2'].strip().split()
        pairs.append((first_sentence_tokens, second_sentence_tokens))


scores_wo = []
for first_sentence_tokens, second_sentence_tokens in pairs:
    first_sentence_set = set(first_sentence_tokens)
    second_sentence_set = set(second_sentence_tokens)
    score_wo = len(first_sentence_set & second_sentence_set) / len(first_sentence_set | second_sentence_set)
    scores_wo.append(score_wo)


print(np.mean(scores_wo))
print(np.std(scores_wo))