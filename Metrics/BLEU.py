import csv
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

DATA_FILE = '../Datasets/Yelp_Random (1).csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence_tokens = row['text1'].strip().split()
        second_sentence_tokens = row['text2'].strip().split()
        pairs.append((first_sentence_tokens, second_sentence_tokens))


scores_bleu = []
scores_bleu1 = []
for first_sentence_tokens, second_sentence_tokens in pairs:
    score_bleu = sentence_bleu([first_sentence_tokens], second_sentence_tokens)
    scores_bleu.append(score_bleu)

    score_bleu1 = sentence_bleu([first_sentence_tokens], second_sentence_tokens, weights=(1, 0, 0, 0))
    scores_bleu1.append(score_bleu1)


print(np.mean(scores_bleu))
print(np.std(scores_bleu))

print(np.mean(scores_bleu1))
print(np.std(scores_bleu1))