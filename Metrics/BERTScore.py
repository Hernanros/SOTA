import csv
import numpy as np
from bert_score import score


DATA_FILE = '../Datasets/Bible_Random.csv'

first_sentences = []
second_sentences = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence = row['text1'].strip()
        second_sentence = row['text2'].strip()
        first_sentences.append(first_sentence)
        second_sentences.append(second_sentence)

P, R, F1 = score(first_sentences, second_sentences, lang='en')
scores = F1.numpy()

print(np.mean(scores))
print(np.std(scores))