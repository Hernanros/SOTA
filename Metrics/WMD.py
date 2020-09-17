import csv
import numpy as np
import nltk.corpus
from gensim.models import KeyedVectors

DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

stopwords = nltk.corpus.stopwords.words('english')

model = KeyedVectors.load_word2vec_format(
    '../WordVectors/GoogleNews-vectors-negative300 (1).bin.gz',
    binary=True)
model.init_sims(replace=True)

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence_tokens = row['text1'].lower().strip().split()
        first_sentence_tokens = [w for w in first_sentence_tokens if w not in stopwords]
        second_sentence_tokens = row['text2'].lower().strip().split()
        second_sentence_tokens = [w for w in second_sentence_tokens if w not in stopwords]
        pairs.append((first_sentence_tokens, second_sentence_tokens))


scores = []
for first_sentence_tokens, second_sentence_tokens in pairs:
    distance = model.wmdistance(first_sentence_tokens, second_sentence_tokens)
    if str(distance) != 'inf':
        scores.append(distance)

print(np.mean(scores))
print(np.std(scores))
