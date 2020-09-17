import csv
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder(cuda_device=0)

DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence_tokens = row['text1'].strip().split()
        second_sentence_tokens = row['text2'].strip().split()
        pairs.append((first_sentence_tokens, second_sentence_tokens))

pairs = pairs[:10]



first_sentences, second_centences = zip(*pairs)
elmo_embeddings_1 = [s for s in elmo.embed_sentences(first_sentences, batch_size=200)]
elmo_embeddings_2 = [s for s in elmo.embed_sentences(second_centences, batch_size=200)]

l2_scores = []
for first_sentence_embeddings, second_sentence_embeddings in zip(elmo_embeddings_1, elmo_embeddings_2):

    print(first_sentences[0])
    print(first_sentence_embeddings.shape)
    exit()
    first_sentence_embeddings = np.mean(first_sentence_embeddings, 1)
    first_sentence_embeddings = np.mean(first_sentence_embeddings, 0)

    second_sentence_embeddings = np.mean(second_sentence_embeddings, 1)
    second_sentence_embeddings = np.mean(second_sentence_embeddings, 0)

    vector = first_sentence_embeddings - second_sentence_embeddings

    score_l2 = np.linalg.norm(vector)
    l2_scores.append(score_l2)

print(np.mean(l2_scores))
print(np.std(l2_scores))







