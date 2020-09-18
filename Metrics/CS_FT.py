import csv
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.metrics.pairwise import cosine_similarity


model = FastText.load_fasttext_format('C:\\Users\\Stiv\\PycharmProjects\\MetricsProject\\word_embeddings\\cc.en.300.bin')
model.init_sims(replace=True)

def compute_cs_word2vec(reference, candidate):
    reference = reference.strip().split()
    candidate = candidate.strip().split()

    reference_vectors = []
    for word in reference:
        if word in model.wv.vocab:
            reference_vectors.append(model.wv[word])
        else:
            pass
    reference_vectors = np.array(reference_vectors)


    candidate_vectors = []
    for word in candidate:
        if word in model.wv.vocab:
            candidate_vectors.append(model.wv[word])
        else:
            pass
    candidate_vectors = np.array(candidate_vectors)


    try:
        min_reference_vector = np.min(reference_vectors, axis=0)
        min_candidate_vector = np.min(candidate_vectors, axis=0)
    except:
        return None

    mean_reference_vector = np.mean(reference_vectors, axis=0)
    mean_candidate_vector = np.mean(candidate_vectors, axis=0)

    max_reference_vector = np.max(reference_vectors, axis=0)
    max_candidate_vector = np.max(candidate_vectors, axis=0)


    reference_vector = np.concatenate((min_reference_vector, mean_reference_vector, max_reference_vector))
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    reference_vector = np.expand_dims(reference_vector, axis=0)

    candidate_vector = np.concatenate((min_candidate_vector, mean_candidate_vector, max_candidate_vector))
    candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)
    candidate_vector = np.expand_dims(candidate_vector, axis=0)

    score = cosine_similarity(reference_vector, candidate_vector)[0][0]
    return 1 - score


DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence = row['text1']
        second_sentence = row['text2']
        pairs.append((first_sentence, second_sentence))


scores_cs = []
for first_sentence, second_sentence in pairs:
    score_cs = compute_cs_word2vec(first_sentence, second_sentence)
    if score_cs != None:
        scores_cs.append(score_cs)


print(np.mean(scores_cs))
print(np.std(scores_cs))