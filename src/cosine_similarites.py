"""
run glove & fasttext embedding cosine similarity feature extraction
"""
from typing import List
import pandas as pd
import re
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import torchtext.vocab as torch_vocab


class CosineSimilarity:

    def __init__(self):
        self.models = dict(fasttext=api.load("fasttext-wiki-news-subwords-300"),
                           glove=torch_vocab.GloVe(name='twitter.27B', dim=100))
        self.models['fasttext'].init_sims(replace=True)

    def compute_cs(self, reference: List[str], candidate: List[str], model: str):

        reference_vectors = np.array([self.models[model][word] for word in reference if word in self.models[model].vocab])

        candidate_vectors = np.array([self.models[model][word] for word in candidate if word in self.models[model].vocab])

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

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df['text_1'].str.strip().str.split()
        text2 = df['text_2'].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['glove_cosine'] = pairs.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'glove'), axis=1)
        df['fasttext_cosine'] = pairs.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'fasttext'), axis=1)
        return df


