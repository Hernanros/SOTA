"""
run glove & fasttext embedding cosine similarity feature extraction

models:
- glove uses torchtext, and predownloaded can be determined using 'vectors_cache'
- fasttext uses gensim downloader, path is always ~/gensim-data. To control it, make a symlink
"""
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import torchtext.vocab as torch_vocab
from src.features import Metric
from tqdm import tqdm


class CosineSimilarity(Metric):

    def __init__(self, val, glove_path=None):
        super(CosineSimilarity, self).__init__(val)
        self.downloaded = False
        self.glove_path = glove_path
        self.models = {}
        self.vocab = {}

    def download(self):
        self.models = dict(glove=torch_vocab.GloVe(name='twitter.27B', dim=100, cache=self.glove_path),
                           fasttext=api.load("fasttext-wiki-news-subwords-300"))
        self.vocab = dict(glove=self.models['glove'].itos, fasttext=self.models['fasttext'].vocab)
        self.downloaded = True

    def compute_cs(self, reference: List[str], candidate: List[str], model: str):
        reference_vectors = np.array([np.array(self.models[model][word]) for word in reference if word in self.vocab[model]])
        candidate_vectors = np.array([np.array(self.models[model][word]) for word in candidate if word in self.vocab[model]])

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

        score = cosine_similarity(reference_vector, candidate_vector).item()
        return 1 - score

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.downloaded:
            self.download()        

        print("cosine_similarites start")
        tqdm.pandas()
        text1 = df[self.text1].str.strip().str.split()
        text2 = df[self.text2].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['glove_cosine'] = pairs.progress_apply(lambda row: self.compute_cs(row[self.text1], row[self.text2],
                                                                              'glove'), axis=1)
        df['fasttext_cosine'] = pairs.progress_apply(lambda row: self.compute_cs(row[self.text1], row[self.text2],
                                                                                 'fasttext'), axis=1)
        return df
