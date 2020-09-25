"""
run glove & fasttext embedding cosine similarity feature extraction

models:
- glove uses torchtext, and predownloaded can be determined using 'vectors_cache'
- fasttext uses gensim downloader, path is always ~/gensim-data. To control it, make a symlink
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

    def __init__(self, glove_path=None):
        self.downloaded = False
        self.glove_path = glove_path

    def download(self):
        self.models = dict(glove=torch_vocab.GloVe(name='twitter.27B', dim=100, cache=self.glove_path),
                           fasttext=api.load("fasttext-wiki-news-subwords-300"))
        self.downloaded = True

    def compute_cs(self, reference: List[str], candidate: List[str], model: str):

<<<<<<< HEAD:src/features/cosine_similarites.py
        reference_vectors = []
        for word in reference:
            if word in self.models[model].wv.vocab:
                reference_vectors.append(self.fasttext.wv[word])
            else:
                pass
        reference_vectors = np.array(reference_vectors)
        reference = reference.strip().split()
        candidate = candidate.strip().split()

        reference_vectors = []
        for word in reference:
            if word in self.models[model].wv.vocab:
                reference_vectors.append(self.fasttext.wv[word])
            else:
                pass
        reference_vectors = np.array(reference_vectors)
=======
        reference_vectors = np.array([self.models[model][word] for word in reference if word in self.models[model].vocab])
>>>>>>> adam/align-features-with-ivan:src/cosine_similarites.py

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

<<<<<<< HEAD:src/features/cosine_similarites.py

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:

        if not self.downloaded:
            self.download()        

        print("cosine_similarites start");

        df['glove_cosine'] = df.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'glove'))
        df['fasttext_cosine'] = df.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'fasttext'))
        return df
        # for i in range(df.shape[0]):
        #     s1 = str(df['text_1'][i])
        #     s2 = str(df['text_2'][i])
        #     df.loc[i, 'glove_allwords'] = self.embedding_cosine_distance(s1, s2, stopwords_remove=False,
        #                                                                  remove_non_model=False, method='glove')
        #     df.loc[i, 'glove_withoutstop'] = self.embedding_cosine_distance(s1, s2, stopwords_remove=True,
        #                                                                     remove_non_model=False, method='glove')
        #     df['ftext_allwords'] = self.embedding_cosine_distance(s1, s2, stopwords_remove=False,
        #                                                           remove_non_model=True, method='fasttext')
        #     df['ftext_withoutstop'] = self.embedding_cosine_distance(s1, s2, stopwords_remove=True,
        #                                                              remove_non_model=True, method='fasttext')

        print("cosine_similarites end");


=======
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df['text_1'].str.strip().str.split()
        text2 = df['text_2'].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['glove_cosine'] = pairs.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'glove'), axis=1)
        df['fasttext_cosine'] = pairs.apply(lambda row: self.compute_cs_word2vec(row.text_1, row.text_2, 'fasttext'), axis=1)
>>>>>>> adam/align-features-with-ivan:src/cosine_similarites.py
        return df


