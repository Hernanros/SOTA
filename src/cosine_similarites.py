"""
run glove & fasttext embedding cosine similarity feature extraction
"""

import re
import numpy as np
import scipy
from nltk.corpus import stopwords
import gensim.downloader as api
import nltk
import torchtext.vocab as torch_vocab


class CosineSimilarity:

    def __init__(self):
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
        self.models = dict(glove=torch_vocab.GloVe(name='twitter.27B', dim=100),
                           fasttext=api.load("fasttext-wiki-news-subwords-300"))

    def preprocess(self, raw_text: str, stopwords_remove=True, remove_non_model=False, method='glove') -> list:
        assert method in ['glove', 'fasttext'], 'Method given is not recognized, must be "glove" or "fasttext"'
        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", str(raw_text))
        # convert to lower case and split
        words = letters_only_text.lower().split()
        # print(words)

        if remove_non_model:
            words = list(filter(lambda x: x in self.models[method].vocab, words))

        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))

        return cleaned_words if stopwords_remove else words

    def embedding_cosine_distance(self, s1: str, s2: str, method='glove',
                                  stopwords_remove=True, remove_non_model=False) -> float:
        assert method in ['glove', 'fasttext'], 'Method given is not recognized, must be "glove" or "fasttext"'
        vector_1 = [self.models[method][word] for word in self.preprocess(s1, stopwords_remove=stopwords_remove,
                                                                          remove_non_model=remove_non_model,
                                                                          method=method)]
        vector_2 = [self.models[method][word] for word in self.preprocess(s2, stopwords_remove=stopwords_remove,
                                                                          remove_non_model=remove_non_model,
                                                                          method=method)]

        if len(vector_1) == 0 or len(vector_2) == 0:
            return -1

        # print(np.stack(vector_1))
        vector_1 = np.mean(np.stack(vector_1), axis=0)
        vector_2 = np.mean(np.stack(vector_2), axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        return round((1 - cosine) * 100, 2)

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:

        df['glove_allwords'] = df.apply(lambda x: self.embedding_cosine_distance(str(x.text_1), str(x.text_2),
                                                                                 stopwords_remove=False,
                                                                                 remove_non_model=False,
                                                                                 method='glove'), axis=1)
        df['glove_withoutstop'] = df.apply(lambda x: self.embedding_cosine_distance(str(x.text_1), str(x.text_2),
                                                                                    stopwords_remove=True,
                                                                                    remove_non_model=False,
                                                                                    method='glove'), axis=1)
        df['ftext_allwords'] = df.apply(lambda x: self.embedding_cosine_distance(str(x.text_1), str(x.text_2),
                                                                                 stopwords_remove=False,
                                                                                 remove_non_model=True,
                                                                                 method='fasttext'), axis=1)
        df['ftext_withoutstop'] = df.apply(lambda x: self.embedding_cosine_distance(str(x.text_1), str(x.text_2),
                                                                                    stopwords_remove=True,
                                                                                    remove_non_model=True,
                                                                                    method='fasttext'), axis=1)

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
        return df

