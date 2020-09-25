import pandas as pd
import numpy as np
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from scipy.spatial.distance import euclidean
import torch

class EuclideanElmoDistance:

    def __init__(self):
        self.downloaded = False

    def download(self):        
        self.embeddings = ELMoEmbeddings()
        self.downloaded = True

    def create_embedding(self, sentence: list) -> torch.Tensor:

        if not self.downloaded:
            self.download()
                    
        # embed words in sentence
        sent = Sentence(sentence)
        self.embeddings.embed(sent)
        # return average embedding of words in sentence
        return torch.stack([token.embedding for token in sent]).mean(axis=0)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df['text_1'].str.strip().apply(self.create_embedding)
        text2 = df['text_2'].str.strip().apply(self.create_embedding)
        vector = text1 - text2
        df['L2_score'] = vector.apply(np.linalg.norm)
            # pairs.apply(lambda row: euclidean(row.text_1, row.text_2), axis=1)
        return df
