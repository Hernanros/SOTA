import pandas as pd
import numpy as np
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
import torch
from src.features import Metric


class EuclideanElmoDistance(Metric):

    def __init__(self, val):
        super(EuclideanElmoDistance, self).__init__(val=val)
        self.downloaded = False
        self.embeddings = None

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
        mean = torch.stack([token.embedding for token in sent]).mean(axis=1)
        return mean.mean(axis=0).numpy()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df[self.text1].str.strip().str.split().apply(self.create_embedding)
        text2 = df[self.text2].str.strip().str.split().apply(self.create_embedding)
        vector = text1 - text2
        df['L2_score'] = vector.apply(np.linalg.norm)
        return df
