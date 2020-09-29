import pandas as pd
import numpy as np
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
import torch
from src.features import Metric
from tqdm import tqdm


class EuclideanElmoDistance(Metric):

    def __init__(self, val):
        super(EuclideanElmoDistance, self).__init__(val=val)
        self.downloaded = False
        self.embeddings = None

    def download(self):        
        self.embeddings = ELMoEmbeddings()
        self.downloaded = True

    def create_embedding(self, sentence: list):

        if not self.downloaded:
            self.download()
                    
        # embed words in sentence
        sent = Sentence(sentence)
        self.embeddings.embed(sent)
        # return average embedding of words in sentence
        return torch.stack([token.embedding for token in sent]).mean(axis=1)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        print(f'Creating embeddings for {self.text1} column')
        text1 = df[self.text1].str.strip().str.split().progress_apply(self.create_embedding, axis=1)
        print(f'Creating embeddings for {self.text2} column')
        text2 = df[self.text2].str.strip().str.split().progress_apply(self.create_embedding, axis=1)
        print('Calculating Elmo L2 distance')
        vector = text1 - text2
        df['L2_score'] = vector.progress_apply(np.linalg.norm, axis=1)
        return df
