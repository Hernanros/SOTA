from typing import List
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

    def create_embedding(self, sentence: list) -> torch.Tensor:

        if not self.downloaded:
            self.download()
                    
        # embed words in sentence
        sent = Sentence(sentence)
        self.embeddings.embed(sent)
        # return average embedding of words in sentence
        return torch.stack([token.embedding for token in sent]).mean(axis=1)

    def calculate_l2_distance(self, candidate: List[str], reference: List[str]) -> float:
        candidate_embedding = self.create_embedding(candidate)
        reference_embedding = self.create_embedding(reference)
        vector = candidate - reference_embedding
        return np.linalg.norm(vector)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        text1 = df[self.text1].str.strip().str.split()
        text2 = df[self.text2].str.strip().str.split()
        print('Calculating Elmo L2 distance')
        pairs = pd.concat([text1, text2], axis=1)
        df['L2_score'] = pairs.progress_apply(lambda row: self.calculate_l2_distance(row[self.text1], row[self.text2]),
                                              axis=1)
        return df
