import pandas
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

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        df['L2_score'] = df.apply(lambda x: euclidean(self.create_embedding(x.text_1), self.create_embedding(x.text_2)),
                                  axis=1)
        return df
