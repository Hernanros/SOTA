import pandas as pd
import chakin
from gensim.models import KeyedVectors

vector_dir = '../data/pretrained_vectors'


class WMD:

    def __init__(self):
        self.vector_path = chakin.download(number=16, save_dir=vector_dir)  # select GloVe.840B.300d
        self.model = KeyedVectors.load_word2vec_format(self.vector_path)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df['WMD'] = df.apply(lambda x: self.model.wmdistance(x.text_1, x.text_2))
        return df
