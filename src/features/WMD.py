import pandas as pd
import chakin
from gensim.models import KeyedVectors

 
class WMD:

    def __init__(self,vector_dir=None):
        self.vector_path = chakin.download(number=16, save_dir=vector_dir)  # select GloVe.840B.300d
        print(f"[WMD] vector_path:{self.vector_path}")
        self.model = KeyedVectors.load_word2vec_format(self.vector_path, binary=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df['WMD'] = df.apply(lambda x: self.model.wmdistance(x.text_1, x.text_2))
        return df
