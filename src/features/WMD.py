import pandas as pd
import chakin
from gensim.models import KeyedVectors
import os
 
class WMD:

    def __init__(self,vector_path=None):
        
        if not os.path.exists(vector_path):
            print("[WMD] downloading glove")
            #chakin.download(number=16, save_dir=vector_path)  # select GloVe.840B.300d

        self.model = KeyedVectors.load_word2vec_format(vector_path+"/glove.840B.300d.zip", binary=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df['WMD'] = df.apply(lambda x: self.model.wmdistance(x.text_1, x.text_2))
        return df
