"""
calculating wmd using gensim. 
glove zip => txt => gensim => wmdistance

zip => txt => glove2word2vec (gensim) => model.wmdistance
"""

import pandas as pd
import chakin
from gensim.models import KeyedVectors
#import torchtext.vocab as torch_vocab
import os
 
class WMD:

    def __init__(self,vector_path=None):
        self.downloaded = False
        self.vector_path = vector_path
        
    def download(self):
        if not os.path.exists(self.vector_path):
            print("[WMD] downloading glove")
            #chakin.download(number=16, save_dir=self.vector_path)  # select GloVe.840B.300d

        self.model = KeyedVectors.load_word2vec_format(self.vector_path+"/glove.840B.300d.zip", binary=True)
        self.downloaded = True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.downloaded:
            self.download()

        df['WMD'] = df.apply(lambda x: self.model.wmdistance(x.text_1, x.text_2))
        return df
