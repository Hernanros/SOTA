import pandas as pd
import chakin
from gensim.models import KeyedVectors
import nltk.corpus

vector_dir = '../data/pretrained_vectors'


class WMD:

    def __init__(self):
        self.vector_path = chakin.download(number=16, save_dir=vector_dir)  # select GloVe.840B.300d
        self.model = KeyedVectors.load_word2vec_format(self.vector_path)
        self.stopwords = nltk.corpus.stopwords.words('english')

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df.text_1.str.lower().str.strip().str.split()
        text2 = df.text_2.str.lower().str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['WMD'] = pairs.apply(lambda x: self.model.wmdistance(x.text_1, x.text_2))
        return df
