"""
run bleu feature extract
"""
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from src.features import Metric


class Bleu(Metric):

    def __init__(self, val='text_'):
        super(Bleu, self).__init__(val=val)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        text1 = df[self.text1].str.strip().str.split()
        text2 = df[self.text2].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['bleu'] = pairs.apply(lambda row: sentence_bleu([row[self.text1]], row[self.text2]), axis=1) #Open question whether to keep removing of stopwords or not?
        df['bleu1'] = pairs.apply(lambda row: sentence_bleu([row[self.text1]], row[self.text2]), axis=1)

        return df

