import pandas as pd
from src.features import Metric


class NgramOverlap(Metric):

    def __init__(self, n, val='text_'):
        super(NgramOverlap, self).__init__(val=val)
        self.n = n

    def gram_overlap(self, sent_a, sent_b):
        first_sentence_set = set(sent_a)
        second_sentence_set = set(sent_b)
        score_wo = len(first_sentence_set & second_sentence_set) / len(first_sentence_set | second_sentence_set)
        return score_wo

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df['text_1'].str.strip().str.split()
        text2 = df['text_2'].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['1-gram_overlap'] = pairs.apply(lambda row: self.gram_overlap(row.text_1, row.text_2,))
        return df
