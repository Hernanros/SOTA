import pandas as pd
from src.features import Metric
from tqdm import tqdm


class NgramOverlap(Metric):

    def __init__(self, n, val='text_'):
        super(NgramOverlap, self).__init__(val=val)
        self.n = n

    @staticmethod
    def gram_overlap(sent_a, sent_b):
        first_sentence_set = set(sent_a)
        second_sentence_set = set(sent_b)
        score_wo = len(first_sentence_set & second_sentence_set) / len(first_sentence_set | second_sentence_set)
        return score_wo

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        text1 = df['text_1'].str.strip().str.split()
        text2 = df['text_2'].str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['1-gram_overlap'] = pairs.progress_apply(lambda row: self.gram_overlap(row[self.text1], row[self.text2]),
                                                    axis=1)
        return df
