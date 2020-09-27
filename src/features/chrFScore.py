import pandas as pd
from nltk.translate import chrf_score
from src.features import Metric
from tqdm import tqdm


class chrFScore(Metric):

    def __init__(self, val):
        super(chrFScore, self).__init__(val=val)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # apply function to all word pairs in dataset
        tqdm.pandas()
        text1 = df[self.text1].str.strip()
        text2 = df[self.text2].str.strip()
        pairs = pd.concat([text1, text2], axis=1)
        df['chrfScore'] = pairs.progress_apply(lambda row: chrf_score.sentence_chrf(row[self.text1], row[self.text2]),
                                               axis=1)
        return df
