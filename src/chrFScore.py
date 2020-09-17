import pandas as pd
from nltk.translate import chrf_score


class chrFScore:

    def __init__(self):
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # apply function to all word pairs in dataset
        text1 = df['text_1'].str.strip()
        text2 = df['text_2'].str.strip()
        pairs = pd.concat([text1, text2], axis=1)
        df['chrfScore'] = pairs.apply(lambda row: chrf_score.sentence_chrf(row.text_1, row.text_2,), axis=1)
        return df
