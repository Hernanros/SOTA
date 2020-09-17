import pandas
from nltk.translate import chrf_score


class chrFScore:

    def __init__(self):
        pass

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        df['text1'] = df['text1'].str.strip().str.split()
        df['text2'] = df['text2'].str.strip().str.split()
        df['chrfScore'] = df.apply(lambda row: chrf_score.sentence_chrf(row.text_1, row.text_2,), axis=1)
        return df
