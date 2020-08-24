from nltk.translate import chrf_score


class chrFScore:

    def __init__(self):
        pass

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        df['chrfScore'] = df.apply(lambda x: chrf_score.sentence_chrf(x.text_1, x.text_2,), axis=1)
        return df
