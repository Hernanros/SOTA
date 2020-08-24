from bert_score import score


class BertScore:

    def __init__(self):
        pass

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        df['BertScore'] = score(df['text_1'].to_list(), df['text_2'].to_list(), lang="en", verbose=False)

        return df