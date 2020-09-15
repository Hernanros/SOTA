import pandas
from bert_score import score


class BertScore:

    def __init__(self):
        pass

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        df['text1'] = df['text_1'].str.strip()
        df['text_2'] = df['text_2'].str.strip()
        _, _, F1 = score(df['text1'].tolist(), df['text2'].tolist(), lang='en')
        df['BertScore'] = F1.numpy()
        return df