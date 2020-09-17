import pandas
from bert_score import score


class BertScore:

    def __init__(self):
        pass

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # apply function to all word pairs in dataset
        text1 = df['text_1'].str.strip().tolist()
        text2 = df['text_2'].str.strip().tolist()
        _, _, F1 = score(text1, text2, lang='en')
        df['BertScore'] = F1.numpy()
        return df
