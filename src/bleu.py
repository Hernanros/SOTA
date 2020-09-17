"""
run bleu feature extract
"""
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu


class Bleu:

    def __init__(self):
        pass


    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        text1 = df['text_1'].str.strip()
        text2 = df['text_2'].str.strip()
        pairs = pd.concat([text1, text2], axis=1)
        df['bleu'] = pairs.apply(lambda row: sentence_bleu([row.text_1], row.text_2), axis=1)
        df['bleu1'] = pairs.apply(lambda row: sentence_bleu([row.text_1], row.text_2), axis=1)
        return df

