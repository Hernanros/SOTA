"""
run bleu feature extract
"""
import pandas
from nltk.translate.bleu_score import sentence_bleu


class Bleu:

    def __init__(self):
        self.txt_col_format = 'text'

    def setTextFormat(self, val):
        self.txt_col_format = val

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:

        # for index, row in df.iterrows():
        #     row[self.txt_col_format+'1'] = row[self.txt_col_format+'1'].strip().split()
        #     row[self.txt_col_format+'2'] = row[self.txt_col_format+'2'].strip().split()
        df[f'{self.txt_col_format}1'] = df[f'{self.txt_col_format}1'].str.strip().str.split()
        df[f'{self.txt_col_format}2'] = df[f'{self.txt_col_format}2'].str.strip().str.split()
        df['bleu'] = df.apply(lambda row: sentence_bleu([row[f'{self.txt_col_format}1']], row[f'{self.txt_col_format}2']), axis=1) #Open question whether to keep removing of stopwords or not?
        df['bleu1'] = df.apply(lambda row: sentence_bleu([row[self.txt_col_format+'1']], row[f'{self.txt_col_format}2']), axis=1)

        return df

