import pandas as pd
import nltk
stopwords = nltk.corpus.stopwords.words('english')


class Metric:
    def __init__(self, val, stopwords=False):
        self.txt_col_format = val
        self.text2 = f'{self.txt_col_format}2'
        self.text1 = f'{self.txt_col_format}1'
        self.stopwords = stopwords

    @staticmethod
    def remove_stopwords(text_series: pd.Series) -> pd.Series:
        stopwords_to_remove = ['against', 'no', 'nor', 'not']
        new_stopwords = [word for word in stopwords if word not in stopwords_to_remove]
        no_stopwords = text_series.apply(lambda text: ' '.join([word for word in text.split() if word not in new_stopwords])).str.strip()
        return no_stopwords

