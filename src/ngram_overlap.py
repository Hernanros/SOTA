import pandas as pd
from nltk.tokenize import RegexpTokenizer
import numpy as np


class NgramOverlap:

    def __init__(self, n):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.n = n

    def ngrammer(self, sent, gram):
        grams = []
        tokenized = self.tokenizer.tokenize(sent)
        if len(tokenized) >= gram:
            for i in range(len(tokenized) - gram + 1):
                grams.append(' '.join(tokenized[i:i + gram]))
        return grams

    def gram_overlap(self, sent_a, sent_b, gram):
        grams_a = set(self.ngrammer(sent_a, gram))
        grams_b = set(self.ngrammer(sent_b, gram))
        intersection = np.sum([gram in grams_b for gram in grams_a])
        return 2 * intersection / (len(grams_a) + len(grams_b))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for n in range(self.n):
            df[f'{n + 1}-gram_overlap'] = df.apply(lambda x: self.gram_overlap(x.text_1, x.text_2, n + 1))
        return df
