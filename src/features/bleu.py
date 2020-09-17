"""
run bleu feature extract
"""
import pandas
from nltk.translate.bleu_score import sentence_bleu


class Bleu:

    def __init__(self):
        pass


    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:

        df['text1'] = df.text1.str.strip().str.split()
        df['text2'] = df.text2.str.strip().str.split()
        df['bleu'] = df.apply(lambda row: sentence_bleu([row.text_1], row.text_2), axis=1) #Open question whether to keep removing of stopwords or not?
        df['bleu1'] = df.apply(lambda row: sentence_bleu([row.text_1], row.text_2), axis=1)
        # df['bleu_allwords'] = 0
        # df['bleu_withoutstop'] = 0
        # for i in range(df.shape[0]):
        #     s1 = str(df['text_1'][i])
        #     s2 = str(df['text_2'][i])
        #     #print(s1,",",s2)
        #     df.loc[i, 'bleu_allwords'] = self.BLEU1score(s1, s2, stopwords_remove=False)
        #     df.loc[i, 'bleu_withoutstop'] = self.BLEU1score(s1, s2, stopwords_remove=True)

        return df

