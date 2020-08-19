"""
run bleu feature extract
"""

import pandas as pd   
import re
import numpy as np
import os   
import scipy
from nltk.corpus import stopwords

from nltk import pos_tag, word_tokenize
import nltk


class Bleu:

    def __init__(self):
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

    def preprocess(self,raw_text,stopwords_remove=True):
        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", str(raw_text))
        # convert to lower case and split 
        words = letters_only_text.lower().split()
        #print(words)
        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))
        return cleaned_words if stopwords_remove else words

    def BLEU1score(self,s1, s2,stopwords_remove=True):
        hypothesis = [word for word in self.preprocess(s1,stopwords_remove)]
        reference = [word for word in self.preprocess(s2,stopwords_remove)]
        #print(hypothesis,",",reference)
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
        return BLEUscore


    def run(self,df):

        df['bleu_allwords'] = 0
        df['bleu_withoutstop'] = 0

        for i in range(df.shape[0]):
            s1 = str(df['text_1'][i])
            s2 = str(df['text_2'][i])
            #print(s1,",",s2)
            dis = self.BLEU1score(s1,s2,stopwords_remove=False)
            df.loc[i,'bleu_allwords']=dis

        for i in range(df.shape[0]):
            s1 = str(df['text_1'][i])
            s2 = str(df['text_2'][i])
            #print(s1,",",s2)
            dis = self.BLEU1score(s1,s2,stopwords_remove=True)
            df.loc[i,'bleu_withoutstop']=dis

        return df

