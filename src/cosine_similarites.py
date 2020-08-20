#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd   
import re
import numpy as np
import matplotlib.pyplot as plt
import os   
import scipy
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

from nltk import pos_tag, word_tokenize
import nltk
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
import torchtext.vocab as torch_vocab
import argparse
import torch

import pickle

#%%

with open('data\combined_data.pickle', 'rb') as handle:
    df = pickle.load(handle)


# In[2]:


model = torch_vocab.GloVe(name='twitter.27B',dim=100)
nltk.download()


# In[3]:



df['glove_allwords'] = 0
df['glove_withoutstop'] = 0



# In[18]:


def preprocess(raw_text,stopwords_remove=True,remove_non_model=False):
    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", str(raw_text))
    # convert to lower case and split 
    words = letters_only_text.lower().split()
    #print(words)

    if remove_non_model:
        words = list(filter(lambda x: x in model.vocab, words))

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))
    
    return cleaned_words if stopwords_remove else words

def cosine_distance_wordembedding_method(s1, s2,stopwords_remove=True,remove_non_model=False):

    vector_1 = [model[word] for word in preprocess(s1,stopwords_remove,remove_non_model)]
    vector_2 = [model[word] for word in preprocess(s2,stopwords_remove,remove_non_model)]
    
    if len(vector_1)==0 or len(vector_2)==0:
        return -1
    
    #print(np.stack(vector_1))
    vector_1 = np.mean(np.stack(vector_1),axis=0)
    vector_2 = np.mean(np.stack(vector_2),axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1-cosine)*100,2)


# In[7]:


scores = []
for i in range(df.shape[0]):
    s1 = str(df['text_1'][i])
    s2 = str(df['text_2'][i])
    #print(s1,",",s2)
    dis = cosine_distance_wordembedding_method(s1,s2,stopwords_remove=False)
    df.loc[i,'glove_allwords']=dis


# In[8]:


scores = []
for i in range(df.shape[0]):
    s1 = str(df['text_1'][i])
    s2 = str(df['text_2'][i])
    #print(s1,",",s2)
    dis = cosine_distance_wordembedding_method(s1,s2,stopwords_remove=True)
    df.loc[i,'glove_withoutstop']=dis



# In[10]:



model = KeyedVectors.load_word2vec_format('e:\\WORK\\ML\\data\\embed\\fasttrack\\cc.en.300.vec')


# In[14]:


df['ftext_allwords'] = 0
df['ftext_withoutstop'] = 0


# In[20]:


scores = []
for i in range(df.shape[0]):
    s1 = str(df['text_1'][i])
    s2 = str(df['text_2'][i])
    #print(s1,",",s2)
    dis = cosine_distance_wordembedding_method(s1,s2,stopwords_remove=False,remove_non_model=True)
    df.loc[i,'ftext_allwords']=dis


# In[21]:


scores = []
for i in range(df.shape[0]):
    s1 = str(df['text_1'][i])
    s2 = str(df['text_2'][i])
    #print(s1,",",s2)
    dis = cosine_distance_wordembedding_method(s1,s2,stopwords_remove=True,remove_non_model=True)
    df.loc[i,'ftext_withoutstop']=dis


# In[23]:



with open('data\combined_data.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#check
with open('data\combined_data.pickle', 'rb') as handle:
    b = pickle.load(handle)

