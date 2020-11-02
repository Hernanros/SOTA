"""
calculating wmd using gensim. 
glove zip => txt => gensim => wmdistance

zip => txt => glove2word2vec (gensim) => model.wmdistance
"""
import nltk
import pandas as pd
import chakin
from gensim.models import KeyedVectors
#import torchtext.vocab as torch_vocab
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import zipfile
from src.features import Metric
import os
from tqdm import tqdm
import string 
import re


nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

class WMD(Metric):

    def __init__(self, val, vector_path=None):
        super(WMD, self).__init__(val)
        self.downloaded = False
        self.vector_path = vector_path
        self.model = None
        self.w2vfile_path = os.path.join(self.vector_path, 'glove.840B.300d.txt')
        self.glove_w2v_format = os.path.join(self.vector_path, 'glove.840B.300d.w2v.txt')
        self.zip_path = os.path.join(self.vector_path, 'glove.840B.300d.zip')

    def convert_to_w2v(self):
        print("[WMD] glove=>w2v")
        glove_file = datapath(self.w2vfile_path)
        glove_w2v_format = get_tmpfile(self.glove_w2v_format)
        _ = glove2word2vec(glove_file, glove_w2v_format)

    def download_vectors(self):
        print("[WMD] downloading glove")
        chakin.download(number=16, save_dir=self.vector_path)  # select GloVe.840B.300d

    def unzip_vectors(self):
        print("[WMD] unzipping")
        zip_ref = zipfile.ZipFile(self.zip_path)
        zip_ref.extractall(self.vector_path)
        zip_ref.close()

    def download(self):
        if not os.path.exists(self.glove_w2v_format):
            if not os.path.exists(self.w2vfile_path):
                if not os.path.exists(self.zip_path):
                    self.download_vectors()
                else:
                    self.unzip_vectors()
            self.convert_to_w2v()
        self.downloaded = True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df [~((df[self.text1].apply(lambda x: len(re.findall('\[math\]',x)))>0)|(df[self.text2].apply(lambda x: len(re.findall('\[math\]',x)))>0))]
        
        remove = string.punctuation
        pattern = r"[{}]".format(remove) # create the pattern
        
        
        
        tqdm.pandas()
        if not self.downloaded:
            self.download()
        
        print("[WMD] load model")
        self.model = KeyedVectors.load_word2vec_format(self.glove_w2v_format)
        print("[WMD] model loaded")
        metric_names = ['WMD']
        try:
            df.drop(columns=metric_names, inplace=True)
        except KeyError:
            pass
        
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        pairs[self.text1] = pairs[self.text1].apply(lambda x: [word for word in re.sub(pattern,"",x).lower().strip().split() if word not in stopwords])
        pairs[self.text2] = pairs[self.text2].apply(lambda x: [word for word in re.sub(pattern,"",x).lower().strip().split() if word not in stopwords])
        print("[WMD] after update pairs" )
        pairs[metric_names[0]] = pairs.progress_apply(lambda x: self.model.wmdistance(x[self.text1], x[self.text2]),
                                                      axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
