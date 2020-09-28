"""
calculating wmd using gensim. 
glove zip => txt => gensim => wmdistance

zip => txt => glove2word2vec (gensim) => model.wmdistance
"""

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


class WMD(Metric):

    def __init__(self, val, vector_path=None):
        super(WMD, self).__init__(val)
        self.downloaded = False
        self.vector_path = vector_path
        self.model = None
        
    def download(self):
        w2vfile_path = os.path.join(self.vector_path, 'glove.840B.300d.txt')
        glove_w2v_format = os.path.join(self.vector_path, 'glove.840B.300d.w2v.txt')
        zip_path = os.path.join(self.vector_path, 'glove.840B.300d.zip')
        if not os.path.exists(glove_w2v_format):
            if not os.path.exists(zip_path):
                print("[WMD] downloading glove")
                chakin.download(number=16, save_dir=self.vector_path)  # select GloVe.840B.300d

            if not os.path.exists(w2vfile_path):
                print("[WMD] unzipping")
                zip_ref = zipfile.ZipFile(zip_path)
                zip_ref.extractall(self.vector_path)
                zip_ref.close()

        if not os.path.exists(glove_w2v_format):
            print("[WMD] glove=>w2v")
            glove_file = datapath(w2vfile_path)
            glove_w2v_format = get_tmpfile(glove_w2v_format)
            _ = glove2word2vec(glove_file, glove_w2v_format)
        
        self.model = KeyedVectors.load_word2vec_format(glove_w2v_format)
        self.downloaded = True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        if not self.downloaded:
            self.download()
        text1 = df[self.text1].str.lower().str.strip().str.split()
        text2 = df[self.text2].str.lower().str.strip().str.split()
        pairs = pd.concat([text1, text2], axis=1)
        df['WMD'] = pairs.progress_apply(lambda x: self.model.wmdistance(x[self.text1], x[self.text2]), axis=1)
        return df
