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
            else:
                self.convert_to_w2v()
        else:
            self.model = KeyedVectors.load_word2vec_format(self.glove_w2v_format)

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
