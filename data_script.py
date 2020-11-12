import pandas as pd
import os
import numpy as np
import pickle
import glob

for i, file in enumerate(glob.glob('data/raw_data/*.csv')):
    if i==0:
        df = pd.read_csv(file, index_col = 0)
        df['dataset'] = file[5:-4].lower()
    else:
        temp = pd.read_csv(file, index_col = 0)
        temp['dataset'] = file[5:-4].lower()
        df = pd.concat((df,temp),axis = 0)
df['random'] = df.dataset.apply(lambda x: 'random' in  x).astype(int)
df['duration'] = pd.to_datetime(df.SubmitTime)-pd.to_datetime(df.AcceptTime)