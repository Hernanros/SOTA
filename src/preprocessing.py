import unicodedata
import contractions
import pandas as pd
import numpy as np
from tqdm import tqdm


def text_preprocessing(text_series: pd.Series) -> pd.Series:
    tqdm.pandas(desc=f'Preprocessing {text_series.name}')
    decoded = text_series.progress_apply(lambda text: unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode())
    uncontracted = decoded.progress_apply(lambda text: contractions.fix(text, leftovers=True, slang=True))
    no_punctuation = uncontracted.str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    lowercase = no_punctuation.str.lower().replace(r'^\s*$', np.nan, regex=True)
    return lowercase

