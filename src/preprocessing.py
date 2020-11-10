import unicodedata
import contractions
import pandas as pd


def text_preprocessing(text_series: pd.Series) -> pd.Series:
    decoded = text_series.apply(lambda text: unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode())
    uncontracted = decoded.apply(lambda text: contractions.fix(text, leftovers=True, slang=True))
    no_punctuation = uncontracted.str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    lowercase = no_punctuation.str.lower()
    return lowercase

