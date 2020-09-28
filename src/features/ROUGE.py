"""implements ROUGE metrics"""
import pandas as pd
import rouge
import numpy as np
from src.features import Metric
from tqdm import tqdm


class ROUGE(Metric):

    def __init__(self, val):
        super(ROUGE, self).__init__(val=val)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        text1 = df[self.text1].str.strip()
        text2 = df[self.text2].str.strip()
        pairs = pd.concat([text1, text2], axis=1)
        evaluator = rouge.Rouge(metrics=['rouge-1'])
        df['ROUGE-1'] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                              [row[self.text2]])[0]['rouge-1']['f'])
        evaluator = rouge.Rouge(metrics=['rouge-2'])
        df['ROUGE-2'] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                              [row[self.text2]])[0]['rouge-2']['f'])
        evaluator = rouge.Rouge(metrics=['rouge-l'])
        df['ROUGE-l'] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                              [row[self.text2]])[0]['rouge-l']['f'])
        return df
