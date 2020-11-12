"""implements ROUGE metrics"""
import pandas as pd
import rouge
from src.features import Metric
from tqdm import tqdm


class ROUGE(Metric):

    def __init__(self, val, stopwords=True):
        super(ROUGE, self).__init__(val=val, stopwords=stopwords)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-l']
        try:
            df.drop(columns=metric_names, inplace=True)
        except KeyError:
            pass
        pairs = df.groupby('pair_id')[[self.text1, self.text2]].last()
        if self.stopwords:
            pairs[self.text1] = self.remove_stopwords(pairs[self.text1]).str.strip()
            pairs[self.text2] = self.remove_stopwords(pairs[self.text2]).str.strip()
        else:
            pairs[self.text1] = pairs[self.text1].str.strip()
            pairs[self.text2] = pairs[self.text2].str.strip()

        evaluator = rouge.Rouge(metrics=['rouge-1'])
        pairs[metric_names[0]] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                                       [row[self.text2]]
                                                                                       )[0]['rouge-1']['f'], axis=1)
        evaluator = rouge.Rouge(metrics=['rouge-2'])
        pairs[metric_names[1]] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                                       [row[self.text2]]
                                                                                       )[0]['rouge-2']['f'], axis=1)
        evaluator = rouge.Rouge(metrics=['rouge-l'])
        pairs[metric_names[2]] = pairs.progress_apply(lambda row: evaluator.get_scores([row[self.text1]],
                                                                                       [row[self.text2]]
                                                                                       )[0]['rouge-l']['f'], axis=1)
        df = df.merge(pairs[metric_names], how='left', left_on='pair_id', right_index=True)
        return df
