"""implements ROUGE metrics"""
import pandas as pd
import rouge
import numpy as np

class ROUGE:

    def __init__(self):
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        scores_rouge1 = []
        scores_rouge2 = []
        scores_rougel = []
    
        evaluator = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])

        i = 0
        while i < len(df):
            sentence_ori = df.iloc[i]['text_1'].strip()
            sentence_gen = df.iloc[i]['text_2'].strip()

            scores = evaluator.get_scores([sentence_ori], [sentence_gen])
            scores_rouge1.append(scores[0]['rouge-1']['f'])
            scores_rouge2.append(scores[0]['rouge-2']['f'])
            scores_rougel.append(scores[0]['rouge-l']['f'])

            i += 1

        df['ROUGE-1 mean'] = np.mean(scores_rouge1)
        df['ROUGE-1 std'] = np.std(scores_rouge1)
        df['ROUGE-2 mean'] = np.mean(scores_rouge2)
        df['ROUGE-2 std'] = np.std(scores_rouge2)
        df['ROUGE-L mean'] = np.mean(scores_rougel)
        df['ROUGE-L stg'] = np.std(scores_rougel)

        return df
