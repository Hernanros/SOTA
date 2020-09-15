"""implements ROUGE metrics"""
import pandas as pd
from rouge.rouge import rouge_n_sentence_level
from rouge.rouge import rouge_l_sentence_level


class ROUGE:

    def __init__(self):
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        recall_1_list = []
        recall_2_list = []
        precision_1_list = []
        precision_2_list = []
        F_1_list = []
        F_2_list = []

        recall_l_list = []
        precision_l_list = []
        F_l_list = []

        i = 0
        while i < len(df):
            sentence_ori = df.iloc[i]['text_1']
            sentence_gen = df.iloc[i]['text_2']

            sentence_ori = sentence_ori.split()
            sentence_gen = sentence_gen.split()

            # Calculate ROUGE-1
            recall_1, precision_1, rouge_1 = rouge_n_sentence_level(sentence_gen, sentence_ori, 1)
            # Calculate ROUGE-2
            recall_2, precision_2, rouge_2 = rouge_n_sentence_level(sentence_gen, sentence_ori, 2)
            # Calculate ROUGE-l
            recall_l, precision_l, rouge_l = rouge_l_sentence_level(sentence_gen, sentence_ori)

            recall_1_list.append(recall_1)
            recall_2_list.append(recall_2)
            recall_l_list.append(recall_l)
            precision_1_list.append(precision_1)
            precision_2_list.append(precision_2)
            precision_l_list.append(precision_l)
            F_1_list.append(rouge_1)
            F_2_list.append(rouge_2)
            F_l_list.append(rouge_l)
            i += 1

        df['ROUGE-1 recall'] = recall_1_list
        df['ROUGE-1 precision'] = precision_1_list
        df['ROUGE-1 F'] = F_1_list
        df['ROUGE-2 recall'] = recall_2_list
        df['ROUGE-2 precision'] = precision_2_list
        df['ROUGE-2 F'] = F_2_list
        df['ROUGE-L recall'] = recall_l_list
        df['ROUGE-L precision'] = precision_l_list
        df['ROUGE-L F'] = F_l_list
        return df
