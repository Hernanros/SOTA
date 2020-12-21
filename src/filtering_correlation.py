'''
All of the functionality needed to test whether the filtering affected the correlation.
'''

import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import pickle

MODULE_PATH = Path(__file__).resolve().parents[1].resolve()
DATA_PATH = MODULE_PATH / 'data' / 'datasets'
ba_path =  MODULE_PATH / 'data' / 'bad_annotators' / 'combined_ba.pickle'

METRICS = ['bleu', 
           'bleu1',
           'glove_cosine',
           'fasttext_cosine',
           'BertScore',
           'chrfScore',
           'POS Dist score',
           '1-gram_overlap',
           'ROUGE-1',
           'ROUGE-2',
           'ROUGE-l',
           'L2_score',
           'WMD']

def get_baseline_correlation(df, sorted = True):
    mean_label = df.groupby("pair_id")['label'].mean()
    metric_columns = df.groupby("pair_id")[METRICS].mean()
    metric_columns["label"] = mean_label
    if sorted:
        baseline_metrics = metric_columns.corr().abs().iloc[-1].sort_values(ascending=False)[1:]
    else:
        baseline_metrics = metric_columns.corr().abs().iloc[-1,:-1]
    return pd.Series(baseline_metrics)

def get_filtering_correlation(df, ba_combined, baseline_metrics = None, percentage_increase = True):
    '''
    Get the correlation score based off the filtering heuristic.
    '''
    if baseline_metrics is None:
        df_correlation = get_baseline_correlation(df, sorted=False)

    heuristics = list(ba_combined.keys())[:-1] #ignore the combined key (all ba), we cover that when we take all of the heuristics
    all_heuristics = [c for i in range(len(heuristics)) for c in itertools.combinations(heuristics,i+1)]

    df_filtering = pd.DataFrame()
    for keys in all_heuristics:
        ba = []
        for key in keys:
            ba += ba_combined[key]
        ba = list(set(ba))
        df_ba = df[~df.annotator.isin(ba)]
        mean_label = df_ba.groupby("pair_id")['label'].mean()
        metric_columns = df_ba.groupby("pair_id")[METRICS].mean()
        metric_columns["label"] = mean_label

        if percentage_increase:
            corr_abs = pd.Series(metric_columns.corr().abs().iloc[-1,:-1])
            diff_score = (corr_abs.subtract(df_correlation))
            filtered_correlation = np.round((diff_score / df_correlation) * 100,3)
            filtered_correlation.name = str(keys)
        else:
            filtered_correlation = (metric_columns.corr().abs().iloc[-1].sort_values(ascending=False)[1:] - df_correlation.iloc[:,0])
            filtered_correlation.name = str(keys)
            index = filtered_correlation.index

        df_filtering = pd.concat((df_filtering, filtered_correlation),axis=1)

    df_correlation = pd.concat((df_correlation, df_filtering),axis=1)
    index = df_filtering.index

    if percentage_increase:
        index.name = "Percentage Increase/Decrease"
    else:
        index.name = "Total Increase/Decrease"

    return df_correlation.T

def append_heuristic_info(df_correlation, df_combined):

    df_correlation['mean_increase'] = [0] + list(df_correlation.iloc[1:,:].mean(axis=1).values)
    # first row is 'label' which is the baseline with no heuristics
    df_correlation['num_heuristics'] = [0] + [len(eval(x)) for x in df_correlation.index.to_list()[1:]]

    # get the combination of bad labelers for the heuristics to try them out
    with open(ba_path, 'rb') as f:
        combined_ba = pickle.load(f)

    num_ba = []
    rows_filtered = []
    for heur in [eval(x) for x in df_correlation.index.to_list()[1:]]:
        ba = []
        for h in heur:
            ba += combined_ba[h]
        ba = list(set(ba))
        num_ba.append(len(ba))
        rows_filtered.append(np.round((df_combined[df_combined.annotator.isin(ba)].shape[0] / df_combined.shape[0])*100,2))
    df_correlation['num_ba'] = [0] + num_ba
    df_correlation['percent_rows_filtered'] = [0] + rows_filtered

    return df_correlation

def get_heuristic_scores(df,heuristic):

    with open(ba_path, 'rb') as f:
        combined_ba = pickle.load(f)
    ba = []
    for heur in heuristic:
        ba += combined_ba[heur]
    ba = list(set(ba))

    df_ba = df[~df.annotator.isin(ba)]
    mean_label = df_ba.groupby("pair_id")['label'].mean()
    metric_columns = df_ba.groupby("pair_id")[METRICS].mean()
    metric_columns["label"] = mean_label
    metrics = metric_columns.corr().abs().iloc[-1].sort_values(ascending=False)[1:]
    return pd.DataFrame(metrics)

if __name__ == '__main__':
    df_combined = pd.read_csv(DATA_PATH / 'combined_dataset.csv', index_col=0)

    with open(ba_path, 'rb') as f:
        combined_ba = pickle.load(f)
    avg_corr = {}
    for name, group in df_combined.groupby('dataset'):
        corr = get_filtering_correlation(group,combined_ba, percentage_increase=True)
        corr = append_heuristic_info(corr,group)
        top_corr = corr['mean_increase'].sort_values(ascending=False)[0]
        avg_corr[name] = top_corr



