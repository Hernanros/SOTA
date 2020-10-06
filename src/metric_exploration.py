import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import tqdm
from scipy.stats import pearsonr as pcorr
import itertools


def get_corr(df: pd.DataFrame, bad_annotator: list) -> dict:
    """
    Get the correlation between the various metrics and the human labeling filtering out particular "bad annotators"

    parameters:
        df -- {pd.DataFrame} -- combined dataset
        bad_annotator -- {list} -- list of all the annotator ID's we want to filter out

    return:
        {pd.DataFrame} - correlations by each dataset of metric and human label
        {pd.DataFrame} - correlations by each dataset of metric and reduced human label (-1,0,1)
        {pd.Series} - correlations of all datasets of metric and human label
        {pd.Series} - correlations of all datasets of metris and reduced human label
    """

    non_metric_columns = ['text1', 'text2', 'label', 'dataset', 'random', 'duration', 'total_seconds', 'pair_id',
                          'reduced_label', 'annotator']

    if bad_annotator:
        df = df[~df.annotator.isin(bad_annotator)]
        # Remove all pairs if there is only one annotator
        df = df.groupby('pair_id').filter(lambda x: x.annotator.count() >= 2)

    metrics = [x for x in df.columns if x not in non_metric_columns]
    all_labels = metrics + ['label'] + ['reduced_label']
    df = df.groupby(['pair_id', 'dataset', 'random'])[all_labels].mean().reset_index()

    label_corr = dict()
    reduced_label_corr = dict()

    # Iterate through the datasets and get the correlation of each metric with label & reduced label (separately)
    for name, group in df.groupby('dataset'):
        label_corr[name] = group[metrics].corrwith(group['label'])
        reduced_label_corr[name] = group[metrics].corrwith(group['reduced_label'])

    combined_datasets_label_corr = df[metrics].corrwith(df['label'])
    combined_datasets_reduced_label_corr = df[metrics].corrwith(df['reduced_label'])

    random_label_corr = dict()
    random_reduced_label_corr = dict()

    for name, group in df.groupby('random'):
        random_label_corr[name] = group[metrics].corrwith(group['label'])
        random_reduced_label_corr[name] = group[metrics].corrwith(group['reduced_label'])

    correlations_dict = dict()
    correlations_dict['label_by_dataset'] = pd.DataFrame.from_dict(label_corr).T
    correlations_dict['reduced_label_by_dataset'] = pd.DataFrame.from_dict(reduced_label_corr).T
    correlations_dict['label_by_random'] = pd.DataFrame.from_dict(random_label_corr).T
    correlations_dict['reduced_label_by_random'] = pd.DataFrame.from_dict(random_reduced_label_corr).T
    correlations_dict['label_by_combined'] = pd.Series(combined_datasets_label_corr)
    correlations_dict['reduced_label_by_combined'] = pd.Series(combined_datasets_reduced_label_corr)
    return correlations_dict


def compare_correlations(dict_baseline, dict_filtered):
    """
    Compares the correlations between the baseline dataframe and the filtered dataframe based off removing bad annotators

    parameters:
        dict_baseline -- {dict} -- dictionary of baseline df correlation scores
        dict_filtered -- {dict} -- dictionary of filtered df correlation scores

    returns:
        ab_dict -- {dict} -- dictionary of the filtered scores minus the baseline scores
    """
    ab_dict = dict()

    for key in dict_baseline.keys():
        ab_dict[key] = dict_filtered[key] - dict_baseline[key]

    return ab_dict
