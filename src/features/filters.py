# TODO function to load dataset
# TODO implement following functions:
#  1. filter dataset from list of Annotator IDs,
#  2. filter text pairs with only 1 annotator,
#  3. create list of annotator IDs from dataset based on conditions -
#  less critical, already exists in notebooks/exploration/[SS]Removing Bad annotators

import pandas as pd


def filter_annotator_ids(df: pd.DataFrame, annotator_list: list) -> pd.DataFrame:
    return df.loc[df.annotator.isin(annotator_list)].copy()

def filter_single_annotator_examples(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby('pair_id').filter(lambda x: x['label'].count() > 1)
    return df.loc[df.pair_id.isin(groups.index)].copy()
