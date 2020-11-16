import pandas as pd
import numpy as np
from pathlib import Path
import configparser

FILEPATH = Path(Path.cwd().parents[1].resolve()) / 'data'
CRED_PATH = Path(__file__).resolve().parents[1] / 'credentials.ini'


def get_environment_variables():
    config_parser = configparser.ConfigParser()
    config_parser.read(CRED_PATH)
    PATH_ROOT = config_parser['MAIN']["PATH_ROOT"]
    PATH_DATA = config_parser['MAIN']["PATH_DATA"]
    GloVe_840B_300d_PATH = config_parser['MAIN']["GloVe_840B_300d_PATH"]
    Glove_twitter_27B_PATH = config_parser['MAIN']["Glove_twitter_27B_PATH"]
    ENV = config_parser['MAIN']["ENV"]
    return PATH_ROOT, PATH_DATA, GloVe_840B_300d_PATH, Glove_twitter_27B_PATH, ENV


def convert_tsv_to_csv_sts(path=FILEPATH):
    """
    Load STS benchmark data. - from the Github sourcecode
    """
    genres, sent1, sent2, labels, scores = [], [], [], [], []
    for line in open(path):
        genre = line.split('\t')[0].strip()
        filename = line.split('\t')[1].strip()
        year = line.split('\t')[2].strip()
        other = line.split('\t')[3].strip()
        score = line.split('\t')[4].strip()
        s1 = line.split('\t')[5].strip()
        s2 = line.split('\t')[6].strip()
        label = float(score)
        genres.append(genre)
        sent1.append(s1)
        sent2.append(s2)
        labels.append(label)
        scores.append(score)
    labels = (np.asarray(labels)).flatten()

    # include which category (train/dev/test) as column for future reference
    dataset_categ = [os.path.split(path)[1].replace(".csv", "")] * len(labels)
    dataset = ['sts'] * len(labels)

    return pd.DataFrame({'genres': genres, 'text_1': sent1, 'text_2': sent2,
                         'label': labels, 'scores': scores, 'dataset': dataset, 'dataset-categ': dataset_categ})

# def combine_datasets(path = FILEPATH):
#     for i, file in enumerate(glob.glob(FILEPATH / raw_data /*.csv')):
#         if i==0:
#             df = pd.read_csv(file, index_col = 0)
#             df['dataset'] = file[5:-4].lower()
#         else:
#             temp = pd.read_csv(file, index_col = 0)
#             temp['dataset'] = file[5:-4].lower()
#             df = pd.concat((df,temp),axis = 0)
#     df['random'] = df.dataset.apply(lambda x: 'random' in  x).astype(int)
#     df['duration'] = pd.to_datetigit dime(df.SubmitTime)-pd.to_datetime(df.AcceptTime)
#
#     df = df.drop(['HITId','HITTypeId', 'Title', 'Description', 'Keywords',
#         'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AssignmentId','AssignmentStatus','AcceptTime', 'SubmitTime','AutoApprovalTime', 'ApprovalTime', 'RejectionTime','RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate','Last30DaysApprovalRate', 'Last7DaysApprovalRate','Approve', 'Reject'],axis = 1)
#
#     df=df.rename(columns = {'Input.text1':'text_1','Input.text2':'text_2','Answer.semantic-similarity.label':'label'})
#
#
#     df.label = df.label.apply(lambda x:x[0]).astype(int)
#     df = df.reset_index()
#     df.drop(columns="index",inplace=True)
#     df['total_seconds'] = [int(df['duration'][i].total_seconds()) for i in range(df.shape[0])]
#
#     #As there is no id for same pair documents - I will create it
#     df["pair_id"] = [f"pair_{i//3}" for i in range(df.shape[0])]
#
#     #We will first replace 1-2 with [-1] and 4-5 with [1]
#     df['reduced_label'] = [1 if x > 3 else -1 if x < 3 else 0 for x in df.label]
#
#
# if __name__== '__main__':
#     # convert tsv to csv
#     dfs = []
#     for files in glob.glob(FILEPATH / 'datasets' / "*.tsv" ):
#         df = convert_tsv_to_csv_sts(files)
#         df.to_csv(FILEPATH / f'{files}.csv')
#
#     combined = combine_datasets()
#     combined.to_csv(FILEPATH / 'combined.csv')
