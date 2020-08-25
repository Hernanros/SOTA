"""
This file implements running the different semantic similiarity metrics on a dataset of paired sentences
"""

import pickle
import argparse
from .bleu import Bleu
from .cosine_similarites import CosineSimilarity
from .elmo_euclidean_distance import EuclideanElmoDistance
from .bertscore import BertScore
from .chrFScore import chrFScore
from .WMD import WMD
from .POS_distance import POSDistance
from .ngram_overlap import NgramOverlap


def main(args):
    print(args.pickle)

    picklefile = 'data\combined_data_test.pickle'
    extractors = dict(bleu=Bleu(), cosine_similarites=CosineSimilarity(), elmo_similarites=EuclideanElmoDistance(),
                      bert=BertScore(), chrf_score=chrFScore(), pos_distance=POSDistance(), wmd=WMD(),
                      ngram_overlap=NgramOverlap(args.max_n))
    features = args.features
    if features == 'ALL':
        features = list(extractors.keys())
    else:
        features = features.lower().split(',')

    with open(picklefile, 'rb') as handle:
        df = pickle.load(handle)

    for feature_name, extractor in extractors.items():
        if feature_name in features:
            df = extractor.run(df)

    with open(picklefile, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, help='pickle path for combined dataset')
    parser.add_argument('--features', type=str, help='use "ALL" for all features, or comma separated list of features')
    parser.add_argument('--max_n', type=int, help='maximum number of n-gram overlap score to calculate, e.g. max_n=2 creates 1-gram-overlap & 2-gram-overlap')

    args = parser.parse_args()
    main(args)
