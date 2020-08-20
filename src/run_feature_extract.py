import pickle
import argparse
import bleu
import cosine_similarites


def main(args):
    print(args.pickle)

    picklefile = 'data\combined_data_test.pickle'
    extractors = dict(bleu=bleu.Bleu(), cosine_similarites=cosine_similarites.CosineSimilarity())
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

    args = parser.parse_args()
    main(args)