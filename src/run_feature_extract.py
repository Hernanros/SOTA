
import pickle
import argparse
import bleu

def main(args):
    print(args.pickle)

    picklefile = 'data\combined_data_test.pickle'
    features = ['bleu']

    with open(picklefile, 'rb') as handle:
        df = pickle.load(handle)    


    for feature in features:
        if feature is 'bleu':
            extractor = bleu.Bleu()
            df = extractor.run(df)
        elif feature is 'something1':
            pass        
        elif feature is 'something2':
            pass        
        elif feature is 'something3':
            pass        

    with open(picklefile, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, help='pickle path for combined dataset')
    parser.add_argument('--features', type=str, help='use "ALL" for all features, or comma separated list of features')

    args = parser.parse_args()
    main(args)