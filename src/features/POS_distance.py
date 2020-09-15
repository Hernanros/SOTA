"""implements POS distance metric as a class"""
import pandas
from nltk import pos_tag, word_tokenize
import nltk
import torchtext.vocab as torch_vocab
import torch

class POSDistance:

    def __init__(self):
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
        self.dic_glove = torch_vocab.GloVe(name='twitter.27B', dim=100)

    def run(self, df: pandas.DataFrame) -> pandas.DataFrame:

        loss_nn_list = []
        total_loss_nn = 0
        count = 0
        i = 0
        while i < len(df):
            sentence_ori = df.iloc[i]['text_1'].copy()
            sentence_gen = df.iloc[i]['text_2'].copy()
            temp_res_ori = pos_tag(word_tokenize(sentence_ori))
            temp_res_gen = pos_tag(word_tokenize(sentence_gen))
            temp_nn_ori = []
            temp_nn_gen = []
            temp_nn_vector_ori = []
            temp_nn_vector_gen = []
            for tube in temp_res_ori:
                if tube[1] == 'NN' or tube[1] == 'NNS':
                    temp_nn_ori.append(tube[0])
            for tube in temp_res_gen:
                if tube[1] == 'NN' or tube[1] == 'NNS':
                    temp_nn_gen.append(tube[0])
            for word in temp_nn_ori:
                try:
                    temp_nn_vector_ori.append(self.dic_glove.vectors[self.dic_glove.stoi[word]])
                except KeyError:
                    a = 1
            for word in temp_nn_gen:
                try:
                    temp_nn_vector_gen.append(self.dic_glove.vectors[self.dic_glove.stoi[word]])
                except KeyError:
                    a = 1
            if temp_nn_vector_ori != [] and temp_nn_vector_gen != []:
                loss_list = []
                for vector_target in temp_nn_vector_ori:
                    for vector_gen in temp_nn_vector_gen:
                        tensor_gen = torch.FloatTensor(vector_gen)
                        tensor_target = torch.FloatTensor(vector_target)
                        temp_loss = torch.dist(tensor_gen, tensor_target)
                        loss_list.append(temp_loss)
                loss_list_new = sorted(loss_list)
                loss_list_new1 = loss_list_new[:min(len(temp_nn_vector_ori), len(temp_nn_vector_gen))]
                loss = (sum(loss_list_new1) / len(loss_list_new1)) * (
                            1 + abs(len(temp_nn_vector_ori) - len(temp_nn_vector_gen)) / len(temp_nn_vector_ori))
                total_loss_nn += loss
                loss_nn_list.append(loss.numpy().item())
                count += 1
            else:
                loss_nn_list.append(-1)
            i += 1

        df['POS dist score'] = loss_nn_list
        return df

