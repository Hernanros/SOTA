from nltk import pos_tag
import torchtext.vocab as torch_vocab
import csv
import numpy as np
import argparse
import torch
parser = argparse.ArgumentParser()


DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence = row['text1'].strip()
        second_sentence = row['text2'].strip()
        pairs.append((first_sentence, second_sentence))


dic_glove = torch_vocab.GloVe(name='twitter.27B', dim=100)
loss_nn = []
for first_sentence, second_sentence in pairs:
    sentence_ori = first_sentence
    sentence_gen = second_sentence
    temp_res_ori = pos_tag(sentence_ori.split())
    temp_res_gen = pos_tag(sentence_gen.split())
    temp_nn_ori = []
    temp_nn_gen = []
    temp_nn_vector_ori = []
    temp_nn_vector_gen = []
    for tube in temp_res_ori:
        if tube[1] == 'NN':
            temp_nn_ori.append(tube[0])
    for tube in temp_res_gen:
        if tube[1] == 'NN':
            temp_nn_gen.append(tube[0])    
    for word in temp_nn_ori:
        try:
            temp_nn_vector_ori.append(dic_glove.vectors[dic_glove.stoi[word]])
        except KeyError:
            a = 1 
    for word in temp_nn_gen:
        try:
            temp_nn_vector_gen.append(dic_glove.vectors[dic_glove.stoi[word]])
        except KeyError:
            a = 1         
    if temp_nn_vector_ori != [] and temp_nn_vector_gen != []:
        loss_list = []
        for vector_target in temp_nn_vector_ori:
            for vector_gen in temp_nn_vector_gen:
                tensor_gen = torch.FloatTensor(vector_gen)
                tensor_target = torch.FloatTensor(vector_target)
                temp_loss = torch.dist(tensor_gen,tensor_target)
                loss_list.append(temp_loss)
        loss_list_new = sorted(loss_list)
        loss_list_new1 = loss_list_new[:min(len(temp_nn_vector_ori),len(temp_nn_vector_gen))]
        loss_nn.append((sum(loss_list_new1)/len(loss_list_new1))*(1+abs(len(temp_nn_vector_ori)-len(temp_nn_vector_gen))/len(temp_nn_vector_ori)))

print(np.mean(loss_nn))
print(np.std(loss_nn))