import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import tqdm
from scipy.stats import pearsonr as pcorr
import itertools
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# non_metric_columns = ['text1','text2','label','dataset','random','duration','total_seconds','pair_id','reduced_label','annotator','radical','radical_random','radical_non_random','is_radical','is_centralist','num_labels','bad_annotator']

class Metrics_Corr():
    '''
    In order to make a generalizable correlation for future uses in differing circumstances.

    Parameters:
        df -- {pd.DataFrame} -- our combined metrics with all metrics and distinctions
        non_metrics_columns -- {list} -- all of the columns that arent metrics
        categories -- {list} -- list for each type of categories we want to filter by (by default ['dataset','random])
    '''

    def __init__(self,df, non_metric_columns, categories = ['dataset','random']):
        self.df = df
        self.non_metric_columns = non_metric_columns
        self.categories = categories

    def get_corr(self, bad_annotator: list) -> dict:
        '''
        Get the correlation between the various metrics and the human labeling filtering out particular "bad annotators"

        parameters:
            bad_annotator -- {list} -- list of all the annotator ID's we want to filter out

        return:
            correlations_dict -- {dict} -- correlations for each of the categories and the combined dataset for the label and the reduced label
        '''

        df = self.df.copy()

        if bad_annotator:
            df= df[~df.annotator.isin(bad_annotator)]
            #Remove all pairs if there is only one annotator
            df = df.groupby('pair_id').filter(lambda x: x.annotator.count() >= 2)

        metrics = [x for x in df.columns if x not in self.non_metric_columns]
        all_labels = metrics + ['label'] + ['reduced_label']
        df = df.groupby(['pair_id'] + self.categories)[all_labels].mean().reset_index()

        correlations_dict = dict()

        #Iterate through the various categories and get the correlation of each metric with label & reduced label (separately)
        for category in self.categories:
            label_corr = dict()
            reduced_label_corr = dict()
            for name,group in df.groupby(category):
                label_corr[name] = group[metrics].corrwith(group['label'])
                reduced_label_corr[name] = group[metrics].corrwith(group['reduced_label'])
            correlations_dict['label_by_' + category] = pd.DataFrame.from_dict(label_corr).T
            correlations_dict['reduced_label_by_' + category] = pd.DataFrame.from_dict(reduced_label_corr).T

        combined_datasets_label_corr = df[metrics].corrwith(df['label'])
        combined_datasets_reduced_label_corr = df[metrics].corrwith(df['reduced_label'])

        correlations_dict['label_by_combined'] = pd.Series(combined_datasets_label_corr)
        correlations_dict['reduced_label_by_combined'] = pd.Series(combined_datasets_reduced_label_corr)

        return correlations_dict


    def compare_correlations(self, bad_annotator : list) -> dict:
        ''' 
        Compares the correlations between the baseline dataframe and the filtered dataframe based off removing bad annotators

        parameters:
            bad_annotator -- {list} -- list of all the annotators you want for filtered dataframe

        returns:
            ab_dict -- {dict} -- dictionary of the filtered scores minus the baseline scores
        '''
        ab_dict = dict()

        dict_baseline = self.get_corr(None)
        dict_filtered = self.get_corr(bad_annotator)


        for key in dict_baseline.keys():
            ab_dict[key] = dict_filtered[key] - dict_baseline[key]

        return ab_dict






class Metrics_Models():
    '''
    In order to make a generalizable method to run the Linear and Non-Linear Model.

    Parameters:
        df -- {pd.DataFrame} -- our combined metrics with all metrics and distinctions
        non_metric_columns -- {list} -- all of the columns that arent metrics
        categories -- {list} -- list for each type of categories we want to filter by (by default ['dataset','random])
    '''

    def __init__(self,df,non_metric_columns, categories = ['dataset','random']):
        self.non_metric_columns = non_metric_columns
        self.categories = categories

        self.metrics = [x for x in df.columns if x not in self.non_metric_columns]
        all_labels = self.metrics + ['label'] + ['reduced_label']
        self.df = df.groupby(['pair_id'] + self.categories)[all_labels].mean().reset_index()

        #Nans dont work for linear and non-linear moels
        self.df.dropna(axis='index', inplace=True)


    def get_data_scaled(self, df):
        '''
        For use in Linear Models - to scale all of the data

        Parameters:
            df -- {pd.DataFrame} -- base dataframe want to run the model on

        Return:
            data -- {pd.DataFrame} -- just the metrics, scaled
        '''

        data = df.drop(['pair_id'] + self.categories + ['label','reduced_label'], axis=1).copy()

        column_names = list(data.columns) 

        x = data.values #returns a numpy array

        #scale the data values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled, columns=column_names)

        return data

    def RF(self,max_depth,X_train,y_train,X_test):
        '''
        Random Forest Regression.

        Parameters:
            max_depth -- {int} -- depth of the Random Forest Regressor
            X_train -- {pd.DataFrame} -- Train data
            y_train -- {pd.Series} -- Train labels
            X_test -- {pd.DataFrame} -- Test data

        Return:
            y_pred -- {list} -- Test predicted labels
            model -- {model} -- The RF Model

        '''
        model = RandomForestRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    def MLP(self,num_hl,X_train,y_train,X_test):
        '''
        Multi-Layered Perceptron.

        Parameters:
            num_hl -- {int} -- size of the hidden layer
            X_train -- {np.array} -- Train data
            y_train -- {np.array} - Train label
            X_test -- {np.array} - Test data
        
        Return:
            y_pred -- {np.array} - Test predicted labels
            model -- {model} -- The MLP Model
        '''
        model = Basemodel(len(self.metrics),num_hl,1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = torch.Tensor(X_test.to_numpy()).to(dtype=torch.float32)

        train_set = DS(X_train,y_train)
        train_loader=DataLoader(dataset= train_set, batch_size = 32, shuffle = True, num_workers = 2)

        model = train_epoch(train_loader,model,criterion,optimizer,num_epochs= 30)
        
        if torch.cuda.is_available:
            return model(X_test).cpu().detach().numpy(), model
        else:
            return model(X_test).detach().numpy(), model


    def run_model(self, model_type = "RF", max_depth = 3, num_hl = 128):
        '''
        Run either the RF or the MLP.

        Parameters:
            model_type -- {string} -- What type of model to use:
                                        - RF : Random Forest Regressor
                                        - MLP: Multi-Layered Perceptron (NN)
            max_depth -- {int} -- size of the Random Forest 
            num_hl -- {int} -- size of the Hidden layer in the MLP
        
        Return:
            scores -- {dict} -- the MSE of prediction with labels based off the categories
            If model_type == "RF":
                fi_values -- {dict} -- dictionary of the feature importance of the metrics based off categories
        '''

        if (model_type != "RF") & (model_type != "MLP"):
            print("Model type is wrong! Takes either `RF` or `MLP`")
            return None

        scores = dict()
        fi_values = dict()
        for category in self.categories:
            category_scores = dict()
            feature_importance = dict()
            for name, group in self.df.groupby(category):

                labels = group.label
                labels_reduced = group.reduced_label
                data = self.get_data_scaled(group)

                X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
                X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(data, labels_reduced, test_size=0.2)

                #Get the score from the models
                y_pred, model = eval("self."+model_type)(max_depth,X_train,y_train,X_test)
                category_scores[model_type + '_label_by_' + str(category)+ "_" + str(name)] = mean_squared_error(y_test, y_pred).T

                y_pred_reduced, model2 = eval("self."+model_type)(max_depth,X_train_reduced,y_train_reduced,X_test_reduced)
                category_scores[model_type + '_label_reduced_by_' + str(category)+ "_" + str(name)] = mean_squared_error(y_test_reduced, y_pred_reduced).T

                if model_type == "RF":
                    feature_importance['fi_label_by_' + str(category)+ "_" + str(name)]  = pd.DataFrame({'feature': data.columns.values, 'importance':model.feature_importances_}).sort_values('importance', ascending=False) 
                    feature_importance['fi_reduced_label_by_' + str(category)+ "_" + str(name)]  = pd.DataFrame({'feature': data.columns.values, 'importance':model2.feature_importances_}).sort_values('importance', ascending=False) 
            
            scores[category] = pd.DataFrame.from_dict(category_scores, orient='index').T
            
            if model_type == "RF":
                fi_values[category] = feature_importance

        #Get the scores for the whole dataset
        labels = self.df.label
        labels_reduced = self.df.reduced_label
        data = self.get_data_scaled(self.df)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(data, labels_reduced, test_size=0.2)

        #Get the score from the models
        y_pred, model = eval("self."+model_type)(max_depth,X_train,y_train,X_test)
        scores[model_type + '_label_combined'] = mean_squared_error(y_test, y_pred).T

        y_pred_reduced, model2 = eval("self."+model_type)(max_depth,X_train_reduced,y_train_reduced,X_test_reduced)
        scores[model_type + '_label_reduced_combined'] = mean_squared_error(y_test_reduced, y_pred_reduced).T

        if model_type == "RF":
            fi_values['fi_label_combined']  = pd.DataFrame({'feature': data.columns.values, 'importance':model.feature_importances_}).sort_values('importance', ascending=False) 
            fi_values['fi_reduced_label_combined']  = pd.DataFrame({'feature': data.columns.values, 'importance':model2.feature_importances_}).sort_values('importance', ascending=False) 


        if model_type == "RF":
            return scores, fi_values

        else:
            return scores









class DS(Dataset):
    '''
    Basic Dataset for the MLP.
    '''
    def __init__(self,df,labels):
        super(DS).__init__()
        self.df = df
        self.labels = labels

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        feat = self.df[idx,:]
        label = self.labels[idx]        

        return feat,label

class Basemodel(nn.Module):
  
  def __init__(self,n_feature,n_hidden,n_output, keep_probab = 0.1):
    '''
    input : tensor of dimensions (batch_size*n_feature)
    output: tensor of dimension (batchsize*1)
    '''
    super().__init__()
  
    self.input_dim = n_feature    
    self.hidden = nn.Linear(n_feature, n_hidden) 
    self.predict = nn.Linear(n_hidden, n_output)
    self.dropout = nn.Dropout(keep_probab)
    # self.pool = nn.MaxPool2d(2, 2)
    # self.norm = nn.BatchNorm2d(self.num_filters)


  def forward(self, x):
    x = self.dropout(F.relu(self.hidden(x)))
    x = self.predict(x)
    return x

def train_epoch(tr_loader,model,criterion,optimizer, num_epochs):

    if torch.cuda.is_available():
      device = torch.device('cuda:0')
      model.to(device)
    else:
      device = torch.device('cpu:0')

    for epoch in range(num_epochs):
    #   print("started training epoch no. {}".format(epoch+1))
      for step,batch in enumerate(tr_loader):
            feats,labels = batch
            feats = feats.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
      
    return model
