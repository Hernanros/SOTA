import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_train_test_data(train_path: str, all_metrics: list, test_path: str = None) -> tuple:
    '''
    Retrieve the data from the paths and split into train/test based off if they are from the same dataset or otherwise.

    Parameters:
        train_path -- {str} -- Path to the training dataset
        metrics -- {list} -- name of the metrics we want to use as features
        test_path -- {str} -- Path to the test dataset (by default is None, and then is the same as Train_path)

    Returns:
        {tuple} -- (X_train, X_test, y_train, y_test)
    '''

    if test_path is None:
        df = pd.read_csv(train_path, index_col=0)
        
        #If we are dealing with the sts dataset, where it has within it a pre-defined train/val/test
        if Path(train_path).stem == 'sts':
            train_data = df[df['dataset-categ'] == 'sts-train']
            #to include both 'sts-dev' and 'sts-test'
            test_data  = df[df['dataset-categ'] != 'sts-train']
        else:
            #shuffle the dataframe
            len_df = int(df.shape[0] * 0.8)

            df = df.sample(frac=1)
            train_data= df.iloc[:len_df]
            test_data = df.iloc[len_df:]
    else:
        train_data = pd.read_csv(train_path, index_col=0)
        test_data = pd.read_csv(test_path, index_col=0)

    #To test it on 
    metrics = [x for x in test_data.columns if x in all_metrics]
    if len(metrics) != len(all_metrics):
        print(f"Still missing the following metrics: {set(all_metrics).difference(set(metrics))}")

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    print(f"Size of train_data: {train_data.shape[0]}\tSize of test_data: {test_data.shape[0]}")
    return (train_data[metrics], test_data[metrics], train_data['label'], test_data['label'])

####### RF #######

def RF_corr(X_train,X_test,y_train,y_test, max_depth = 3):
    '''
    Random Forest Regression.

    Parameters:
        max_depth -- {int} -- depth of the Random Forest Regressor
        X_train -- {pd.DataFrame} -- Train data
        y_train -- {pd.Series} -- Train labels

    Return:
        y_pred -- {list} -- Test predicted labels
        model -- {model} -- The RF Model

    '''
    model = RandomForestRegressor(max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return pearsonr(y_pred,y_test)[0]

##################

###  MLP MODEL ###

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

def MLP_corr(X_train,X_test,y_train,y_test, num_hl = 128):
    model = Basemodel(X_train.shape[1],num_hl,1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = torch.Tensor(X_test.to_numpy()).to(dtype=torch.float32)

    train_set = DS(X_train,y_train)
    # test_set = DS(X_test,y_test)
    train_loader=DataLoader(dataset= train_set, batch_size = 32, shuffle = True, num_workers = 2)
    # test_loader=DataLoader(dataset= test_set, batch_size = 32, shuffle = True, num_workers = 2)

    model = train_epoch(train_loader,model,criterion,optimizer,num_epochs= 30)
    
    if torch.cuda.is_available:
        y_pred = model(X_test).cpu().detach().numpy().flatten()
    else:
        y_pred = model(X_test).detach().numpy().flatten()

    return pearsonr(list(y_pred),list(y_test))[0]

##################