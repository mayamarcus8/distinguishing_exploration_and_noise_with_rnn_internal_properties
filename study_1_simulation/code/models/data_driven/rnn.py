import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 4
OUTPUT_SIZE = 3

class behavior_dataset(Dataset):
    def __init__(self,data):
        
        length = len(data)
        num_actions = 3

        # action 
        action = np.array(data['card_selected']) - 1
        unique_actions = np.unique(action)

        if len(unique_actions) < num_actions:
            missing_actions = list(set(range(num_actions)) - set(unique_actions))
            action = np.append(action, missing_actions)
            action = torch.tensor(action.reshape(length + len(missing_actions)), dtype=int)
            # one hot encoding
            action_onehot = nn.functional.one_hot(action, num_classes=num_actions)
            action_onehot = action_onehot[:-len(missing_actions)]

        else:
            # action 
            action = torch.tensor((action).reshape(length),dtype=int)
            # one hot encoding 
            action_onehot = nn.functional.one_hot(action,len(action.unique()))
        
        # reward
        reward = torch.tensor((np.array(data['reward'])).reshape(length),dtype=int)
        
        # concatinating reward and action
        reward_action = torch.cat([reward[ :, np.newaxis], action_onehot],1)
        
        # adding dummy zeros to the beginning and ignoring the last one
        # [r (t-1) , a (t-1)]
        reward_action_shift = nn.functional.pad(reward_action,[0,0,1,0])[:-1]
            
        X = reward_action_shift
        y = action_onehot

        self.x = X.type(dtype=torch.float32)
        self.y = y.type(dtype=torch.float32)
        
        self.len = length

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len    

class GRU_NN(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.hidden_size = hidden_size
        
        super(GRU_NN, self).__init__()
        self.hidden = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, hn = self.hidden(x)
        output = self.out(output)
        output = F.softmax(output,dim=-1)
        return output, hn

def train_model(net, train_loader, val_loader , test_loader, epochs, lr):
        
    train_loss, train_ll = np.zeros(epochs), np.zeros(epochs)
    val_loss ,val_ll = np.zeros(epochs), np.zeros(epochs)
    test_loss, test_ll = np.zeros(epochs), np.zeros(epochs)

    # move net to GPU
    net.to(device)
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Loop over epochs
    for i in range(epochs):
        
        # Loop over training batches
        running_loss_tr = []
        for j,(X_train,y_train) in enumerate(train_loader):
            
            # move to GPU
            X_train , y_train = X_train.to(device), y_train.to(device)
            # reshape to 1 X batch_size X input_size
            X_train = X_train.reshape(1,X_train.shape[0], INPUT_SIZE)
            # zero the gradient buffers
            optimizer.zero_grad() 
            out, hn = net(X_train)
            # Reshape to (SeqLen x Batch, OutputSize)
            out = out.view(-1, OUTPUT_SIZE)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step() # Does the update
            running_loss_tr.append(loss.item())
            
        train_loss[i], train_ll[i] = eval_net(net, train_loader)
        val_loss[i], val_ll[i] = eval_net(net, val_loader)
        test_loss[i], test_ll[i] = eval_net(net, test_loader)
        
        net.train()

    return net, train_loss , train_ll , val_loss, val_ll, test_loss, test_ll

def eval_net(net,val_loader):
    criterion = nn.BCELoss()
    running_loss_te = []
    with torch.no_grad():
        net.eval()
        for j, (X, y_true) in enumerate(val_loader):
            
            # move to GPU
            X, y_true = X.to(device), y_true.to(device) # move to GPU
            # reshape to 1 X batch_size X input_size
            X = X.reshape(1,X.shape[0],INPUT_SIZE)
            out, hn = net(X)
            # Reshape to (SeqLen x Batch, OutputSize)
            out = out.view(-1, OUTPUT_SIZE)
            loss = criterion(out, y_true)
            running_loss_te.append(loss.item())
            
            y_true = torch.argmax(y_true,1)
            ll = out.gather(1, y_true.view(-1,1))
            ll = -torch.sum(torch.log(ll))
            ll = float(ll.to('cpu').detach())
            
    return np.array(running_loss_te).mean(), ll
        