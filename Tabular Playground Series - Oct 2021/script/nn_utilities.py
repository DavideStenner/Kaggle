import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score

from utilities import (
    RANDOM_STATE, TARGET_COL, N_FOLD, FOLD_STRAT_NAME,
    PARAMS_LGB_BASE
)

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
   
    

#########à
##########
    
    
class TabularDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx], dtype=torch.float)
        }
        return dct
    
class InferenceDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
        }
        return dct
    
    
    

#########################################################################################################
#########################################################################################################
class Model_ff(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(Model_ff, self).__init__()
        
        self.middle_size = int(hidden_size/2)
        
        self.layer_1 = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.1),
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
        )
        
        self.layer_2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU()
        )

        self.layer_3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU()
        )
        
        self.layer_4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.middle_size),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.middle_size),
            nn.Dropout(0.1),
            nn.Linear(self.middle_size, 1)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.classifier(x)
        
        return x
    
    
class Model_list(nn.Module):
    def __init__(self, output_dims: List[int], dropout_list: List[float], num_features):
        super().__init__()
        
        layers: List[nn.Module] = []

        input_dim: int = num_features
        for i, output_dim in enumerate(output_dims):
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_list[i]))
            input_dim = output_dim
        
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, 1))
    
        self.layers: nn.Module = nn.Sequential(*layers)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return logits
    
#############################
def train_fn(model, optimizer, scheduler, criterion, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device).unsqueeze(1)

        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
            
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, criterion, dataloader, device):
    model.eval()
    
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device).unsqueeze(1)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        final_loss += loss.item()
        
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds




#########################à

def run_training(train, valid, fold, model_nn, num_feature, hidden, batch_size, epochs, 
                 seed, early_stop_step, early_stop, device, learning_rate, weight_decay, verbose= True, save = True):

    assert isinstance(train, list) & isinstance(valid, list) & (len(train) == 2) & (len(valid) == 2)
    
    seed_everything(seed)
            
    x_train, y_train  = train
    x_valid, y_valid =  valid
    
    train_dataset = TabularDataset(x_train, y_train)
    valid_dataset = TabularDataset(x_valid, y_valid)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = model_nn(
        num_features=num_feature,
        hidden_size=hidden,
    )
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate*10, epochs=epochs, steps_per_epoch=len(trainloader))
    
    criterion = nn.BCEWithLogitsLoss()
    
    early_step = 0
    
    best_loss = np.inf
    best_auc = -np.inf
    
    for epoch in range(epochs):
        
        train_loss = train_fn(model, optimizer, scheduler, criterion, trainloader, device)
        valid_loss, valid_preds = valid_fn(model, criterion, validloader, device)

        valid_auc = roc_auc_score(y_valid, valid_preds)
        
        if verbose:
            print(f"EPOCH: {epoch},  train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid_auc: {valid_auc:.5f}")
        
        if valid_auc > best_auc:
            
            best_auc = valid_auc
            best_pred = valid_preds
            
            if save:
                torch.save(model.state_dict(), f"FOLD_{fold}_.pth")
            
        elif(early_stop == True):
            
            early_step += 1
            if (early_step >= early_stop_step):
                break
                
    return best_auc, best_pred

#########################################################################################################

#########################################################################################################
#########################################################################################################
class Model_mlp_ae(nn.Module):
    """
    https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp?scriptVersionId=73762661
    """
    
    def __init__(self, num_features, hidden_size):
        super(Model_mlp_ae, self).__init__()
        
        self.num_features = num_features
        ae_num_features = int(num_features * 0.75)
        concat_num_features = ae_num_features + num_features
        half_size = int(hidden_size/2)
        
        self.scaled_input = nn.Sequential(nn.BatchNorm1d(num_features))
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, ae_num_features),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(ae_num_features, num_features),
        )
        
        self.output_ae = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.1),
            nn.Linear(num_features, ae_num_features),
            nn.BatchNorm1d(ae_num_features),
            nn.Dropout(0.1),
            nn.Linear(ae_num_features, 1),
        )

        self.concat_ae_input = nn.Sequential(
            nn.BatchNorm1d(concat_num_features),
            nn.Dropout(0.2),
            nn.Linear(concat_num_features, hidden_size * 2),
            nn.GELU()
        )
        
        self.layer_1 = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
        )

        self.layer_2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, half_size),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(half_size),
            nn.Dropout(0.1),
            nn.Linear(half_size, 1)
        )

    def forward(self, x):
        scaled_input = self.scaled_input(x)
        
        #gaussian noise
        gauss_noise = torch.empty(self.num_features).normal_(mean=0,std=0.05).to(scaled_input.device)

        encoder = scaled_input + gauss_noise
        encoder = self.encoder(encoder)
        
        decoder = self.decoder(encoder)
        output_sigmoid_ae = self.output_ae(decoder)
        
        concat = torch.cat((scaled_input, encoder), dim = -1)
        concat_layer = self.concat_ae_input(concat)
        
        layer_1 = self.layer_1(concat_layer)
        layer_2 = self.layer_2(layer_1)
        
        class_output = self.classifier(layer_2)
        
        return decoder, output_sigmoid_ae, class_output

#########################################################################################################
#########################################################################################################

def run_training_ae(train, valid, fold, model_nn, num_feature, hidden, batch_size, epochs, 
                 seed, early_stop_step, early_stop, device, learning_rate, weight_decay, verbose= True, save = True):

    assert isinstance(train, list) & isinstance(valid, list) & (len(train) == 2) & (len(valid) == 2)
    
    seed_everything(seed)
            
    x_train, y_train  = train
    x_valid, y_valid =  valid
    
    train_dataset = TabularDataset(x_train, y_train)
    valid_dataset = TabularDataset(x_valid, y_valid)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = model_nn(
        num_features=num_feature,
        hidden_size=hidden,
    )
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate*10, epochs=epochs, steps_per_epoch=len(trainloader))
    
    criterion = {
        'bce': nn.BCEWithLogitsLoss(),
        'mse': nn.MSELoss()
    }
    
    early_step = 0
    
    best_loss = np.inf
    best_auc = -np.inf
    
    for epoch in range(epochs):
        
        train_loss = train_fn_ae(model, optimizer, scheduler, criterion, trainloader, device)
        valid_loss, valid_preds = valid_fn_ae(model, criterion, validloader, device)

        valid_auc = roc_auc_score(y_valid, valid_preds)
        
        if verbose:
            print(f"EPOCH: {epoch},  train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid_auc: {valid_auc:.5f}")
        
        if valid_auc > best_auc:
            
            best_auc = valid_auc
            best_pred = valid_preds
            
            if save:
                torch.save(model.state_dict(), f"FOLD_{fold}_.pth")       
            
        elif(early_stop == True):
            
            early_step += 1
            if (early_step >= early_stop_step):
                break
                
    return best_auc, best_pred

#########################################################################################################
#########################################################################################################
def train_fn_ae(model, optimizer, scheduler, criterion, dataloader, device):
    
    bce_criterion = criterion['bce']
    mse_criterion = criterion['mse']
    
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device).unsqueeze(1)
        
        decoder, output_sigmoid_ae, class_output = model(inputs)        
        
        mse_loss = mse_criterion(decoder, inputs)
        bce_ae_loss = bce_criterion(output_sigmoid_ae, targets)
        bce_loss = bce_criterion(class_output, targets)
        
        loss = mse_loss + bce_ae_loss + bce_loss
        loss.backward()
            
        optimizer.step()
        scheduler.step()
        
        final_loss += bce_loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn_ae(model, criterion, dataloader, device):
    bce_criterion = criterion['bce']

    model.eval()
    
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device).unsqueeze(1)
        
        _, _, class_output = model(inputs)
        loss = bce_criterion(class_output, targets)

        final_loss += loss.item()
        
        valid_preds.append(class_output.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn_ae(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            _, _, class_output = model(inputs)
        
        preds.append(class_output.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds

#########################################################################################################
#########################################################################################################



class Model_list(nn.Module):
    def __init__(self, output_dims: List[int], dropout_list: List[float], num_features):
        super().__init__()
        
        layers: List[nn.Module] = []

        input_dim: int = num_features
        for i, output_dim in enumerate(output_dims):
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_list[i]))
            input_dim = output_dim
        
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, 1))
    
        self.layers: nn.Module = nn.Sequential(*layers)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return logits
    
def run_training_model_fix(train, valid, fold, model, batch_size, epochs, 
                 seed, early_stop_step, early_stop, device, learning_rate, weight_decay, verbose = True, save = True):

    assert isinstance(train, list) & isinstance(valid, list) & (len(train) == 2) & (len(valid) == 2)
    
    seed_everything(seed)
            
    x_train, y_train  = train
    x_valid, y_valid =  valid
    
    train_dataset = TabularDataset(x_train, y_train)
    valid_dataset = TabularDataset(x_valid, y_valid)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate*10, epochs=epochs, steps_per_epoch=len(trainloader))
    
    criterion = nn.BCEWithLogitsLoss()
    
    early_step = 0
    
    best_loss = np.inf
    best_auc = -np.inf
    
    for epoch in range(epochs):
        
        train_loss = train_fn(model, optimizer, scheduler, criterion, trainloader, device)
        valid_loss, valid_preds = valid_fn(model, criterion, validloader, device)

        valid_auc = roc_auc_score(y_valid, valid_preds)
        
        if verbose:
            print(f"EPOCH: {epoch},  train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid_auc: {valid_auc:.5f}")
       
        if valid_auc > best_auc:
            
            best_auc = valid_auc
            best_pred = valid_preds
            
            if save:
                torch.save(model.state_dict(), f"FOLD_{fold}_.pth")

        elif(early_stop == True):
            
            early_step += 1
            if (early_step >= early_stop_step):
                break
        
    return best_auc, best_pred


