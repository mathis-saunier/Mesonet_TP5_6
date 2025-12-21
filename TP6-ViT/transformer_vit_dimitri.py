# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import matplotlib.pyplot as plt
import math
import tqdm
from tqdm import tqdm
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        # add the batch dimension
        pos_encoding = pos_encoding.unsqueeze(0) #.transpose(0, 1) 
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.size(1), :])

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim_model,dropout_p, max_len):
        super().__init__()
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        demi_dim = int(dim_model/2)
        # X DIMENSION
        x_pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        #                                         first half the dimension for the x frequencies
        division_term = torch.exp(torch.arange(0, demi_dim, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        x_pos_encoding[:, 0:demi_dim:2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        x_pos_encoding[:, 1:demi_dim:2] = torch.cos(positions_list * division_term)
        # extension dans la dimension y en repetant les valeurs selon x
        x_pos_encoding = x_pos_encoding.reshape((max_len,dim_model,1))     
        x_pos_encoding = x_pos_encoding.repeat(1,1,max_len)
        x_pos_encoding = torch.permute(x_pos_encoding,(0,2,1))

        # Y DIMENSION
        y_pos_encoding = torch.zeros(max_len, dim_model)
        #                                         second half the dimension for the y frequencies
        division_term = torch.exp(torch.arange(demi_dim,dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        y_pos_encoding[:,demi_dim:dim_model:2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        y_pos_encoding[:, demi_dim+1:dim_model:2] = torch.cos(positions_list * division_term)
        # extension dans la dimension x en repetant les valeurs
        y_pos_encoding = y_pos_encoding.repeat(1,max_len,1)
        y_pos_encoding=y_pos_encoding.reshape((max_len,max_len,dim_model))

        # create 2D positionnal encoding 
        # on suppose des images carrées où le positionnal encoding peut être pré-calculé et mis à plat
        # à l'avance, sinon il faudrait géréer les dimensions de chaque image dans le batch....
        pos_encoding2D =  torch.flatten(x_pos_encoding+y_pos_encoding,start_dim=0,end_dim=1)
        
        #print("pos_encoding2D.shape:",pos_encoding2D.shape)
        # Saving buffer (same as parameter without gradients needed)
        # add the batch dimension
        pos_encoding2D = pos_encoding2D.unsqueeze(0)

        self.register_buffer("pos_encoding2D",pos_encoding2D)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        # on suppose des images touets de même dimension, donc 
        return self.dropout(token_embedding + self.pos_encoding2D)
    
###########################################################
class CNN_ViT(nn.Module):
    def __init__(self, config,device):
        super(CNN_ViT, self).__init__()
        self.input_features = config['input_features']
        self.batch_size = config['batch_size']
        self.num_epochs =config['num_epochs']
        self.learning_rate =config['learning_rate']
        self.num_classes =config['num_classes']
        self.hidden_size =config['hidden_size']
        self.num_heads =config['num_heads']
        self.num_layers =config['num_layers']
        self.dropout =config['dropout']
        self.x_pad_idx =config['x_pad_idx']
        self.y_pad_idx =config['y_pad_idx']
        self.max_length =config['max_length']
        self.START_TOKEN =config['START_TOKEN']
        self.END_TOKEN =config['END_TOKEN']
        self.DEVICE = device

        self.positional_encoding_layer = PositionalEncoding(dim_model = self.hidden_size, 
                                                            dropout_p = self.dropout, 
                                                            max_len = self.max_length)
        
        self.positional_encoding_layer2D = PositionalEncoding2D(dim_model = self.hidden_size, 
                                                            dropout_p = self.dropout, 
                                                            max_len = int(math.sqrt(self.max_length)))
        
        # CNN Output is 16 channels
        self.x_embedding = nn.Linear(16, config['hidden_size'])      
                
        self.y_embedding = nn.Embedding(config['num_classes'],config['hidden_size'])         
        
        self.transformer = nn.Transformer(d_model=config['hidden_size'], 
                                                      nhead=config['num_heads'], 
                                                      num_encoder_layers =config['num_layers'],
                                                      #num_encoder_layers = 1,
                                                      num_decoder_layers =config['num_layers'],
                                                      dim_feedforward = config['hidden_size']*4,
                                                      dropout=config['dropout'],
                                                      batch_first=True,
                                                      norm_first=True)
        
        self.norm = nn.LayerNorm(config['hidden_size'])
        
        # Output layer for text prediction
        self.output_layer = nn.Linear(config['hidden_size'], config['num_classes'])

        self.cnn = nn.Sequential(    
            # entrée 120 X 120
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
                   
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Carte 15 X 15 X 256 features
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # sequence 225 X 16 features
            nn.Flatten(start_dim=2,end_dim=3),
            
        )
        
    def forward(self, x, y, y_output_mask, src_key_padding_mask, tgt_key_padding_mask):
        # x shape: (batch_size, 1, 120, 120)
        
        x = self.cnn(x) # (Batch, 16, 225)
        
        x = torch.permute(x,(0,2,1)) # (Batch, 225, 16)
        
        x = self.x_embedding(x) * math.sqrt(self.hidden_size)

        x = self.positional_encoding_layer2D(x) 
        
        y = self.y_embedding(y) * math.sqrt(self.hidden_size)
        y = self.positional_encoding_layer(y)

        transformer_out = self.transformer(x, y,
                                           tgt_mask=y_output_mask,
                                           src_key_padding_mask=None,
                                           tgt_key_padding_mask=tgt_key_padding_mask)
        
        transformer_out = self.norm(transformer_out)
        
        output = self.output_layer(transformer_out)
        return output
    
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
#########################################################
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    nb_batches = len(dataloader)
    epoch_loss = 0
    
    model.train()
    
    for batch, (X, y,X_l,y_l) in enumerate(dataloader):
        
        X = X.to(model.DEVICE)
        y = y.to(model.DEVICE)

        y_input = y[:,:-1] 
        y_output = y[:,1:] 
        
        l = y_input.size(1) # la longueur maximale d'un élément du batch
        y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)
        
        x_padding_mask = (X[:,:,0] == model.x_pad_idx).to(model.DEVICE)
        y_input_padding_mask = (y_input == model.y_pad_idx).to(model.DEVICE)
        
        # Compute prediction and loss
        pred = model(X.float(), y_input,
                     y_output_mask,
                     x_padding_mask, 
                     y_input_padding_mask) #tgt_is_causal = True)
        pred = pred.permute(1, 2, 0) 
        y_output = y_output.permute(1, 0) 

        loss = loss_fn(pred, y_output)
        epoch_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print("\nTraining loss:",epoch_loss / nb_batches)
    return epoch_loss / nb_batches

#########################################################
def valid_loop(dataloader, model, loss_fn):

    size = len(list(dataloader.dataset))
    nb_batches = len(dataloader)
    valid_loss = 0
    
    with torch.no_grad():
        for batch, (X, y,X_l,y_l) in enumerate(dataloader):
            
            X = X.to(model.DEVICE)
            y = y.to(model.DEVICE)
            
            y_input = y[:,:-1] 
            y_output = y[:,1:] 
            
            l = y_input.size(1) # la longueur maximale d'un élément du batch
            y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)
             
            x_padding_mask = (X[:,:,0] == model.x_pad_idx).to(model.DEVICE)
            y_input_padding_mask = (y_input == model.y_pad_idx).to(model.DEVICE)
            
            # Compute prediction and loss
            pred = model(X.float(),y_input,
                         y_output_mask,
                         x_padding_mask,
                         tgt_key_padding_mask = y_input_padding_mask)
            
            pred = pred.permute(1, 2, 0)      
            y_output = y_output.permute(1, 0) 
            
            loss = loss_fn(pred, y_output)
            valid_loss += loss.item()
    
    valid_loss /= nb_batches
    
    return valid_loss
