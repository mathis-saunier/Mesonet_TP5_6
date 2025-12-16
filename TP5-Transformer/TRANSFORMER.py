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
        #return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        #print("token_embedding.shape",token_embedding.shape)
        #print("pos_encoding.shape",self.pos_encoding.shape)
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.size(1), :])

    
###########################################################
class Transformer(nn.Module):
    def __init__(self, config,device):
        super(Transformer, self).__init__()
        self.input_features = config['input_features']
        self.batch_size = config['batch_size']
        self.num_epochs =config['num_epochs']
        self.learning_rate =config['learning_rate']
        self.num_classes =config['num_classes']
        self.hidden_size =config['hidden_size']
        self.num_heads =config['num_heads']
        self.num_layers =config['num_layers']
        self.dropout =config['dropout']
        self.pad_idx =config['x_pad_idx']
        self.max_length =config['max_length']
        self.START_TOKEN =config['START_TOKEN']
        self.END_TOKEN =config['END_TOKEN']
        self.DEVICE = device

        self.positional_encoding_layer = PositionalEncoding(dim_model = self.hidden_size, 
                                                            dropout_p = self.dropout, 
                                                            max_len = self.max_length)
        
        self.x_embedding = nn.Linear(config['input_features'],config['hidden_size'])        
        
        self.y_embedding = nn.Embedding(config['num_classes'],config['hidden_size'])         
        
        self.transformer = nn.Transformer(d_model=config['hidden_size'], 
                                                      nhead=config['num_heads'], 
                                                      num_encoder_layers =config['num_layers'],
                                                      num_decoder_layers =config['num_layers'],
                                                      dim_feedforward = config['hidden_size'],
                                                      dropout=config['dropout'],
                                                      batch_first=True)
        # Output layer for text prediction
        self.output_layer = nn.Linear(config['hidden_size'], config['num_classes'])

    def forward(self, x,y,y_output_mask,src_key_padding_mask,tgt_key_padding_mask):

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        #x = self.x_embedding(x) * math.sqrt(self.hidden_size)
        x = self.x_embedding(x)

        x = self.positional_encoding_layer(x)
        
        y = self.y_embedding(y) * math.sqrt(self.hidden_size)
        y = self.positional_encoding_layer(y)

        #print("x.shape",x.shape)
        #print("y.shape",y.shape)
        #print("src_key_padding_mask.shape",src_key_padding_mask.shape)
        #print("y_output_mask.shape",y_output_mask.shape)
        #print("tgt_key_padding_mask.shape",tgt_key_padding_mask.shape)
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(x, y,
                                           tgt_mask = y_output_mask,
                                           src_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)
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
    #print("Training loop...")
    size = len(dataloader.dataset)
    nb_batches = len(dataloader)
    epoch_loss = 0
    
    model.train()
    
    for batch, (X, y,X_l,y_l) in enumerate(dataloader):
        
        X = X.to(model.DEVICE)
        y = y.to(model.DEVICE)

        #y_input = y[:-1,:] # on retire le END_TOKEN car nest jamais traité en entrée
        #y_output = y[1:,:] # on ne cherche jamais à predire le START_TOKEN
        
        y_input = y[:,:-1] 
        y_output = y[:,1:] 
        
        l = y_input.size(1) # la longueur maximale d'un élément du batch
        #print("y_input.size(0)",y_input.size(0),y_input.size(1))
        y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)

        #x_padding_mask = (X[:,:,0] == model.pad_idx).transpose(0, 1).to(model.DEVICE)
        #y_input_padding_mask = (y_input == model.pad_idx).transpose(0, 1).to(model.DEVICE)
        
        x_padding_mask = (X[:,:,0] == model.pad_idx).to(model.DEVICE)
        y_input_padding_mask = (y_input == model.pad_idx).to(model.DEVICE)
        
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
            
            #y_input = y[:-1,:] # on retire le END_TOKEN car nest jamais traité en entrée
            #y_output = y[1:,:] # on ne cherche jamais à predire le START_TOKEN
            y_input = y[:,:-1] 
            y_output = y[:,1:] 
            
            l = y_input.size(1) # la longueur maximale d'un élément du batch
            y_output_mask = model.get_tgt_mask(l).to(model.DEVICE)
            
            #x_padding_mask = (X[:,:,0] == model.pad_idx).transpose(0, 1).to(model.DEVICE)
            #y_input_padding_mask = (y_input == model.pad_idx).transpose(0, 1).to(model.DEVICE)
            
            x_padding_mask = (X[:,:,0] == model.pad_idx).to(model.DEVICE)
            y_input_padding_mask = (y_input == model.pad_idx).to(model.DEVICE)
            
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



