#general libraries
import random 
import numpy as np
import pandas as pd
import time
import os

#pytorch libraries
import torch
from torch.autograd import Variable
from torch import nn
from torch.functional import F
import torch.optim as optim


class GRU(nn.Module):
    """
    
    Gated Recurrent Unit model used text classification tasks

    """

    def __init__(self, vocab_size=None, embedding_dim=None, hidden_dim=None, num_layers=None, bidirectional=None, 
                 rnn_layer_dropout=None, rnn_out_dropout=0., output_dim=None,  batch_first=None, weights=None, unk_token=None, 
                 pad_token=None, device=None, topic_dim=0):
        super(GRU, self).__init__()

        #embedding layer
        if weights is not None:
            self.vocab_size, self.embedding_dim = weights.shape
            self.embed = nn.Embedding.from_pretrained(weights)
        else:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        
        self.embed.weight.data[unk_token] = torch.zeros(self.embedding_dim)
        self.embed.weight.data[pad_token] = torch.zeros(self.embedding_dim)
        
        #specify device
        self.device = device
        self.topic_dim = topic_dim

        #lstm layer vars
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_layer_dropout = rnn_layer_dropout
        self.bidirectional = bidirectional
        self.directionality = 1
        if self.bidirectional == True:
            self.directionality = 2   
        self.batch_first = batch_first
        self.rnn = nn.GRU(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers,
                          batch_first = self.batch_first, bidirectional = self.bidirectional, dropout = self.rnn_layer_dropout) #dropout only effective for multi-layer

        #forward carry vars
        self.output_dim = output_dim
        self.rnn_out_dropout = nn.Dropout(p=rnn_out_dropout)
        self.fc = nn.Linear(self.hidden_dim*self.directionality+self.topic_dim, self.output_dim)

    def ScaleTensor(self, outmap):
        "scales tensors between 0-1"

        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min) # Broadcasting rules apply
        return outmap

    def forward(self, sentence=None, sentence_lengths=None, topic_preds=None):
        """
        
        Completes forward pass

        """

        #get embedded sentence
        embed_out = self.embed(sentence)
                
        #determine shape
        if self.batch_first == True:
            #take first dim as batch_size
            current_batch_size = embed_out.shape[0]
        else:
            #take second dim as batch_size
            current_batch_size = embed_out.shape[1]
        
        #create hidden layers (D∗num_layers, N, Hout) for batched // (D∗num_layers, Hout) for unbatched (N = batch_size)
        h_0 = Variable(torch.zeros(self.num_layers*self.directionality, current_batch_size, self.hidden_dim)) 
        h_0 = h_0.to(self.device)

        #feed into lstm layer
        self.rnn.flatten_parameters()
        _ , final_hidden_state  = self.rnn(embed_out, h_0)

        #reformat final hidden state
        if self.bidirectional == True:
            final_hidden_state_fc = torch.cat((final_hidden_state[-1,:,:], final_hidden_state[-2,:,:]), dim =1)
        else:
            final_hidden_state_fc = final_hidden_state[-1,:,:].squeeze()

        #scale tensor
        final_hidden_state_fc = self.ScaleTensor(final_hidden_state_fc)

        #feed final hidden state into fc layer
        fc_in = self.rnn_out_dropout(final_hidden_state_fc)
        
        #add topic layer       
        if self.topic_dim > 0:
            fc_in = torch.cat((fc_in, topic_preds), dim=1)

        #pass to linear layer
        model_out = self.fc(fc_in.type(dtype = torch.float32))

        #return out
        return model_out 
      
class EmbedCNN(nn.Module):
    """
    
    1-D CNN model used text classification tasks

    """
    def __init__(self, weights=None, freeze_embedding=False, vocab_size=None, embedding_dim=300, filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100], output_dim=2, cnn_out_dropout=0., unk_token = 0, pad_token = 1, device = None, 
                 topic_dim=0):

        super(EmbedCNN, self).__init__()
        
        #assign topic dims
        self.topic_dim = topic_dim
        self.device = device
    
        #embedding layer
        if weights is not None:
            self.vocab_size, self.embedding_dim = weights.shape
            self.embed = nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)
        else:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
            
        self.embed.weight.data[unk_token] = torch.zeros(self.embedding_dim)
        self.embed.weight.data[pad_token] = torch.zeros(self.embedding_dim)
        
        #conv layers
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim, out_channels=num_filters[i], 
                                          kernel_size=filter_sizes[i]) for i in range(len(filter_sizes))])
        
        #fc layer and dropout
        self.fc = nn.Linear(np.sum(num_filters)+self.topic_dim, output_dim)
        self.cnn_out_dropout = nn.Dropout(p=cnn_out_dropout)

    def ScaleTensor(self, outmap):
        """
        
        scales tensors between 0-1
        
        """

        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min) # Broadcasting rules apply
        return outmap

    def forward(self, sentence=None, sentence_lengths=None, topic_preds=None):
        """
    
        Gated Recurrent Unit model used text classification tasks

        """

        # Get embeddings. Output shape: (b, max_len, embed_dim)
        embed_out = self.embed(sentence).float()

        # Permute to match input shape requirement of `nn.Conv1d`. Output shape: (b, embed_dim, max_len)
        embed_reshaped = embed_out.permute(0, 2, 1)
        del(embed_out)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        conv_out = [F.relu(conv1d(embed_reshaped)).to(self.device) for conv1d in self.conv1d_list]
        del(embed_reshaped)

        # Max pooling. Output shape: (b, num_filters[i], 1)
        pool_out = [F.max_pool1d(conv, kernel_size= int(conv.shape[2])) for conv in conv_out]
        del(conv_out)

        # Concatenate x_pool_list to feed the fully connected layer. Output shape: (b, sum(num_filters))
        cat_out = torch.cat([pool.squeeze(dim=2) for pool in pool_out], dim=1)
        del(pool_out)

        #scale tensor
        cat_out = self.ScaleTensor(cat_out)

        # Compute logits. Output shape: (b, n_classes)
        fc_in = self.cnn_out_dropout(cat_out)
        del(cat_out)

        #add topic layer       
        if self.topic_dim > 0:
            fc_in = torch.cat((fc_in, topic_preds), dim=1)

        #calculate linear layer
        model_out = self.fc(fc_in)

        return model_out