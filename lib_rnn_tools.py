#general libraries
import random 
import numpy as np
import pandas as pd
import time
import os
import re
from tqdm import tqdm
from collections import defaultdict

#nltk
import nltk
from nltk.tokenize import word_tokenize
#nltk.download("all")

#pytorch libraries
import torch
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.functional import F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


def TokenizeTexts(texts, tokenizer = lambda x: x.split()):
    """

    Returns list of tokenized setences from list of strings or pd.Series given a specific tokenizer

    """ 
    

    #perform tokenization
    if type(texts) == pd.core.series.Series:
        texts = texts.str.lower()
        tokenized_texts_out = texts.apply(tokenizer)
    else:
        texts = pd.Series(texts)
        tokenized_texts_out = texts.apply(tokenizer)
    
    
    return tokenized_texts_out

def WordToIdx(tokenized_texts, padding=True, specials=True):
    """

    Creates dictionary containing tokens and their corresponding index numbers

    """ 
    
    assert type(tokenized_texts) in [list, pd.core.series.Series], 'input type must be in list or pd.Series'
    
    #create empty dict
    word2idx = {}
    
    #add <pad> and <unk> tokens to the vocabulary
    if padding:
        word2idx['<pad>'] = 1
    if specials:
        word2idx['<unk>'] = 0
        
    #set idx to 2 to not overwrite <pad> or <unk>
    idx = 2
    
    #iterate through tokenized texts and update dictionary
    for tokenized_text in tokenized_texts:
        #add new token to `word2idx`
        for token in tokenized_text:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
   
    return word2idx

def GetPercLen(tokenized_texts, percentile = 100):
    """
    
    Returns the setence length or tweets for a given percentile

    """
    #get percentile length
    perc_len = np.percentile([len(x) for x in tokenized_texts], percentile)
    
    #round percentile length
    rounded_perc_len = int(np.round(perc_len))
    
    return rounded_perc_len

def EncodeTokensFixedLen(tokenized_texts, word2idx, des_len):
    """
    
    Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.
    
    """
    
    #error check
    assert type(tokenized_texts) in [list, pd.core.series.Series], 'input type must be in list or pd.Series'

    #loop through tokens
    encoded_tokens = []
    for tokenized_sent in tokenized_texts:
        
        #crop long sents
        if len(tokenized_sent) > des_len:
            tokenized_sent = tokenized_sent[0:des_len]
            
        #pad sentences to max_len
        tokenized_sent += ['<pad>'] * (des_len - len(tokenized_sent))

        #encode tokens to input_ids
        encoded_token = [int(word2idx.get(token)) for token in tokenized_sent]
        encoded_tokens.append(np.array(encoded_token, dtype = int)) 
    
    return encoded_tokens

def EncodeTokensMixedLen(tokenized_texts, word2idx):
    """
    
    Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.
    
    """
    
    #error check
    assert type(tokenized_texts) in [list, pd.core.series.Series], 'input type must be in list or pd.Series'

    encoded_tokens = []
    for tokenized_sent in tokenized_texts:
        
        #encode tokens to input_ids
        encoded_token = [int(word2idx.get(token)) for token in tokenized_sent]
        encoded_tokens.append(np.array(encoded_token, dtype = int)) #keep as array for manipulation
    
    return encoded_tokens

def LoadPretrainedVectorsFromText(word2idx, embedding_file_path = None, embedding_dim = 300):
    """
    
    Get vectors words in word2idx from pre-trained embedding text file

    """
    
    print("\nLoading pretrained vectors...")
    fin = open(embedding_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #     n, d = map(int, fin.readline().split())

    #get embedding dimension from first line
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        break
       
    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    if word2idx['<pad>']:
        embeddings[word2idx['<pad>']] = np.zeros((embedding_dim,))
    
    #load pretrained vectors
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:embedding_dim+1], dtype=np.float32)

    print(f"\nThere are {count} / {len(word2idx)} pretrained vectors found.")
    
    embeddings = Tensor(embeddings)
    
    return embeddings


def GetPreTrained(word2idx=None, pretrained_model='cskip_wiki_100d', pretrained_dir = './embedding/pretrained_embeddings/'):
    """

    Returns the path of the file for a specified pre-trained embedding

    """
    #get full_path
    fullpath = pretrained_dir+pretrained_model+'.txt'
    
    #get file embedding
    embedding_str = 'cskip_wiki_100d'[-4:-1]
    embedding_dim = int(embedding_str)

    #load pretrained model
    pre_weights = LoadPretrainedVectorsFromText(word2idx=word2idx, embedding_file_path=fullpath, embedding_dim=embedding_dim)
    
    #show pretrained vector shape
    print("Vector size:", pre_weights.shape)
    
    return pre_weights, embedding_dim
