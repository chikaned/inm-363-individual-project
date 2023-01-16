#import gen libs
import numpy as np
import pandas as pd

#pytorch
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

#import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#gert bert from transformers
from transformers import BertTokenizer

#octis dataset
from octis.dataset.dataset import Dataset as Oct_Dataset

#import custom libs
from lib_rnn_tools import *
from lib_embedding_shallow import *


################################################################################################################################

class CustomOctisDataset():
    """

    Object for CustomOctisDataset

    """   
    def __init__(self, X=None, y=None, save_dir = r'./octis_datasets/' ):
        
        self.X = X
        self.y = y
        self.save_dir = save_dir
    
    #assign functions
    
    def GenerateOctisDataset(self, train_split = 0.7):
        """

        Wrapper function to generate an Octis Dataset for Topic Modelling
        
        """

        #check if dirpath exists 
        if os.path.exists(self.save_dir):
            print("{} already exists!".format(self.save_dir))
            pass
        else:
            print("{} created!".format(self.save_dir))
            os.mkdir(self.save_dir)
        
        #save vars
        print("\n Making new dataset...\n")
        corpus_df = self.GetCorpus(X=self.X, y=self.y, train_split=train_split, save_dir=self.save_dir)
        vocab = self.GetVocab(self.X, save_dir=self.save_dir)
        self.GetMetaData(X=self.X, vocab=vocab, corpus_df=corpus_df, save_dir = self.save_dir)

        #get dataset
        octis_dataset = Oct_Dataset()
        octis_dataset.load_custom_dataset_from_folder(self.save_dir)

        return octis_dataset

    def GetCorpus(self, X=None, y=None, train_split=None, save_dir = './octis_datasets/'):
        """

        Creates corpus.tsv file for Octis dataset
            
        """

        #get test, train indexes
        if train_split > 0.0:
            train, _ = train_test_split(X, test_size=train_split, shuffle=False)
            train_indexes = train.index.tolist()

            #create valid_list
            valid_list = []
            for i in range(len(X)):
                if i in train_indexes:
                    valid_list.append('train')
                else:
                    valid_list.append('val')

            #create df
            X_series = pd.Series(X.values)
            valid_series = pd.Series(valid_list)
            corpus_df = pd.DataFrame([X_series, valid_series]).T
        
        elif train_split == 0.0:
            #create df
            X_series = pd.Series(X.values)
            valid_series = pd.Series(['train' for x in X_series])
            corpus_df = pd.DataFrame([X_series, valid_series]).T


        #create save path
        save_path = save_dir+'corpus.tsv'
        print("Corpus saved to : {}".format(save_path))

        #save corpus to tsv
        corpus_df.to_csv(save_path, sep="\t", header=False, index=False)

        return corpus_df

    def GetVocab(self, X=None, save_dir = './octis_datasets/'):
        """
        
        Produce vocab text file for custom Octis Dataset
        
        """

        #get vocab
        vectorizer = CountVectorizer()
        vectorizer.fit(X)
        vocab = [x for x in vectorizer.vocabulary_.keys()]

        #create save path
        save_path = save_dir+'vocab.txt'
        print("Vocab saved to : {}".format(save_path))

        #save vocab to txt file
        with open(save_path, 'w') as fp:
            for item in vocab:
                # write each item on a new line
                fp.write("%s\n" % item)

        return vocab  

    def GetMetaData(self, X=None, vocab=None, corpus_df=None, save_dir = None):
        """
        
        Produce metadata json file for custom implementation of Octis datasets
        
        """

        #get vars
        total_documents = len(X)
        vocabulary_length = len(vocab)
        preprocessing_info = []
        labels = []
        total_labels = 0
        max_train_ind = max(corpus_df[corpus_df.iloc[:,1]=='train'].index)
        last_training_doc = max_train_ind
        if 'val' in corpus_df.iloc[:,1]:
            max_val_ind = max(corpus_df[corpus_df.iloc[:,1]=='val'].index)
            last_validation_doc = max_val_ind
        else:
            last_validation_doc = max_val_ind = 'N/A'
        

        #create dict
        metadata = {"total_documents": total_documents,
                     "vocabulary_length": vocabulary_length, 
                     "preprocessing-info": preprocessing_info,
                     "labels": labels, 
                     "total_labels": total_labels, 
                     "last-training-doc": last_training_doc, 
                     "last-validation-doc": last_validation_doc}

        #create save path
        save_path = save_dir+'metadata.json'
        print("Metadata saved to : {}".format(save_path))

        #save to json file
        pd.Series(metadata).to_json(save_path)

        return metadata

class PredDataset(torch.utils.data.Dataset):
    """

    Dataset class for predicting texts using a trained RNN model
    
    """

    def __init__(self, X=None):

        self.X = X


    def __len__(self):
        """

        Get len of input item

        """    
        
        return len(self.X)

    def __getitem__(self, index):
        """

        Return input and label at index

        """  
    
        _x = self.X[index]

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.X[idx]

class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, X=None, y=None):
        """

        Creates custom torch Dataset to be loaded in to torch Dataloader with __len__ and __getiem__ functions for embedding

        """
        self.X = X
        self.y = y

        if len(self.X) != len(self.y):
            raise Exception("The length of X does not match the length of y")

    def __len__(self):
        """

        Get len of input item
        
        """    
        
        return len(self.X)

    def __getitem__(self, index):
        """

        Return input and label at index

        """  
    
        _x = self.X[index]
        _y = self.y[index]

        return (_x, _y)
    
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.y[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.X[idx]

class CustomTopicDataset(torch.utils.data.Dataset):
    
    def __init__(self, X=None, y=None, z=None):
        """
        Creates custom torch Dataset to be loaded in to torch Dataloader with __len__ and __getiem__ functions for embedding

        """
        self.X = X
        self.y = y
        self.z = z

        if len(self.X) != len(self.y):
            raise Exception("The length of X does not match the length of y")
            
    def __name__(self):
        return 'CustomTopicDataset'

    def __len__(self):
        """

        Get len of input item
        
        """    
        
        return len(self.X)

    def __getitem__(self, index):
        """

        Return input and label at index

        """  
    
        _x = self.X[index]
        _y = self.y[index]
        _z = self.z[index]

        return (_x, _y, _z)
    
    def get_batch_labels(self, idx):
        #fetch a batch of labels
        return np.array(self.y[idx])

    def get_batch_texts(self, idx):
        #fetch a batch of inputs
        return self.X[idx]
    
    def get_batch_topics(self, idx):
        #fetch a batch of topics
        return self.z[idx]
  
#################################################################################################################################

class BatchGenerator():
    """

    Object for creating bucket-style dataloaders for training neural text classifiers
    """

    def __init__(self, X=None, y=None, z=None):
        
        #init vars        
        self.X = X
        self.y = y
        self.z = z

    def StringToCategoryTensor(self, y):
        """

        Converts list of strings with binary variables into binary torch.Tensor

        """
        if type(y[0]) == str:
            if len(np.unique(y)) == 2:
                y = [1 if x == 'fake' else 0 for x in y]
            else:
                y = pd.factorize(self.y)[0] #[0] returns array of categorized data [1] returns index 

        #return tensor  
        y = Tensor(y)
            
        return y
    
    def DatasetSplitter(self, dataset, split = 0.85):
        """

        Creates training and validation datasets

        """
        #get lengths
        dataset_length = len(dataset)
        train_size = train_size = int(split * dataset_length)
        valid_size = dataset_length - train_size

        #split dataset
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset = dataset, 
                                                                     lengths = [train_size, valid_size],
                                                                     generator=torch.Generator().manual_seed(42))

        return train_dataset, valid_dataset
    
    def collate_fn_mixed(self, batch):
        """

        Returns variable padded batches where padding length is taken from the longest sequence in each batch

        """

        #get batch and batch length
        batch_output = zip(*batch)
        batch_len = len(list(batch_output))
        
        #get topic data if relevant
        if batch_len == 3:
            inputs, labels, topics = zip(*batch)#get batched inputs and labels
            out_topics = torch.Tensor(np.array(topics)).to(torch.float64)
        else:
            inputs, labels = zip(*batch)#get batched inputs and labels
        
        #get lens
        input_lens = [len(x) for x in inputs]
        out_lens = torch.Tensor(input_lens).to(torch.int32)

        #pad sequences
        ten_inputs = [torch.Tensor(np.array(x, dtype = np.int32)) for x in inputs]
        padded_inputs = pad_sequence(ten_inputs, batch_first=True, padding_value= 0)
        out_inputs = padded_inputs.to(torch.int32)

        #format labels
        out_labels = torch.Tensor(np.array(labels)).to(torch.float32)
        
        #return appropriate out
        if batch_len == 3:
            return [(out_inputs, out_lens), out_topics, out_labels]
        else:
            return [(out_inputs, out_lens), out_labels]
        
    def MixedIterator(self, padding = True, specials = True, batch_size = 32, split = 0.85, shuffle = False, regression=False, drop_last = True):
        """
        
        Creates a batched iterator with varying sequence lengths between batches and a word to index dictionary
        
        """
        
        if self.y is not None:
            #checl for regression
            if regression == False:
                y = self.StringToCategoryTensor(self.y)
            else:
                y = self.y
        
        #tokenize text and get index
        tokenized_texts =  TokenizeTexts(self.X)
        wordtoidx = WordToIdx(tokenized_texts, padding = padding, specials = specials)
        encoded_tokens_mixed = EncodeTokensMixedLen(tokenized_texts, wordtoidx)

        #create dataset and batches
        if self.z is not None:
            #print("\nTopics detected!")
            self.MixedDataset = CustomTopicDataset(encoded_tokens_mixed, y, self.z)
        else:
            #print("\nNo Topics detected!")
            self.MixedDataset = CustomDataset(encoded_tokens_mixed, y)

        if split < 1.0:
            #split dataset
            TrainDataset, ValidDataset = self.DatasetSplitter(self.MixedDataset, split)
        
            #create dataloader
            MixedTrainDataLoader = torch.utils.data.DataLoader(dataset=TrainDataset, collate_fn=self.collate_fn_mixed, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=1) #reverse size order
            MixedValidDataLoader = torch.utils.data.DataLoader(dataset=ValidDataset, collate_fn=self.collate_fn_mixed, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=1)
            
            return MixedTrainDataLoader, MixedValidDataLoader, wordtoidx
        
        else:
            #create dataloader
            MixedTrainDataLoader = torch.utils.data.DataLoader(dataset=self.MixedDataset, collate_fn=self.collate_fn_mixed, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=1)

            return MixedTrainDataLoader, None, wordtoidx