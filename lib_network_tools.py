#genera libs
from enum import auto
from tqdm import tqdm
import time
import numpy as np

#torch
import torch
from torch import nn

#GPU cache clearing lib
import gc
from GPUtil import showUtilization as gpu_usage
from numba import cuda

#sklearn metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

#custom libs
from lib_generator_tools import *



def FreeGpuCache():
    """

    Clears GPU cache and shows usage before and after

    """

    #clear cache using torch
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

def CountParameters(model):
    
    """
    
    Counts updatable parameters within a neural network
    

    """
    
    #create dict
    params_dict = {}
    total_params = 0
    
    #loop through trainable parameters to get names and params
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        params_dict.update({name : params})
        total_params+=params
     
    #show total params
    print(f"Total Trainable Params: {total_params}")
    return params_dict

def ClipGradient(model=None, clip_value=2.0, hard_clip=False): 

    #ref:https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem

    """

    Limits gradient updates applied to model during backpropagation

    """

    if hard_clip == True:
        #hardclipping
        nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

    else:
        #norm clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value, norm_type=2)

    return None

def GetF1Score(y_true, y_prob):
    """
    
    Caclulate weighted F1 score from log probabilities 
    
    """
    y_pred =  torch.max(y_prob, 1)[1]
    F1_score = f1_score(y_true.cpu(), y_pred.cpu(), average = 'weighted')

    return F1_score 
        
def GetAccuracy(y_true, y_prob):
    """
    
    Caclulate accuracy from log probabilities 
    
    """
    
    num_corrects = (torch.max(y_prob, 1)[1].view(y_true.size()).data == y_true.data).float().sum()
    acc = 100.0 * num_corrects/len(y_true)
    
    return acc  

def GetYPredNN(network=None, X=None, y=None, z=None, valid_iter=None,device='cuda', batch_size=32, split=0.85, drop_last=True):

    #get sent network to device
    network = network.to(device)

    #create generator for X_test
    if valid_iter ==None:
        BatchGen = BatchGenerator(X=X, y=y) #y set to X 
        train_iter, valid_iter, _ = BatchGen.MixedIterator(split=split, batch_size=batch_size, drop_last=drop_last)
        
    #set network to eval
    network.eval()
    
    y_preds = []
    y_trues = []

    #extract data
    for batch in valid_iter:
        #format data for relevant inputs
        if type(valid_iter.dataset.dataset) == CustomTopicDataset:
            batch_text, batch_topics, batch_labels = batch
            batch_topics = torch.autograd.Variable(batch_topics).float()
            batch_topics = batch_topics.to(device)
        else:
            batch_text, batch_labels = batch 
        sents, _ = batch_text
        sents = sents.to(device)
        labels = batch_labels.to(device)


        #get logits
        output = network(sents)
    
        #convert to predictions
        _, y_pred = torch.max(output.data, 1)

        #detach to numpy
        if device == 'cuda':
            y_pred = y_pred.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        else:
            y_pred = y_pred.detach().numpy()
            labels = labels.detach().numpy()
            
        y_preds.extend(y_pred)
        y_trues.extend(labels)

    return y_preds, y_trues

class NetModeller():
    """

    Wrapper for objective function in optuna studies
    
    """

    def __init__(self, network, train_iter, valid_iter, loss_fn = nn.CrossEntropyLoss(), 
                 optimizer = None, epochs = 10, clip_value = None, device = None):
        
        #init vars        
        self.network = network
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.clip_value = clip_value
        self.device = device

    def NetworkTrain(self):
        """

        Trains a trainable network and returns mean loss score
        
        """
        
        
        #activate eval mode
        self.network.train().to(self.device)
        
        #set train loss
        cumu_loss = 0.0
        cumu_acc = 0.0

        #training loop
        for _, batch in enumerate(self.train_iter):

            #zero gradient
            for param in self.network.parameters():
                param.grad = None

            #format data for relevant inputs
            if type(self.train_iter.dataset.dataset) == CustomTopicDataset:
                batch_text, batch_topics, batch_labels = batch
                batch_topics = torch.autograd.Variable(batch_topics).float()
                batch_topics = batch_topics.to(self.device)
            else:
                batch_text, batch_labels = batch 
        
            #prepare inputs
            sents, lengths = batch_text
            lengths = lengths.to(self.device)
            sents = sents.to(self.device)

            #prepare target
            batch_labels = torch.autograd.Variable(batch_labels).long()
            y_true = batch_labels.to(self.device)

            #enable grad
            with torch.set_grad_enabled(True):
                #get preds
                if type(self.train_iter.dataset.dataset) == CustomTopicDataset:
                    y_prob = self.network(sents, lengths, batch_topics).to(self.device)
                    del(sents, lengths, batch_topics)
                else:
                    y_prob = self.network(sents, lengths).to(self.device)
                    del(sents, lengths)

                #calculate loss items
                loss = self.loss_fn(y_prob, y_true)    
                
                #backprop
                loss.backward()

                #acc if not regression
                acc = GetF1Score(y_true, y_prob)

                #clip gradient
                if self.clip_value:
                    ClipGradient(self.network, self.clip_value, hard_clip=False)

                #step
                self.optimizer.step()
            
                #collect garbage
                FreeGpuCache()

            #cumulative loss
            cumu_loss += loss.item()

            #acc if not regression
            cumu_acc += acc.item()
    
        #mean loss across all batches
        epoch_loss = cumu_loss/len(self.train_iter)

        #cumulative accuracy
        epoch_acc = cumu_acc/len(self.train_iter)

        return epoch_loss, epoch_acc, self.optimizer

    def NetworkEval(self):
        """
        
        Evaluates a network and returns mean loss score
        
        """
        
        #activate eval mode
        self.network.eval().to(self.device)

        #create target loss
        cumu_loss = 0.0
        cumu_acc = 0.0

        #training loop
        for _, batch in enumerate(self.valid_iter):

            with torch.no_grad() :
            
            #format data for relevant inputs
                if type(self.valid_iter.dataset.dataset) == CustomTopicDataset:
                    batch_text, batch_topics, batch_labels = batch
                    batch_topics = torch.autograd.Variable(batch_topics).float()
                    batch_topics = batch_topics.to(self.device)
                else:
                    batch_text, batch_labels = batch 
            
                sents, lengths = batch_text
                lengths = lengths.to(self.device)
                sents = sents.to(self.device)

                #use float if regression
                batch_labels = torch.autograd.Variable(batch_labels).long()
                y_true = batch_labels.to(self.device)
            
                #get preds
                if type(self.valid_iter.dataset.dataset) == CustomTopicDataset:
                    y_prob = self.network(sents, lengths, batch_topics).to(self.device)
                    del(sents, lengths, batch_topics)
                else:
                    y_prob = self.network(sents, lengths).to(self.device)
                    del(sents, lengths)
                
                #calculate items
                loss = self.loss_fn(y_prob, y_true) 

            #calculate accuracy
            acc = GetF1Score(y_true, y_prob)

            #update loss
            cumu_loss += loss.item()
            cumu_acc += acc.item()

            #collect garbage
            FreeGpuCache()

        #mean loss across all batches
        epoch_loss = cumu_loss/len(self.valid_iter)

        #cumulative accuracy
        epoch_acc = cumu_acc/len(self.valid_iter)

        return epoch_loss, epoch_acc

    def SingleNetworkWrapper(self):
        """
        
        Return single training epoch
        
        """

        #train model using train_loader
        train_loss, train_acc, optimizer = self.NetworkTrain()
        
        #validate if valid_loader == True
        if self.valid_iter:
            valid_loss, valid_acc = self.NetworkEval()
        else:
            valid_loss, valid_acc = -999, -999

        return train_loss, train_acc, optimizer, valid_loss, valid_acc
        
    def BatchedNetworkWrapper(self):
        """
        
        Wrapper function for neural network training and evaluation
        
        """
        
        #create empty target lists
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []
        
        #print model
        print_line_len = 80
        print("*"*print_line_len)
        print("\nModel framework... \n")
        print(self.network, '\n')
        print("*"*print_line_len)
        
        #print table
        print("\nStarted training using {}\n".format(self.device))
        print("-"*print_line_len)

        
        #training loop
        for epoch in range(0, self.epochs):

            #show gpu usage
            gpu_usage()
            
            train_loss, train_acc, optimizer, valid_loss, valid_acc = self.SingleNetworkWrapper()
            
            #calculate time and show results
            print(f"{epoch + 1:^5.0f} | {train_loss:^12.6f} | {train_acc:^12.2f} | {valid_loss:^10.5f} | {valid_acc:^10.2f}")
        
            #append items to out lists
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            #clear cache
            FreeGpuCache()    
                        
        #create dict of results
        resdict = {}
        resitems = [{'train_losses':train_losses}, {'train_accs':train_accs}, {'valid_losses':valid_losses}, {'valid_accs':valid_accs}]
        print("-"*print_line_len)
        [resdict.update(x) for x in resitems]
        
        return resdict
