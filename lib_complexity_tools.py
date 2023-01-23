#general libs
import time
import pickle as pkl
from tqdm.notebook import tqdm_notebook

#pytorch libs
import torchvision
from torch import nn

#topic model libraries
from bertopic import BERTopic
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM

#custom libs
from lib_embedding_shallow import TFIDF
from lib_generator_tools import *
from lib_topic_tools import *
from lib_rnn_tools import *


"""

Library for getting prediction and training times for shallow and deep models

**********************************************************************************************************************************************

"""

def ReadPickle(file_path):
    """
    Read a pickle file in the specified location

    
    """
    
    tmp_file = open(file_path, 'rb')
    return pkl.load(tmp_file)
    

class ComplexityTools():
    """
    
    Class for extracting the trianing times, prediciton times and file sizes of text models

    """


    def __init__(self, model=None, pred_model=None):
        #assign vars
        self.model = model
        self.pred_model = pred_model

    def WarmUpGPU(self):
        
        print("\nWarming up GPU...\n")
        
        #enable onednn
        torch.jit.enable_onednn_fusion(True)

        #get random input
        sample_input = [torch.rand(32, 3, 224, 224)]

        #use resnet
        model = getattr(torchvision.models, "resnet50")().eval()

        #trace model with input
        traced_model = torch.jit.trace(model, sample_input)

        #freeze
        traced_model = torch.jit.freeze(traced_model)

        #execute warm-ups
        with torch.no_grad():
            traced_model(*sample_input)
            traced_model(*sample_input)
            traced_model(*sample_input)
        
        del(traced_model)
        
        return print("\nWarmup complete!\n")

    def GetSizeOf(self, object):
        """
        Gets size of serialized pickle file for object measure in bytes
            
        """
        object_size = len(pkl.dumps(object)) #in bytes
        
        return object_size

    def CreateDummyVars(self, dummy_len = 25600, vocab_size = 12180, seq_len = 1000, out_type = 'int'):                
        """

        Create dummy variables for evaluating model complexity with either string or int arrays
        
        """

        assert out_type in ['str', 'int'], "out type must be type 'str' or 'int'"
            
        #create empty target
        DummyVarList = []
        
        #get inv dict if needed
        if out_type == 'str':
            inv_dict = self.GetInvDict()

        #get list of dummy encoded sentences
        for i in range(dummy_len):
            sent = np.random.randint(vocab_size, size=(seq_len, ))
            
            if out_type == 'str':
                sent = [inv_dict[x] for x in sent]
                sent = ' '.join(sent)
            
            DummyVarList.append(sent)

        #get list of dummy encoded labels
        DummyTarget = np.random.randint(2, size=(dummy_len, ))
        
        return DummyVarList, DummyTarget 

    def GetInvDict(self, embedding_file_path = r'./embedding/pretrained_embeddings/glove_wiki_100d.txt'):
        """

        Get the inverted dictionary that can be used to generate words from ints using embedding word2idx (GloVe wiki 100d used by default)

        """
        #read_path
        fin = open(embedding_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        stoi = {}    

        #load pretrained vectors
        for i, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            stoi[i]=word

        return stoi

    def CreateDummyIter(self, dummy_len = 25600, vocab_size = 12180, seq_len = 1000, batch_size = 32):     
        """

        Create dummy iterator for neural network evaluation using 
        
        """
        
        #specify vars
        split = 1.0
        
        #get dummy arrs
        DummyTokenList, DummyTarget = self.CreateDummyVars(dummy_len = dummy_len, vocab_size = vocab_size, 
                                                    seq_len = seq_len, out_type = 'int')

        #convert to pd.Series
        DummyTokenSeries = pd.Series(DummyTokenList)
        DummyTargetSeries = pd.Series(DummyTarget)

        #create batchgen
        from lib_generator_tools import BatchGenerator
        BatchGen = BatchGenerator(X = DummyTokenSeries, y = DummyTargetSeries)

        #create dataset
        DummyDataset = CustomDataset(DummyTokenSeries, DummyTargetSeries)

        #create dataloader
        DummyDataLoader = torch.utils.data.DataLoader(dataset=DummyDataset, collate_fn= BatchGen.collate_fn_mixed, 
                                                        batch_size = batch_size, shuffle = False, drop_last = False)
        
        return DummyDataLoader

    def GetShallowComplexities(self, X, y):
        """
        
        Gets the size, prediction time and training time for traditional machine learning models
        
        """
        
        #get training time
        start_time = time.time()
        self.model.fit(X, y)
        train_time = (time.time() - start_time)

        #get prediction time
        start_time = time.time()
        _ = self.model.predict(X)
        pred_time = (time.time() - start_time)
        
        #get model size
        model_size = self.GetSizeOf(self.model) #size in calcualted in bytes
        
        return (train_time, pred_time, model_size)

    def GetShallowTimes(self, X, y):
        """
        
        Gets training and prediction times for shallow sklearn classifiers
        
        """
        #create target lists
        train_times, pred_times, model_sizes = [], [], []
        
        #train model
        train_model = self.model
        train_time, pred_time, model_size = self.GetShallowComplexities(X, y)
        train_times.append(train_time)
        pred_times.append(pred_time)
        model_sizes.append(model_size)
        del(train_model)
    
        #create dictionary output
        resdict = {}
        resdict['train_time'] = train_times    
        resdict['pred_time'] = pred_times
        resdict['model_size'] = model_sizes
        
        return resdict

    def CPUTrain(self, dataloader, loss_fn = nn.CrossEntropyLoss(), progress_bar=False):
        """
        
        Calculates cumulative feed-forward, backpropagation and optimization CPU times across a dataset in milliseconds
        
        """
        
        #set device
        device = 'cpu'
        
        #activate eval mode
        self.model.train().to(device)

        #set optimizer 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        #instatiate start time
        prep_time = 0
        pred_time = 0
        backprop_time = 0 
        opt_time = 0

        if progress_bar == True:
            loop_object = tqdm(dataloader, desc = 'GPU training')
        else:
            loop_object = dataloader
            
        #training loop
        for bi, batch in enumerate(loop_object):
            
            #start time
            start_time = time.time()
            
            #prep data
            optimizer.zero_grad()
            batch_text, batch_labels = batch 
            sents, lengths = batch_text
            lengths = lengths.to(device)
            sents = sents.to(device)
            batch_labels = torch.autograd.Variable(batch_labels).long()
            y_true = batch_labels.to(device)
            
            #calculate prep time
            prep_time += time.time() - start_time

            #enable grad
            with torch.set_grad_enabled(True):
                
                #start time
                start_time = time.time()
                
                #pred data
                y_prob = self.model(sents, lengths)
                
                #calculate prediction time
                pred_time += time.time() - start_time

                #calculate loss items
                loss = loss_fn(y_prob, y_true) 
                
                #start time
                start_time = time.time()
                
                #backprop
                loss.backward()
                
                #calculate backprop time
                backprop_time += time.time() - start_time
                    
                #start time
                start_time = time.time()

                #step time
                optimizer.step()
                
                #calculate step time
                opt_time += time.time() - start_time
                
        #convert times to milliseconds
        pred_time = pred_time * 1000
        backprop_time = backprop_time * 1000
        opt_time = opt_time * 1000
        train_time = backprop_time+opt_time
        
        return [pred_time, train_time]

    def GPUTrain(self,dataloader, loss_fn = nn.CrossEntropyLoss(), progress_bar=False):
        """
        
        Calculates cumulative feed-forward, backpropagation and optimization GPU times across a dataset in milliseconds
        
        """
        
        #set device
        device = 'cuda'
        
        #activate eval mode
        self.model.train().to(device)

        #set optimizer 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        #instatiate start time
        prep_time = 0
        pred_time = 0
        backprop_time = 0 
        opt_time = 0
        
        if progress_bar == True:
            loop_object = tqdm(dataloader, desc = 'GPU training')
        else:
            loop_object = dataloader
            
        #training loop
        for bi, batch in enumerate(loop_object):
            
            #start time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            #start record
            start.record()

            #prep data
            optimizer.zero_grad()
            batch_text, batch_labels = batch 
            sents, lengths = batch_text
            lengths = lengths.to(device)
            sents = sents.to(device)
            batch_labels = torch.autograd.Variable(batch_labels).long()
            y_true = batch_labels.to(device)

            #calculate prep time
            end.record()
            torch.cuda.synchronize()
            prep_time += start.elapsed_time(end)

            #enable grad
            with torch.set_grad_enabled(True):

                #start record
                start.record()

                #pred data
                y_prob = self.model(sents, lengths)

                #calculate prediction time
                end.record()
                torch.cuda.synchronize()
                pred_time += start.elapsed_time(end)

                #calculate loss items
                loss = loss_fn(y_prob, y_true) 

                #start record
                start.record()

                #backprop
                loss.backward()

                #calculate backprop time
                end.record()
                torch.cuda.synchronize()
                backprop_time += start.elapsed_time(end)

                #start record
                start.record()

                #step time
                optimizer.step()

                #calculate step time
                end.record()
                torch.cuda.synchronize()
                opt_time += start.elapsed_time(end)
        
        train_time = backprop_time+opt_time
        
        return [pred_time, train_time]

    def GetNeuralTimes(self, dummy_iter= None, epochs = 10,  device = 'cpu'):                
        """
        
        Wrapper function for neural model training and evaluation to get prediction, backpropagation and optimziation times
        with either CPU or GPU processing in milli
        
        """
        
        #check action type
        assert device in ['cuda', 'cpu'], "device must be in ['cuda', 'cpu']"
        
        #start vars
        times_list = []
        
        #state loss function
        loss_fn = nn.CrossEntropyLoss()
        
        print("Starting training using {}...".format(device))

        #start loop
        if device == 'cuda':
            self.WarmUpGPU()
            
        for epoch in tqdm_notebook(range(epochs), desc='Calculating runtimes'):
                
            if device == 'cuda':
                times = self.GPUTrain(dataloader = dummy_iter, loss_fn = loss_fn, progress_bar=False)
                
            if device == 'cpu':
                times = self.CPUTrain(dataloader = dummy_iter, loss_fn = loss_fn, progress_bar=False)

            #calculate training time
            times.append(self.GetSizeOf(self.model))
            times_list.append(times)
        
        #create times dict - sum times for epoch
        times_dict = {}
        times_dict.update({'pred_times': np.sum([x[0] for x in times_list])})
        times_dict.update({'train_times': np.sum([x[1] for x in times_list])})
        times_dict.update({'model_sizes': np.mean([x[2] for x in times_list])})
        
        print("Finished!")

        return times_dict  

    def GetShallowTopicTimes(self, X= None):
        """
        
        Get copmlexity times for non-contextualized Gensim topic models (LDA, NMF)
        
        """

        #create target lists
        train_times, pred_times, model_sizes = [], [], []
        
        #loop trhough training
        train_model = self.model
        train_time, pred_time, model_size = self.GetShallowTopicComplexities(X)
        train_times.append(train_time)
        pred_times.append(pred_time)
        model_sizes.append(model_size)
        del(train_model)
        
        #create dictionary output
        resdict = {}
        resdict['train_time'] = train_times   
        resdict['pred_time'] = pred_times
        resdict['model_size'] = model_sizes
        
        return resdict

    def GetDeepTopicTimes(self, X=None, epochs=20):
        """
        
        Get complexity times for contextualized topic models ZeroShotTM and CombinedTM
        
        """
        
        #get tokenized texts
        docs = X

        #static vars
        max_seq_length = 384 #embedding length

        #prep data 
        tp = TopicModelDataPreparation("all-MiniLM-L6-v2", max_seq_length=max_seq_length)

        #start time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        #start record
        start.record()

        #prepare data
        training_dataset = tp.fit(text_for_contextual=docs, text_for_bow=docs) #include labels for CTM - performed using cuda if available

        #calculate prep time
        end.record()
        torch.cuda.synchronize()
        prep_time = start.elapsed_time(end)

        if 'CombinedTM' in str(type(self.model)):
            model = CombinedTM(
                               bow_size=len(tp.vocab), 
                               contextual_size=384, #pretrained size https://www.sbert.net/docs/pretrained_models.html
                               n_components=5, 
                               num_epochs=epochs
                               )
        elif 'ZeroShotTM' in str(type(self.model)):
            model = ZeroShotTM(
                               bow_size=len(tp.vocab), 
                               contextual_size=384, #pretrained size https://www.sbert.net/docs/pretrained_models.html
                               n_components=5, 
                               num_epochs=epochs
                               )
        
        #assign vars
        model.num_epochs = epochs
        model.activation = self.model.activation
        model.optimizer = self.model.optimizer
        model.n_components = self.model.n_components
        model.dropout = self.model.dropout
        model.batch_size = self.model.batch_size
        model.lr = self.model.lr
        model.momentum = self.model.momentum
        model.hidden_sizes = self.model.hidden_sizes
        n_samples = 5

        if model.USE_CUDA == False:

            #start record
            start_time = time.time()

            model.fit(training_dataset, n_samples=n_samples)

            #calculate model time
            train_time = (time.time() - start_time)
            
            #start record
            start_time = time.time()

            #get train time
            model.get_doc_topic_distribution(training_dataset, n_samples=n_samples)

            #calculate model time
            pred_time = (time.time() - start_time)
            train_time = train_time*1000

        if model.USE_CUDA == True:

            #start loop
            self.WarmUpGPU()

            #start record
            start.record()

            model.fit(training_dataset, n_samples=n_samples)

            #calculate model time
            end.record()
            torch.cuda.synchronize()
            train_time = start.elapsed_time(end)
            
            #start record
            start.record()

            #get train time
            model.get_doc_topic_distribution(training_dataset, n_samples=n_samples)

            #calculate prediction time
            torch.cuda.synchronize()
            pred_time = start.elapsed_time(end)
        
        #get model size
        model_size = self.GetSizeOf(model)
        del(model)

        resdict = {}
        resdict['train_time'] = prep_time+train_time    
        resdict['pred_time'] = pred_time
        resdict['model_size'] = model_size
        
        return resdict

    def GetBerTopicTimes(self, X):
        """
        
        Get complexity times for contextualized topic models BerTopic
        
        """
        #start time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        #assign dummy string
        docs = X

        #bertopic hyperparams
        verbose = True
        calculate_probabilities = True

        #instantiate topic model
        model = BERTopic(language="english", 
                            calculate_probabilities = calculate_probabilities, 
                            verbose = verbose)

        #assign vars
        model.nr_topics = self.model.nr_topics
        model.n_gram_range = self.model.n_gram_range,
        model.top_n_words = self.model.top_n_words

        #start loop
        self.WarmUpGPU()

        #start record
        start.record()

        #clean doc
        cleaned_docs = model._preprocess_text(docs)

        #fit docs
        model.fit(cleaned_docs) 

        #calculate model time
        end.record()
        torch.cuda.synchronize()
        train_time = start.elapsed_time(end)

        #start record
        start.record()

        #fit docs
        model.transform(cleaned_docs) 

        #calculate model time
        end.record()
        torch.cuda.synchronize()
        pred_time = start.elapsed_time(end)

        #get model size
        model_size = self.GetSizeOf(model)

        resdict = {}
        resdict['train_time'] = train_time    
        resdict['pred_time'] = pred_time
        resdict['model_size'] = model_size
        del(model)
        
        return resdict

    def EvalauteModel(self, test_input=None, vocab_size =None, dummy_len = 2500, seq_len = 50, epochs = 10):
        """

        Wrapper function to evaluate shallow and deep models
           
        """
        
        if hasattr(self.model, 'predict'): #for shallow learning models
            print("\nModel type: shallow classifier {}\n".format(self.model))
            if test_input == None:
                tmp_X, dummy_y = self.CreateDummyVars(dummy_len = dummy_len, vocab_size = vocab_size, seq_len = seq_len, out_type = 'str')
            else:
                tmp_X, dummy_y = test_input
            csr_X, _ = TFIDF(tmp_X, norm = 'l2')
            dummy_X = csr_X.toarray()
            resdict = self.GetShallowTimes(dummy_X, dummy_y)

        elif hasattr(self.model, 'state_dict'):
            print("\nModel type: deep classifier {}\n".format(self.model))
            if test_input == None:
                dummy_iter = self.CreateDummyIter(dummy_len = dummy_len, vocab_size = self.model.vocab_size, seq_len = seq_len)
            else:
                dummy_iter = test_input
            resdict = self.GetNeuralTimes(dummy_iter= dummy_iter, epochs = epochs,  device = self.model.device)

        elif hasattr(self.model, "info"):
            assert 'Non-negative Matrix Factorization' or 'Latent Dirichlet Allocation' in self.model.info()['name'], "Check OCTIS model type"
            print("\nModel type: shallow topic model {}\n".format(self.model))
            if test_input == None:
                dummy_X, _ = self.CreateDummyVars(dummy_len = dummy_len, vocab_size = vocab_size, seq_len = seq_len, out_type = 'str')
            else:
                dummy_X = test_input
            resdict = self.GetShallowTopicTimes(epochs=epochs, X = dummy_X)
        
        elif hasattr(self.model, "get_predicted_topics"):
            print("\nModel type: CTM topic model {}\n".format(self.model))
            if test_input == None:
                dummy_X, _ = self.CreateDummyVars(dummy_len = dummy_len, vocab_size = vocab_size, seq_len = seq_len, out_type = 'str')
            else:
                dummy_X = test_input
            resdict = self.GetDeepTopicTimes(epochs=epochs, X=dummy_X)
        
        elif hasattr(self.model, "c_tf_idf_"):
            print("\nModel type: BerTopic topic model {}\n".format(self.model))
            if test_input == None:
                dummy_X, _ = self.CreateDummyVars(dummy_len = dummy_len, vocab_size = vocab_size, seq_len = seq_len, out_type = 'str')
            else:
                dummy_X = test_input
            resdict = self.GetBerTopicTimes(X=dummy_X)
            
        return resdict

