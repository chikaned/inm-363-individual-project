#import nltk libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#import DSA libs
import pandas as pd
import numpy as np
from numpy import absolute

#import octis libs
from octis.dataset.dataset import Dataset as OctisDataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence

#import visualisation libs
from lib_topic_tools import *
from lib_optuna_models import *
from lib_complexity_tools import *
from lib_generator_tools import *

#get stopwords
stop_words = set(stopwords.words('english'))

#################################################################################################################################

def GetOCtisXY(dataset_name = "20NewsGroup"):
    """

    Retrieves corpus (X) and labels (y) from octis dataset

    """
    
    oct_dataset = OctisDataset()
    oct_dataset.fetch_dataset(dataset_name)
    X = oct_dataset.get_corpus()
    X = [' '.join(x) for x in X]
    y = oct_dataset.get_labels()
    y = pd.Series(y)
    
    return (X, y)

def CreateDir(parent_dir='./octis_datasets/', child_dir='test'): 
    """

    Createsa a directory in a specified location

    """
    
    #get path
    path = os.path.join(parent_dir, child_dir+'/')
    
    #create dir if dir doesn't already exist
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    return path

#topic tools
class TopicProbs():
    """
    
    Class for getting document-topic predictions from trained topic models used in this study
    
    """


    def __init__(self, model = None, X = None):

        #init vars        
        self.model = model
        self.X = X

    def GetTradTMPred(self):
        """
        
        Get predicted document-topic probabilties for non-contextualized Gensim topic models (LDA & NMF) 
        
        """

    
        #create empty list
        topic_probs_list = []

        #get probabilities for each document
        for document in self.X:

            #get probs
            new_text_corpus =  self.model.id2word.doc2bow(document.split())
            topic_probs = self.model.trained_model[new_text_corpus]

            #create document prob matrix
            document_prob = np.zeros(self.model.trained_model.num_topics)

            #loop through to get topic probs and add to matrix
            for topic in topic_probs:
                topic_number, prob = topic
                document_prob[topic_number] = prob

            topic_probs_list.append(document_prob)

        return np.array(topic_probs_list)
    
    def GetBerTopicPred(self):
        """
        
        Get predicted document-topic probabilties for contextualized BerTopic topic models 
        
        """
        #get docs
        cleaned_docs = self.model._preprocess_text(self.X)
        topics, bert_probs = self.model.transform(cleaned_docs)  

        return bert_probs
    
    def GetZeroShotPred(self):
        """
        
        Get predicted document-topic probabilties for contextualized ZeroShotTM topic models 
        
        """    
        #prepare dataset
        tp = TopicModelDataPreparation("all-MiniLM-L6-v2")
        training_dataset = tp.fit(text_for_contextual=self.X, text_for_bow=self.X)

        #get output lists
        topic_document_matrix = self.model.get_doc_topic_distribution(training_dataset, n_samples = 5)# get all the topic predictions

        return topic_document_matrix
    
    def GetProbs(self):
        """
        
        Decorator function to the TopicProbs class
        
        """

        if "bertopic" in str(self.model):
            print("BerTopic model detected! Getting results...")
            probs = self.GetBerTopicPred()
            
        elif "ZeroShot" in str(self.model):
            print("ZeroShot model detected! Getting results...")
            probs = self.GetZeroShotPred()
            
        elif "LDA" in str(self.model) or "NMF" in str(self.model):
            print("Traditional model detected! Getting results...")
            probs = self.GetTradTMPred()
        else:
            return "Model invalid, please check!"
            
        return probs

#################################################################################################################################

class TopicScores():
    """ 
    
    Class to acquire evaluation scores for topic models used in this study
    
    """

    def __init__(self, model = None, X = None, topk=10):

        #init vars        
        self.model = model
        self.X = X
        self.topk = topk

    def GetTradTMScores(self):
        """
        
        Get evaluation scores for non-contextualized Gensim topic models (LDA & NMF) 
        
        """
        #create save dir
        root_dir = CreateDir(parent_dir='./res_dir/topic_modelling/', child_dir='scores')
        save_dir = CreateDir(parent_dir=root_dir, child_dir='CoCOVIDPlus_clean_tweets_dataset')

        #reformat data
        X = pd.Series(self.X)
        y = None

        #vocab
        octis_dataset = CustomOctisDataset(X=X, y=y, save_dir=save_dir)
        octis_dataset = octis_dataset.GenerateOctisDataset(train_split = 0.)

        #get uMass
        model_output = self.model.train_model(dataset = octis_dataset)                                
        uMass = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='u_mass')
        uMass_score = uMass.score(model_output)
        del(uMass)

        #get cnmpi                            
        c_npmi = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='c_npmi')
        c_npmi_score = c_npmi.score(model_output)
        del(c_npmi)

        #get diversity
        TopDiv = TopicDiversity(topk=self.topk)
        TopDiv_score = TopDiv.score(model_output)
        del(TopDiv)

        #get similarity
        InvertedRBO_scorer = InvertedRBO(weight=0.9, topk=self.topk)
        InvertedRBO_score = InvertedRBO_scorer.score(model_output)
        del(model_output)

        #get uMass
        model_scores = {'uMass': uMass_score, 'TopDiv':TopDiv_score, 'c_npmi':c_npmi_score, 'InvertedRBO':InvertedRBO_score}
        
        return model_scores
    
    def GetBerTopicScores(self):
        """
        
        Get evaluation scores for contextualized BerTopic topic models
        
        """
        
        #create save dir
        root_dir = CreateDir(parent_dir='./res_dir/topic_modelling/', child_dir='scores')
        save_dir = CreateDir(parent_dir=root_dir, child_dir='CoCOVIDPlus_full_tweets_dataset')

        #reformat data
        X = pd.Series(self.X)
        y = None

        #vocab
        octis_dataset = CustomOctisDataset(X=X, y=y, save_dir=save_dir)
        octis_dataset = octis_dataset.GenerateOctisDataset(train_split = 0.)

        #get docs
        cleaned_docs = self.model._preprocess_text(self.X)
        topics, bert_probs = self.model.transform(cleaned_docs) 
        if len(np.unique(topics)) < 2:
            return {'uMass':None, 'TopDiv':None, 'c_npmi':None, 'InvertedRBO':None} 

        #get topics ist
        topics_list = [[words for words, _ in self.model.get_topic(topic)] for topic in range(len(set(topics))-1)]

        #get topic term matrix
        topic_word_matrix = self.model.c_tf_idf_.toarray()

        #get model output
        model_output = {'topic-document-matrix': bert_probs, 'topic-word-matrix': topic_word_matrix, 'topics':topics_list}

        #get uMass                               
        uMass = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='u_mass')
        uMass_score = uMass.score(model_output)
        del(uMass)

        #get cnmpi                            
        c_npmi = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='c_npmi')
        c_npmi_score = c_npmi.score(model_output)
        del(c_npmi)

        #get TopicDiversity 
        TopDiv = TopicDiversity(topk=self.topk)
        TopDiv_score = TopDiv.score(model_output)
        del(TopDiv)
        
        #get similarity
        InvertedRBO_scorer = InvertedRBO(weight=0.9, topk=self.topk)
        InvertedRBO_score = InvertedRBO_scorer.score(model_output)
        del(model_output)

        #get uMass
        model_scores = {'uMass': uMass_score, 'TopDiv':TopDiv_score, 'c_npmi':c_npmi_score, 'InvertedRBO':InvertedRBO_score}

        return model_scores
    

    def GetZeroShotScores(self):
        """
        
        Get evaluation scores for contextualized ZeroShotTM topic models
        
        """

        #create save dir
        root_dir = CreateDir(parent_dir='./res_dir/topic_modelling/', child_dir='scores')
        save_dir = CreateDir(parent_dir=root_dir, child_dir='CoCOVIDPlus_full_tweets_dataset')

        #reformat data
        X = pd.Series(self.X)
        y = None

        #vocab
        octis_dataset = CustomOctisDataset(X=X, y=y, save_dir=save_dir)
        octis_dataset = octis_dataset.GenerateOctisDataset(train_split = 0.)

        #prepare dataset
        tp = TopicModelDataPreparation("all-MiniLM-L6-v2")
        training_dataset = tp.fit(text_for_contextual=self.X, text_for_bow=self.X)

        #get output lists
        topic_document_matrix = self.model.get_doc_topic_distribution(training_dataset, n_samples = 5)# get all the topic predictions
        topics_list = self.model.get_topic_lists()
        if len(topics_list) < 2:
            return {'uMass':None, 'TopDiv':None, 'c_npmi':None, 'InvertedRBO':None}
        topic_word_matrix = self.model.get_topic_word_matrix()

        #make model output dic
        model_output = {'topic-document-matrix': topic_document_matrix, 'topic-word-matrix': topic_word_matrix, 'topics':topics_list}

        #get uMass                            
        uMass = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='u_mass')
        uMass_score = uMass.score(model_output)
        del(uMass)

        #get cnmpi                            
        c_npmi = Coherence(texts=octis_dataset.get_corpus(), topk=self.topk, measure='c_npmi')
        c_npmi_score = c_npmi.score(model_output)
        del(c_npmi)

        #get diversity
        TopDiv = TopicDiversity(topk=self.topk)
        TopDiv_score = TopDiv.score(model_output)
        del(TopDiv)
      
        #get similarity
        InvertedRBO_scorer = InvertedRBO(weight=0.9, topk=self.topk)
        InvertedRBO_score = InvertedRBO_scorer.score(model_output)
        del(model_output)

        #get uMass
        model_scores = {'uMass': uMass_score, 'TopDiv':TopDiv_score, 'c_npmi':c_npmi_score, 'InvertedRBO':InvertedRBO_score}

        return model_scores

    def GetScores(self):
        """
        
        Decorator function for the TopicScores class
        
        """
        
        if "bertopic" in str(self.model):
            print("BerTopic model detected! Getting results...")
            scores = self.GetBerTopicScores()
            
        elif "ZeroShot" in str(self.model):
            print("ZeroShot model detected! Getting results...")
            scores = self.GetZeroShotScores()
            
        elif "LDA" in str(self.model) or "NMF" in str(self.model):
            print("Traditional model detected! Getting results...")
            scores = self.GetTradTMScores()
        else:
            return "Model invalid, please check!"
            
        return scores