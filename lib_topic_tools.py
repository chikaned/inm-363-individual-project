
#import gensim libraries
from types import TracebackType
from gensim.corpora import Dictionary
from gensim.models import LsiModel, LdaModel
from gensim.models.coherencemodel import CoherenceModel

#import nltk libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#import clustering functions
from sklearn.manifold import TSNE

#import visualisation libs
import plotly.express as px
from tqdm import tqdm

#import DSA libs
import pandas as pd
import numpy as np
import multiprocessing
import re
from numpy import absolute

#import octis libs
from octis.dataset.dataset import Dataset as OctisDataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform 
from octis.evaluation_metrics.similarity_metrics  import RBO, PairwiseJaccardSimilarity 
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

    def GetProbs(self):
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

#################################################################################################################################

def TopicModel(documents, max_topics=20):
    """
    takes a list of strings as an input for LSI modelling
    
    """
    
    #prep documents
    out = []
    for sent in documents:
        sent = word_tokenize(sent)
        sent = [x for x in sent if x not in stop_words]
        sent = [x for x in sent if x != 'fullstop']
        out.append(sent)
    documents = pd.Series(out)
    
    #create embedding and dictionary
    dictionary = Dictionary(documents)
    bow = [dictionary.doc2bow(doc) for doc in documents]

    #iterate through modelling
    coherence_dict = {}
    print("Modelling topics...")
    for i in tqdm(range(2, max_topics+2)):
        model = LsiModel(corpus = bow, num_topics = i, id2word=dictionary)
        coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_dict.update({coherence_score:i})

    #create plot
    y = coherence_dict.keys()
    x = coherence_dict.values()
    fig = px.bar(x =x , y = y)
    fig.update_xaxes(title = 'n_topics', type = 'category')
    fig.update_yaxes(title = 'coherence score')
    fig.show()

    #optimal model
    opt_n = coherence_dict[max(coherence_dict)]
    #model_out = LsiModel(bow, num_topics=opt_n, id2word=dictionary)
    model_out = LdaModel(bow, num_topics=opt_n, id2word=dictionary)
    
    return model_out, bow, dictionary, documents

def get_topic(bow, model):
    """
    
    Extracts topic terms from BoW model
    
    """

    res = model[bow]
    res_dict = dict((y, x) for x, y in res)
    topic = res_dict[max(res_dict)]
    return topic

def get_weights(bow, model):
    res = model[bow]
    weights = [y for x, y in res]
    return weights

def topic_count_plot(topic_labels:list, var_name = 'model topics'):
    """
    
    takes a list of values a creates a count plot
    
    """
    results_df = pd.DataFrame(pd.Series(topic_labels).value_counts())
    results_df.columns = ['count']
    fig = px.bar(results_df, x = results_df.index, y = 'count')
    fig.update_xaxes(type = 'category')
    fig.update_xaxes(title = var_name)
    fig.show()
    
def get_top_n_words(model, top_n = 3):
    """
    
    Get top-n words in topics
    
    """

    #get topic strings
    topics = model.show_topics()
    topic_strings= [x[1] for x in topics]
    
    #extract text
    pattern = r'[^A-Za-z_]+'
    X = [re.sub(pattern, ' ', x) for x in topic_strings]
    
    #retrieve top n tokens
    top_n_words = [' '.join(word_tokenize(x)[:top_n]) for x in X]
    
    return top_n_words

def get_mean_topic_vectors(topic_labels, tsne_trans):
    """
    
    Get mean vectors for topic terms
    
    """


    mean_vector_df = pd.DataFrame([topic_labels, tsne_trans]).T
    mean_vector_df.columns = ['topic_label', 'vectors']
    n_groups = mean_vector_df.topic_label.max()

    mean_topic_vectors = []
    for i in range(0, n_groups+1):
        vectors = mean_vector_df[mean_vector_df['topic_label'] == i]['vectors'].mean()
        mean_topic_vectors.append(vectors)
    return mean_topic_vectors

def tsne_plotter(topic_labels, topic_weights):
    """
    
    Create 2-D t-SNE representation
    
    """
    
    #array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    #dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    #tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_trans = tsne_model.fit_transform(arr)

    #specify color map
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])

    #dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    #get params
    mean_topic_vectors = get_mean_topic_vectors(topic_labels, tsne_trans)
    topic_words = get_top_n_words(tsne_model, top_n = 3)

    fig = px.scatter(x=tsne_trans[:,0], y=tsne_trans[:,1], color = colormap[topic_num])

    for coord, words in zip(mean_topic_vectors, topic_words):
        x = coord[0]
        y = coord[1]
        fig.add_annotation(
            x=x, y=y,
            text=words,
            showarrow=False,
            bgcolor = 'white',
            bordercolor = 'black',
            font = {'family':"Courier New", 'size':12},
            opacity = 0.85
        )

    fig.update_xaxes(title = 'component 1')
    fig.update_yaxes(title = 'component 2')
    fig.update_layout(showlegend=False) 
    fig.show()
