#import base libs
import gc
import glob
import os
import pickle as pkl
import numpy as np
import pandas as pd
import logging
import sys
from tqdm import tqdm

#import pre-processing tools
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#import traditional text classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#import topic models
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from gensim.models.coherencemodel import CoherenceModel

#import metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.similarity_metrics import WordEmbeddingsCentroidSimilarity

#import optuna
import optuna
from optuna.visualization import (plot_edf, plot_parallel_coordinate,
                                  plot_param_importances)

#custom_libs
from lib_neural_networks import *
from lib_topic_tools import *
from lib_optuna_models import *
from lib_complexity_tools import *
from lib_generator_tools import *
from lib_network_tools import *



class OptunaModels():
    """

    Wrapper for objective function in optuna studies
    
    """

    def __init__(self, run_name = None, random_state = 42):
        
        #init vars        
        self.run_name = run_name+'/' #folder name to save models form optuna runs
        self.random_state = random_state

    def SaveObject(self, save_object, object_name='NonSpecific_CLF', trial=None):

        """
        
        Saves study object to study save directory
        
        """
        
        #create dirpath
        if self.save_dir:
            dir_path = self.save_dir+self.run_name #example: './optuna/studies/test/'
        
            #check if dirpath exists 
            if os.path.exists(dir_path):
                pass
            else:
                os.mkdir(dir_path)
            
            #create full path
            if trial:
                full_path = dir_path+object_name+'_'+"{}.pkl".format(trial.number) #example:'./optuna/studies/test/NonSpecific_CLF_1.pkl'
            else:
                full_path = dir_path+object_name+".pkl" #example:'./optuna/studies/test/NonSpecific_CLF_1.pkl' where number is referenced explicitly in model name
                
            #save model
            with open(full_path, "wb") as fout:
                pkl.dump(save_object, fout)
                
            return print("Saved {} object to: ".format(object_name), full_path)
        else:
            return None
        
    def LoadObject(self, object_name='classifier', study=False):

        """
        
        Load object saved in Optuna trial
        
        """

        #create dirpath
        if self.save_dir:
            dir_path = self.save_dir+self.run_name #example: './optuna/studies/test/'

            #load saved model using trial number
            if study:
                with open(dir_path+object_name+'_'+"{}.pkl".format(study.best_trial.number), "rb") as fin:
                    load_object = pkl.load(fin)

            #load object without reference to a particular trial
            else:
                with open(dir_path+object_name+'.pkl', "rb") as fin:
                    load_object = pkl.load(fin)

            return load_object
        else:
            return None

    def SavePlotly(self, img_object=None, img_name = 'test'):
        """
        
        Save plotly images from hyperparamter optimzation studies
        
        """

        dir_path = self.save_dir+self.run_name

        #check if dirpath exists 
        if os.path.exists(dir_path):
            pass
        else:
            os.mkdir(dir_path)

        #create path
        img_path = dir_path+img_name+'.png'

        #write file
        img_object.write_image(img_path)

        return "Image saved to: "+img_path

    def RemoveFiles(self, dest_dir = None, rem_pattern='trial_*.pkl'):

        """
        
        Remove files from sub-optimal trials in optuna directory        

        """
        
        #get glob pattern
        glob_pattern = dest_dir+rem_pattern
        
        #get filelist
        fileList = glob.glob(glob_pattern, recursive=True)
        
        #remove files
        for file in fileList:

            try:
                #print("Removing {}".format(file))
                os.remove(file)

            except OSError:

                print("Error while deleting file")

        return "Removed matched files!"

    def SVM_Objective(self, trial):
        """
        
        Objective function for L-SVM text classification models

        """

        if trial.number == 100:
            self.study.stop()

        #scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
     
        #combine input if topic model is present
        if self.z is not None:
            z_scaled = scaler.fit_transform(self.z)
            X_scaled= np.concatenate((self.X, z_scaled), axis=1)

        #split data
        X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, self.y, train_size=self.train_split, random_state=self.random_state)

        #trial params
        kernel = 'linear' 
        degree = 3
        gamma = 'auto'

        #non-dependent hyperparameter        
        C = trial.suggest_float('C', 0.0001, 1.0, log = True)    
        shrinking=  False
        probability= False
        tol= trial.suggest_float('tol', 0.0001, 1.0, log = True)
        cache_size= 500
        class_weight= None
        verbose= False
        max_iter= -1
        
        #assign classifier
        clf = SVC(  
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                shrinking=shrinking,
                probability=probability,
                tol=tol,
                cache_size=cache_size,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
                random_state=self.random_state
                )

        #train classifier
        clf.fit(X_train, y_train)
        pred_labels = clf.predict(X_valid)
        objective_score = f1_score(y_true=y_valid, y_pred=pred_labels, average='weighted')

        #save model 
        self.SaveObject(save_object=clf, object_name='trial_SVM', trial=trial)
        self.SaveObject(save_object=X_valid, object_name='trial_X_valid', trial=trial)
        self.SaveObject(save_object=y_valid, object_name='trial_y_valid', trial=trial)
        
        return objective_score

    def LogR_Objective(self, trial):
        """
        
        Objective function for Logistic Regression text classification models

        """

        if trial.number == 100:
            self.study.stop()

        #scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
     
        #combine input if topic model is present
        if self.z is not None:
            z_scaled = scaler.fit_transform(self.z)
            X_scaled= np.concatenate((self.X, z_scaled), axis=1)

        #split data
        X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, self.y, train_size=self.train_split, random_state=self.random_state)

        #trial params
        penalty= trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']) #removed none
        tol= trial.suggest_float('tol', 0.0001, 1, log = True)
        C= trial.suggest_float('C', 0.0001, 1000, log = True)
        fit_intercept= trial.suggest_categorical('fit_intercept', [True, False])
        intercept_scaling= trial.suggest_float('intercept_scaling', 0.00001, 1.0, log = True)
        class_weight=None
        solver = 'saga'

        #static hyperparams    
        max_iter = 100
        multi_class= 'auto'
        verbose=0
        warm_start= False
        
        #assign regularization penalty
        if penalty == 'elasticnet':
            l1_ratio= trial.suggest_float('l1_ratio', 0.0, 1.0, log = False)
        else:
            l1_ratio = None

        #create classifier
        clf = LogisticRegression(
                            penalty = penalty,
                            tol = tol,
                            C = C,
                            fit_intercept = fit_intercept,
                            intercept_scaling= intercept_scaling,
                            class_weight= class_weight,
                            solver= solver,
                            max_iter= max_iter,
                            multi_class= multi_class,
                            verbose= verbose,
                            warm_start= warm_start,
                            n_jobs=self.n_jobs,
                            l1_ratio=l1_ratio,
                            random_state= self.random_state,
                            )

        #train classifier
        clf.fit(X_train, y_train)
        pred_labels = clf.predict(X_valid)
        objective_score = f1_score(y_true=y_valid, y_pred=pred_labels, average='weighted')
        
        #save model 
        self.SaveObject(save_object=clf, object_name='trial_LogR', trial=trial)
        self.SaveObject(save_object=X_valid, object_name='trial_X_valid', trial=trial)
        self.SaveObject(save_object=y_valid, object_name='trial_y_valid', trial=trial)
        
        return objective_score

    def NaiveB_Objective(self, trial):
        """
        
        Objective function for Naive Bayes text classification models

        """

        if trial.number == 100:
            self.study.stop()

        #scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
     
        #combine input if topic model is present
        if self.z is not None:
            z_scaled = scaler.fit_transform(self.z)
            X_scaled= np.concatenate((self.X, z_scaled), axis=1)

        #split data
        X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, self.y, train_size=self.train_split, random_state=self.random_state)

        #trial parameters
        alpha = trial.suggest_float('alpha', 0.0001, 1.0, log = True)
        fit_prior = trial.suggest_categorical('fit_prior', [True, False])
        class_prior = None

        #create classifier
        clf = MultinomialNB(
                            alpha = alpha,
                            fit_prior = fit_prior,
                            class_prior = class_prior,
                            )

        #train classifier
        clf.fit(X_train, y_train)
        pred_labels = clf.predict(X_valid)
        objective_score = f1_score(y_true=y_valid, y_pred=pred_labels, average='weighted')
        
        #save model 
        self.SaveObject(save_object=clf, object_name='trial_NaiveB', trial=trial)
        self.SaveObject(save_object=X_valid, object_name='trial_X_valid', trial=trial)
        self.SaveObject(save_object=y_valid, object_name='trial_y_valid', trial=trial)
        
        return objective_score

    def RanF_Objective(self, trial):
        """
        
        Objective function for Random Forest text classification models

        """

        if trial.number == 100:
            self.study.stop()

        #scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
     
        #combine input if topic model is present
        if self.z is not None:
            z_scaled = scaler.fit_transform(self.z)
            X_scaled= np.concatenate((self.X, z_scaled), axis=1)

        #split data
        X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, self.y, train_size=self.train_split, random_state=self.random_state)

        #trial parameters
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        min_weight_fraction_leaf = 0.0
        max_features = trial.suggest_categorical("max_feat", ["sqrt", "log2", None])
        max_leaf_nodes=None
        min_impurity_decrease = 0.0
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        n_estimators = trial.suggest_int('n_estimators', 10, 500)

        #assign bootstrap params
        if bootstrap == True:
            oob_score = trial.suggest_categorical("oob_score", [True, False])
        else:
            oob_score = False
        
        #assign other params
        verbose = 0
        warm_start = False
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
        ccp_alpha = 0.0
        max_samples = None

        #create classifier
        clf = RandomForestClassifier(
                                    n_estimators = n_estimators,
                                    criterion = criterion,
                                    max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    min_samples_leaf = min_samples_leaf,
                                    min_weight_fraction_leaf = min_weight_fraction_leaf,
                                    max_features= max_features,
                                    max_leaf_nodes = max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease,
                                    bootstrap= bootstrap,
                                    oob_score=oob_score,
                                    n_jobs=self.n_jobs,
                                    verbose=verbose,
                                    warm_start= warm_start,
                                    class_weight=class_weight,
                                    ccp_alpha= ccp_alpha,
                                    max_samples= max_samples,
                                    random_state=self.random_state,
                                    )

        #train classifier
        clf.fit(X_train, y_train)
        pred_labels = clf.predict(X_valid)
        objective_score = f1_score(y_true=y_valid, y_pred=pred_labels, average='weighted')

        #save model 
        self.SaveObject(save_object=clf, object_name='trial_RanF', trial=trial)
        self.SaveObject(save_object=X_valid, object_name='trial_X_valid', trial=trial)
        self.SaveObject(save_object=y_valid, object_name='trial_y_valid', trial=trial)
        
        return objective_score

    def AdaB_Objective(self, trial):
        """
        
        Objective function for Adaptive Boosting text classification models

        """

        if trial.number == 100:
            self.study.stop()
        
        #scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)
     
        #combine input if topic model is present
        if self.z is not None:
            z_scaled = scaler.fit_transform(self.z)
            X_scaled= np.concatenate((self.X, z_scaled), axis=1)

        #split data
        X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, self.y, train_size=self.train_split, random_state=self.random_state)


        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 1.0, log = True)
        
        #create classifier
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=self.random_state)

        #train classifier
        clf.fit(X_train, y_train)
        pred_labels = clf.predict(X_valid)
        objective_score = f1_score(y_true=y_valid, y_pred=pred_labels, average='weighted')

        #save model 
        self.SaveObject(save_object=clf, object_name='trial_AdaB', trial=trial)
        self.SaveObject(save_object=X_valid, object_name='trial_X_valid', trial=trial)
        self.SaveObject(save_object=y_valid, object_name='trial_y_valid', trial=trial)
        
        return objective_score

    def BerTopic_Objective(self, trial):
        """
        
        Objective function for BerTopic text topic models

        """

        if trial.number == 100:
            self.study.stop()

        #set hyperparameter search space
        if self.num_topics is not None:
            num_topics = self.num_topics
        else:
            num_topics = trial.suggest_int('num_topics', 5, 20)

        #SBERT hyperparameters
        n_gram_range = (1, 1)

        #UMAP hyperparams
        n_neighbours = trial.suggest_int('n_neighbors', 2, 100)
        n_components = trial.suggest_int('n_components', 2, 100)
        min_dist = trial.suggest_float('min_dist', 0.01, 1.0, log=False)

        #HDBSCAN hyperparams
        min_samples = trial.suggest_int('min_samples', 2,100)
        metric = trial.suggest_categorical('metric', ['euclidean','manhattan'])
        
        #static variables
        embedding_model = "all-MiniLM-L6-v2"
        calculate_probabilities = True
        low_memory=False
        top_n_words = self.topk
        min_topic_size =self.topk
        verbose = False

        #get tokenized texts
        docs = self.X

        
        umap_model = UMAP(n_neighbors=n_neighbours, n_components=n_components, 
                          min_dist=min_dist, metric='cosine', random_state=42)

        hdbscan_model = HDBSCAN(min_cluster_size=self.topk, 
                                metric=metric, 
                                min_samples=min_samples,
                                prediction_data=True,
                                cluster_selection_method="leaf")

        #get topic models default embedding "all-MiniLM-L6-v2" 
        model = BERTopic(
                         language="english", 
                         low_memory=low_memory,
                         calculate_probabilities = calculate_probabilities,
                         min_topic_size = min_topic_size, 
                         verbose = verbose,
                         nr_topics = num_topics,
                         umap_model=umap_model,
                         hdbscan_model=hdbscan_model,
                         n_gram_range = n_gram_range,
                         top_n_words = top_n_words,
                         embedding_model=embedding_model
                         )

        #get preds
        cleaned_docs = model._preprocess_text(docs)
        topics, topics_probs = model.fit_transform(cleaned_docs) 
        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()
        tokens_list = [analyzer(doc) for doc in cleaned_docs]
        dictionary = Dictionary(tokens_list)
        corpus = [dictionary.doc2bow(token) for token in tokens_list]
        topics_list = [[words for words, _ in model.get_topic(topic)] for topic in range(len(set(topics))-1)]
        model_output = {'topic-document-matrix': topics_probs, 'topics':topics_list}

        #get classification scores
        if len(topics_list[0]) < 2:
            objective_score = 0
        elif self.topic_metric == 'u_mass':
            Coh = CoherenceModel(topics = topics_list, texts=tokens_list, corpus = corpus, dictionary = dictionary, coherence="u_mass")
            objective_score = Coh.get_coherence()
            del(Coh)
        elif self.topic_metric == 'wecs':
            objective_score = self.WECS.score(model_output)
        #save model
        self.SaveObject(save_object=model, object_name='trial_BerTopic', trial=trial)

        #save study
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+self.model_name, trial=None)

        del(topics, cleaned_docs, topics_probs, vectorizer, analyzer, tokens_list, dictionary, corpus, topics_list, model_output, model)

        return objective_score

    def ZeroProdLDA_Objective(self, trial):
        """
        
        Objective function for ZeroShotTM text topic models

        """

        if trial.number == 100:
            self.study.stop()
        
        #get tokenized texts
        docs = self.X
        tokens_list = TokenizeTexts(self.X)

        #prep data 
        tp = TopicModelDataPreparation("all-MiniLM-L6-v2")
        training_dataset = tp.fit(text_for_contextual=docs, text_for_bow=docs)

        #set hyperparameter search space
        if self.num_topics is not None:
            num_topics = self.num_topics
        else:
            num_topics = trial.suggest_int('num_topics', 5, 20)
        dropout = trial.suggest_float('dropout', 0., 0.5, log = False)

        lr = trial.suggest_float('lr', 1e-4, 0.1, log = True) #0.00001, 0.001, log=False
        hidden_1 = trial.suggest_int('hidden_1', 50, 250)
        hidden_2 = trial.suggest_int('hidden_2', 50, 250)
        hidden_sizes = (hidden_1, hidden_2)
        
        #other variables 
        num_epochs = self.epochs #100 was used in paper for longer texts
        batch_size = 32
        activation = 'softplus' #'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu'
        optimizer_type = 'adam' #sgd

        #run model
        model = ZeroShotTM(
                    bow_size=len(tp.vocab), 
                    contextual_size=384, #bert size
                    n_components=num_topics, 
                    dropout = dropout,
                    batch_size = batch_size,
                    lr = lr,
                    hidden_sizes=hidden_sizes,
                    num_epochs=num_epochs,
                    activation=activation,
                    solver=optimizer_type)
        model.fit(training_dataset, n_samples = 5)
        
        #get output vocab
        topic_document_matrix = model.get_doc_topic_distribution(training_dataset, n_samples = 5)# get all the topic predictions
        topics_list = model.get_topic_lists()
        model_output = {'topic-document-matrix': topic_document_matrix,'topics':topics_list}

        #get classification scores
        if self.topic_metric == 'u_mass':
            dct = Dictionary(tokens_list) 
            Coh = CoherenceModel(topics = topics_list, texts=tokens_list, dictionary = dct, coherence=self.topic_metric)
            objective_score = Coh.get_coherence()
            del(dct, Coh, tokens_list)
        elif self.topic_metric == 'wecs':
            objective_score =self.WECS.score(model_output)

        #save model
        self.SaveObject(save_object=model, object_name='trial_ZeroProdLDA', trial=trial)
        del(model, model_output)

        #save study
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+self.model_name, trial=None)

        #get score
        return objective_score

    def CTMProdLDA_Objective(self, trial):
        """
        
        Objective function for CombinedTM text topic models

        """

        if trial.number == 100:
            self.study.stop()
        
        #get tokenized texts
        docs = self.X
        tokens_list = TokenizeTexts(self.X)

        #prep data 
        tp = TopicModelDataPreparation("all-MiniLM-L6-v2")
        training_dataset = tp.fit(text_for_contextual=docs, text_for_bow=docs) #include labels for CTM

        #set hyperparameter search space
        if self.num_topics is not None:
            num_topics = self.num_topics
        else:
            num_topics = trial.suggest_int('num_topics', 5, 20)
        dropout = trial.suggest_float('dropout', 0., 0.5, log = False)
        batch_size = 32
        lr = trial.suggest_float('lr', 1e-3, 0.1, log = True)
        hidden_1 = trial.suggest_int('hidden_1', 50, 250)
        hidden_2 = trial.suggest_int('hidden_2', 50, 250)
        hidden_sizes = (hidden_1, hidden_2)
        
        #other variables 
        num_epochs = self.epochs 
        activation = 'softplus' #'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu' #softplus used in ProdLDA paper and CTM
        optimizer_type = 'adam' #used in CTM paper

        #run model
        model = CombinedTM(
                        bow_size=len(tp.vocab), 
                        contextual_size=384, #pretrained size https://www.sbert.net/docs/pretrained_models.html
                        n_components=num_topics, 
                        dropout = dropout,
                        batch_size = batch_size,
                        lr = lr,
                        hidden_sizes=hidden_sizes,
                        num_epochs=num_epochs,
                        activation=activation,
                        solver=optimizer_type)
        model.fit(training_dataset, n_samples=5)
        
        #get output vocab
        topic_document_matrix = model.get_doc_topic_distribution(training_dataset, n_samples = 5)# get all the topic predictions
        topics_list = model.get_topic_lists()
        model_output = {'topic-document-matrix': topic_document_matrix,'topics':topics_list}

        #get classification scores
        if self.topic_metric == 'u_mass':
            dct = Dictionary(tokens_list) 
            Coh = CoherenceModel(topics = topics_list, texts=tokens_list, dictionary = dct, coherence=self.topic_metric)
            objective_score = Coh.get_coherence()
            del(dct, Coh, tokens_list)
        else:
            objective_score =self.WECS.score(model_output)

        #save model
        self.SaveObject(save_object=model, object_name='trial_CTMProdLDA', trial=trial)
        del(model, model_output)

        #save study
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+self.model_name, trial=None)

        #get score
        return objective_score

    def OctLDA_Objective(self, trial):
        """
        
        Objective function for Octis LDA text topic models

        """

        if trial.number == 100:
            self.study.stop()
        
        #set hyperparameter search space
        if self.num_topics is not None:
            num_topics = self.num_topics
        else:
            num_topics = trial.suggest_int('num_topics', 5, 20)
        decay = trial.suggest_float('decay', 0.5, 1.0, log = False)
        alpha = trial.suggest_float('alpha', 0.0001, 0.1, log = True)
        gamma_threshold = trial.suggest_float('gamma_threshold', 0.0001, 0.1, log = True)
        passes = 20

        #run model
        model = LDA(num_topics=num_topics, alpha = alpha, gamma_threshold=gamma_threshold, decay=decay, passes=passes, random_state = self.random_state)
        
        #get model output and vocab
        model_output = model.train_model(self.octis_dataset)
        model_vocab = model.id2word.id2token.values()
        model_vocab = list(model_vocab)


        if self.topic_metric == 'u_mass':
            measure = 'u_mass' #higher = more coherence
            Coh = Coherence(texts=self.octis_dataset.get_corpus(), topk=self.topk, measure=measure)
            objective_score = Coh.score(model_output)
        else:
            objective_score =self.WECS.score(model_output)

        #save model
        self.SaveObject(save_object=model, object_name='trial_OctLDA', trial=trial)
        
        #get score
        return objective_score

    def OctNMF_Objective(self, trial):
        """
        
        Objective function for Octis NMF text topic models

        """
        
        if trial.number == 100:
            self.study.stop()

        #set hyperparameter search space
        if self.num_topics is not None:
            num_topics = self.num_topics
        else:
            num_topics = trial.suggest_int('num_topics', 5, 20)
        kappa=trial.suggest_float('kappa', 0.1, 1.0, log = True)
        minimum_probability=trial.suggest_float('minimum_probability', 0.000001, 0.1, log = True)

        #other variables
        chunksize=2000
        passes=20
        w_max_iter=trial.suggest_int('w_max_iter', 100, 500)
        w_stop_condition=0.0001
        h_max_iter=trial.suggest_int('h_max_iter', 25, 100)
        h_stop_condition=0.001
        eval_every=10
        normalize=True
        use_partitions=True

        #run model
        model = NMF(num_topics=num_topics,
                    eval_every = eval_every,
                    kappa=kappa,
                    minimum_probability=minimum_probability,
                    chunksize=chunksize,
                    passes=passes,
                    w_max_iter=w_max_iter,
                    w_stop_condition=w_stop_condition,
                    h_max_iter=h_max_iter,
                    h_stop_condition=h_stop_condition,
                    normalize=normalize,
                    random_state=self.random_state,
                    use_partitions=use_partitions)
        
        #get model output
        model_output = model.train_model(self.octis_dataset)

        #get model output and vocab
        model_output = model.train_model(self.octis_dataset)
        model_vocab = model.id2word.id2token.values()
        model_vocab = list(model_vocab)
         
        #get classification scores
        if self.topic_metric == 'u_mass':
            Coh = Coherence(texts=self.octis_dataset.get_corpus(), topk=self.topk, measure=self.topic_metric)
            objective_score = Coh.score(model_output)
        else:
            objective_score =self.WECS.score(model_output)


        #save model
        self.SaveObject(save_object=model, object_name='trial_OctNMF', trial=trial)
        
        #get score
        return objective_score

    def GRU_Objective(self, trial):
        """
        
        Objective function for GRU text classifiers

        """
        

        if trial.number == 100:
            self.study.stop()
    
        #clear cache
        FreeGpuCache()

        #create BachGen object
        BatchGen = BatchGenerator(X=self.X, y=self.y, z=self.z)

        #get iterators
        train_iter, valid_iter, word2idx = BatchGen.MixedIterator(batch_size = self.batch_size, split = self.train_split, shuffle=True)

        #inferred parameters
        vocab_size= len(word2idx)
        output_dim = len(np.unique(self.y.values))
        pad_token= 1
        unk_token= 0
        batch_first= True
        device= self.device
        epochs = self.epochs

        #get topic dim
        if self.z is not None:
            topic_dim = self.z.shape[1] 
        else:
            topic_dim = 0   

        #get pretrained vectors
        weights, embedding_dim = None, 100

        #model hyperparameters
        hidden_dim=trial.suggest_int('hidden_dim', 50, 300)
        bidirectional=False
        num_layers= 1
        
        #add dropout if num_layers > 1
        if num_layers > 1:
            rnn_layer_dropout=trial.suggest_float('rnn_layer_dropout', 0., 0.5, log=False)
        else:
            rnn_layer_dropout = 0.
            
        #add fc dropout
        rnn_out_dropout=trial.suggest_float('rnn_out_dropout', 0., 0.5, log=False)

        #lr
        lr =trial.suggest_float('lr', 0.00001, 0.001, log=False)
        clip_value = None #not required for adam

        #create model
        network = GRU(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    rnn_layer_dropout=rnn_layer_dropout,
                    rnn_out_dropout=rnn_out_dropout,
                    output_dim=output_dim,
                    batch_first=batch_first,
                    weights=weights,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    device=device,
                    topic_dim=topic_dim
                    )

        # assign optimizer and loss-fn                                   
        loss_fn = F.cross_entropy #loss_fn = F.mse_loss # for regression     
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr = lr)

        NetMod = NetModeller(
                     network=network,
                     train_iter=train_iter,
                     valid_iter=valid_iter,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     epochs=epochs,
                     clip_value=clip_value,
                     device=device
                     )

        #start training loop
        for i in range(epochs):
            #train_loss, train_obj, optimizer, valid_loss, valid_obj
            train_loss, train_obj, _, valid_loss, valid_obj = NetMod.SingleNetworkWrapper()

            #report for pruning
            trial.report(valid_obj, i)

            if trial.should_prune():
                raise optuna.TrialPruned()

      
        #save model
        self.SaveObject(save_object=network, object_name='trial_GRU', trial=trial)

        #save study
        self.SaveObject(save_object=self.study, object_name='trial_adam_optimizer_'+self.model_name, trial=trial)

        #save valid data 
        self.SaveObject(save_object=valid_iter, object_name='trial_valid_iter', trial=trial)
    
        #save study
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+self.model_name, trial=None)
        
        #delete objects
        del(train_iter, valid_iter, word2idx, BatchGen, network, optimizer, loss_fn)

        #clear cache
        FreeGpuCache()

        return valid_obj

    def EmbedCNN_Objective(self, trial):
        """
        
        Objective function for CNN text classifiers

        """

        if trial.number == 100:
            self.study.stop()

        #create BachGen object
        BatchGen = BatchGenerator(X=self.X, y=self.y, z=self.z)

        #create dataset
        train_iter, valid_iter, word2idx = BatchGen.MixedIterator(batch_size = self.batch_size, split = self.train_split)

        #inferred parameters
        vocab_size= len(word2idx)
        output_dim = len(np.unique(self.y.values))
        pad_token= 1
        unk_token= 0
        batch_first= True
        device= self.device
        epochs = self.epochs

        #get topic dim
        if self.z is not None:
            topic_dim = self.z.shape[1] 
        else:
            topic_dim = 0   

        #get weights
        weights, embedding_dim = None, 100

        #model hyperparameters
        filter_sizes = trial.suggest_categorical('filter_sizes', [(2, 3, 4), (2, 4, 6), (3, 4, 5), (3, 5, 7)])
        num_filter = trial.suggest_int('num_filter', 20, embedding_dim)
        num_filters = list([num_filter])*len(filter_sizes)
        freeze_embedding = False
        
        #add dropout
        cnn_out_dropout=trial.suggest_float('cnn_out_dropout', 0., 0.5, log=False)

        #lr
        lr =trial.suggest_float('lr', 0.0001, 0.01, log=False)
        clip_value = None #not required for adam

        #create model
        network = EmbedCNN(
                            weights=weights, 
                            freeze_embedding=freeze_embedding,
                            vocab_size=vocab_size, 
                            embedding_dim=embedding_dim, 
                            filter_sizes=filter_sizes,
                            num_filters=num_filters, 
                            cnn_out_dropout=cnn_out_dropout,
                            output_dim=output_dim, 
                            unk_token=unk_token, 
                            pad_token =pad_token, 
                            device=device,
                            topic_dim=topic_dim
                            )


        # assign optimizer and loss-fn
        # loss_fn = F.mse_loss # for regression                                                
        loss_fn = F.cross_entropy #for classifiction
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr = lr)

        NetMod = NetModeller(
                     network=network,
                     train_iter=train_iter,
                     valid_iter=valid_iter,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     epochs=epochs,
                     clip_value=clip_value,
                     device=device
                     )

        #start training loop
        for i in range(epochs):
            #train_loss, train_obj, optimizer, valid_loss, valid_obj
            train_loss, train_obj, _, valid_loss, valid_obj = NetMod.SingleNetworkWrapper()    

            #report for pruning
            trial.report(valid_obj, i)

            if trial.should_prune():
                raise optuna.TrialPruned()


        #save model
        self.SaveObject(save_object=network, object_name='trial_EmbedCNN', trial=trial)

        #save study
        self.SaveObject(save_object=self.study, object_name='trial_adam_optimizer_'+self.model_name, trial=trial)

        #save valid data 
        self.SaveObject(save_object=valid_iter, object_name='trial_valid_iter', trial=trial)

        #save study
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+self.model_name, trial=None)

        del(train_iter, valid_iter, word2idx, BatchGen, network, optimizer, loss_fn)

        #clear cache
        FreeGpuCache()

        return valid_obj

    def ReturnObjectiveFn(self, model_name):

        """
        
        Decorator function for objective functions.
        
        """

        #shallow models
        if model_name == 'LogR':
            return self.LogR_Objective
            
        if model_name == 'SVM':
            return self.SVM_Objective
            
        if model_name == 'NaiveB':
            return self.NaiveB_Objective
            
        if model_name == 'RanF':
            return self.RanF_Objective 

        if model_name == 'AdaB':
            return self.AdaB_Objective 

        #RNNs
        if model_name == 'GRU':
            return self.GRU_Objective
        
        if model_name =='EmbedCNN':
            return self.EmbedCNN_Objective
        
        if model_name == 'CNN_GRU':
            return self.CNN_GRU_Objective

        #topic Modellers
        if model_name == 'BerTopic':
            return self.BerTopic_Objective

        if model_name == 'ZeroProdLDA':
            return self.ZeroProdLDA_Objective
        
        if model_name == 'CTMProdLDA':
            return self.CTMProdLDA_Objective

        if model_name == 'OctLDA':
            return self.OctLDA_Objective
        
        if model_name == 'OctNMF':
            return self.OctNMF_Objective

    def Optimize(
                self, 
                model_name = None,
                X=None,
                y=None,
                z=None, #topics
                encoder = LabelEncoder(), 
                octis_dataset= None, #topic model
                train_split = 0.85,
                epochs = 25, #NN
                batch_size = 32,
                device = 'cuda',
                n_trials = 10,
                n_jobs = 1,
                topk = 10, #topic model
                num_topics = None,
                topic_metric = 'u_mass',
                model_output = None,
                opt_direction= 'maximize',
                save_dir = './optuna/studies/test/',
                save_plots = True,
                storage_name = 'test_XX'
                ):
        
        """
        
        Runs hyperparamter optimization and returns the best model
        
        """
        
        #data prep     
        self.X = X
        self.y = y
        self.z = z
        if self.z is not None:
            print("Z found with shape  {}: ".format(z.shape))
        self.train_split = train_split
        self.octis_dataset = octis_dataset
        self.encoder = encoder
        self.batch_size = batch_size
        self.model_name = model_name

        #modelling varss
        self.epochs = epochs
        self.device = device
        self.n_jobs = n_jobs

        #topic modelling
        self.num_topics = num_topics
        self.topk = topk
        self.topic_metric = topic_metric
        self.model_output = model_output

        #directory to save results
        self.save_dir = save_dir
        
        #get objective function
        objective_fn = self.ReturnObjectiveFn(model_name = self.model_name)
        
        #run trial
        import optuna
        #create storage and study name 
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = self.save_dir+self.run_name+'optuna_study_'+model_name
        study_name = study_name.replace("/","_")
        storage_url = "sqlite:///{}.db".format(storage_name)
        storage = optuna.storages.RDBStorage(url=storage_url)  #heartbeat_interval=60, grace_period=120)
        
        #turn sqllite storage off
        load_if_exists=False
        storage=None

        #check for pickle study
        saved_study_path = self.save_dir+self.run_name+'optuna_study_'+model_name+'.pkl'
        if os.path.exists(saved_study_path):
            self.study = self.LoadObject(object_name='optuna_study_'+model_name, study = False)
            print("Continuing study from: {}".format(saved_study_path))
        else:
            self.study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=load_if_exists, direction=opt_direction)
        self.study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True, n_jobs = self.n_jobs, gc_after_trial=True) #7 is max -1
        self.SaveObject(save_object=self.study, object_name='optuna_study_'+model_name, trial=None)

        #retrieve and save best model
        best_clf = self.LoadObject(object_name='trial_'+model_name, study = self.study)
        self.SaveObject(save_object=best_clf, object_name='optimal_'+model_name, trial=None)
        print("Loaded best {} object: {}".format(model_name, self.study.best_trial.number))

        #retrieve and save best model validation data
        if hasattr(best_clf, 'predict'): #for shallow learning models
            X_valid = self.LoadObject(object_name='trial_'+'X_valid', study = self.study)
            self.SaveObject(save_object=X_valid, object_name='optimal_'+'X_valid', trial=None)
            y_valid = self.LoadObject(object_name='trial_'+'y_valid', study = self.study)
            self.SaveObject(save_object=y_valid, object_name='optimal_'+'y_valid', trial=None)

        elif hasattr(best_clf, 'state_dict'): #for deep learning models
            X_valid = self.LoadObject(object_name='trial_'+'valid_iter', study = self.study)
            self.SaveObject(save_object=X_valid, object_name='optimal_'+'valid_iter', trial=None)
        
        else: #for topic models
            pass 

        #delete intermediate files
        if self.save_dir:
            self.RemoveFiles(dest_dir=self.save_dir+self.run_name, rem_pattern='trial_*.pkl')

        #save plots
        if save_plots == True:
            par_plot = plot_parallel_coordinate(self.study)
            self.SavePlotly(par_plot, 'parallel_plot_'+model_name)
            imp_plot = plot_param_importances(self.study)
            self.SavePlotly(imp_plot, 'importance_plot_'+model_name)
            edf_plot = plot_edf(self.study)
            self.SavePlotly(edf_plot, 'edf_plot_'+model_name)

        #return model study (and training docs for text classsifiers)
        if hasattr(best_clf, 'state_dict'): #for deep learning models
            return best_clf, X_valid, self.study 
        elif hasattr(best_clf, 'predict'): #for traditional learning models
            return best_clf, X_valid, y_valid, self.study 
        else: #for topic models
            return best_clf, self.study 
