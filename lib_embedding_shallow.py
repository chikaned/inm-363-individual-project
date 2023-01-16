#import relevant libs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

"""
Library for TF and TF-IDF feature extraction methods used in text classification studies

**********************************************************************************************************************************************

"""


def TFIDF(X, norm = 'l1', max_features = None, use_idf=True, analyzer='word'):
    """
    returns training and validation vectors for tokenized list of lists
    norm = type of regularization (l1 or l2)
    standard params
    """
    
    vectorizer = TfidfVectorizer(norm = norm, max_features = max_features, 
                                 use_idf = use_idf, analyzer = analyzer) 
    
    vectors = vectorizer.fit_transform(X)
    vocab = vectorizer.vocabulary_
    
    return vectors, vocab

def CountVec(X, max_features = None):
    """
    returns training and validation vectors for tokenized list of lists
    standard params
    """
    
    vectorizer = CountVectorizer(max_features = max_features) 
    vectors = vectorizer.fit_transform(X)
    vocab = vectorizer.vocabulary_
    
    return vectors, vocab