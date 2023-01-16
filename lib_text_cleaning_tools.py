#import general libs
import pandas as pd
import nltk
from tqdm import tqdm
from IPython.display import clear_output
import re


#import text libs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import Word as tbword
from textblob import TextBlob
import spacy
import emoji

#import custom libs
from lib_text_tools_customdicts import preep


# Load the spacy model
nlp = spacy.load("en_core_web_sm") #en_core_web_trf #en_core_web_sm


def clean_tweet(tweet):
    """

    Removes undesriable elements from text often foudn in social media posts

    """
    if type(tweet) == float:
        return ""
    temp = emoji.demojize(tweet)
    temp = decontracted(temp)
    temp = re.sub(":", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+","", temp) #remove at mentions
    #temp = re.sub("#[A-Za-z0-9_]+","", temp) #remove hashtags
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = temp.split()
    temp = " ".join(word for word in temp)
    return temp


def hashtag_connector(X):
    """

    takes list of strings and joins hashtags

    """
    out = re.sub('# +', 'hashtag_', X)
    return out



def spellchecker(X, mode='list'):
    """
    
    spellchecks string or list of strings 
    str or list --> str or list

    """
    
    if mode == 'list':
        X = pd.Series([word_tokenize(x) for x in X])
        out = []
        for sentence in tqdm(X):
            tmp = []
            for word in sentence:
                word = word.lower()
                word = tbword(word)
                result = word.correct()
                tmp.append(result)
            out.append(tmp)
        X = out     
        X = [' '.join(x) for x in X]
        X = [x.strip() for x in X]
            
    if mode == 'single':
        X = word_tokenize(X)
        out = []
        for word in X:
            word = word.lower()
            word = tbword(word)
            result = word.correct()
            out.append(result)
        X = out
        X = ' '.join(X)
        X = X.strip()
    
    return X
    
    
def decontracted(X):
    """

    converts words with apostrophes into separate words
    
    """
    
    # specific
    X = re.sub(r"won't", "will not", X)
    X = re.sub(r"can\'t", "can not", X)

    # general
    X = re.sub(r"n\'t", " not", X)
    X = re.sub(r"\'re", " are", X)
    X = re.sub(r"\'s", " is", X)
    X = re.sub(r"\'d", " would", X)
    X = re.sub(r"\'ll", " will", X)
    X = re.sub(r"\'t", " not", X)
    X = re.sub(r"\'ve", " have", X)
    X = re.sub(r"\'m", " am", X)
    return X


def text_cleaner(X, tokenizer = word_tokenize, style = 'jm', chop = 'lem', stopwords_rem =True, spellcheck = True):
    """
    removes twitter elements, spellchecks, stems and removes stopwords from list of texts
    replaces # -> 'hashtag_ and . -> 'fullstop

    style: cleans text according to garg et al. or personalised method

    """
    ##CLEAR OUTPUT
    clear_output(wait=True)
    
    ##REMOVE TEXT ELEMENTS FROM TWEETS
    print("Removing twitter elements...")
    out = []
    for i, sentence in tqdm(enumerate(X)):
        #tmp = tpp.clean(sentence)
        if style == 'garg':
            tmp = result = re.sub(r'http\S+', '', sentence)
        else:
            tmp = clean_tweet(sentence)
        out.append(tmp)
    X = pd.Series(out) 
    #X = pd.Series([tpp.clean(x) for x in X])
    
    #TOKENIZE
    print("Tokenization...")
    out = []
    for i, sentence in tqdm(enumerate(X)):
        tmp = tokenizer(sentence)
        out.append(tmp)   
    X = pd.Series(out) 
    #X = pd.Series([tokenizer(x) for x in X])
    
    #SPELLCHECK
    if spellcheck == True:
        print("Spellchecking...")
        out = []
        for sentence in tqdm(X):

            #get proper nouns in sentence
            blob_object = TextBlob(' '.join(sentence))
            blob_tags = blob_object.tags
            pnouns = [x[0].lower() for x in blob_tags if x[1] in ['NNP', 'NNPS']]

            tmp = []
            for word in sentence: 
                word = word.lower()
                if word not in pnouns: #don't correct proper nouns
                    word = tbword(word)
                    word = word.correct()
                tmp.append(word)
            out.append(tmp)
        X = out

    #REMOVE STOPWORDS  

    if stopwords_rem:
        print("Removing stopwords...") 
        out = []
        for sentence in tqdm(X):
            sentence = ' '.join(sentence)
            sentence = nlp(sentence)
            tmp = [x.text for x in sentence if not x.is_stop]
            tmp = [x for x in tmp if x != 'amp']
            tmp = ' '.join(tmp)
            out.append(tmp)
        X = out
        
    #LEM WORDS
    if chop == 'lem':
        print("Lemmatizing...")
        out = []
        for i, sentence in tqdm(enumerate(X)):
            sentence = nlp(sentence)
            tmp = [x.lemma_ for x in sentence]
            out.append(tmp) 
        X = out
          
    #STEM WORDS
    if chop == 'stem':
        print("Stemming...")
        ps = PorterStemmer()
        out = []
        for i, sentence in tqdm(enumerate(X)):
            tmp = [ps.stem(x) for x in sentence]
            out.append(tmp) 
        X = out
    
    else:
        pass
    
    #FULL STOP REPLACER
    X = [' '.join(x) for x in X]
    X = [x.replace('.', '') for x in X] #amend to replace fullstop with meaningful value for substring analysis
           
    #REMOVE CHARACTERS
    pattern = r'[^A-Za-z #]+' #remove punctuation, but include '_' for hashtags and remove numbers
    print("Removing characters from string except:", pattern)
    X = [re.sub(pattern, '', x) for x in X]

    #HASHTAG CONNECTOR
    X = [hashtag_connector(x) for x in X]
       
    #REMOVE LEADING AND TRAILING SPACES
    X = [x.strip() for x in X]
    X = [re.sub(' +', ' ', x) for x in X]
    X = [x.lower() for x in X]
    
    print("Complete!")
    
    #RETURN OUT
    return pd.Series(X)