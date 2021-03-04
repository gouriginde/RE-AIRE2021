import re, string, unicodedata
import nltk
#import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd
from html.parser import HTMLParser
html_parser = HTMLParser()
import glob

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# def replace_contractions(text):
#     """Replace contractions in string of text"""
#     return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    sample =  re.sub(r"http\S+", "", sample)
    #sample = re.sub(r'[^\w\s]', '', sample)  #return puctuation
    sample = re.sub(r"[\[\],@\'?\.$%_:()\-\"&;<>{}|+!*#]", " ", sample, flags=re.I)
    #sample = re.sub(r"\s+"," ", sample, flags = re.I) #remove empty spaces
    sample = re.sub(r"\s+[a-zA-Z]\s+", " ", sample) #remove single characters
    sample = ' '.join(w for w in sample.split() if not any(x.isdigit() for x in w)) #remove of the type e70bae07664def86aefd11c86dac818ab7ea64ea
    return sample.lower()


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    words = words.split() 
    noise_free_words = [word for word in words if not word.isdigit()] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    #new_words = []
    words = words.split()
    noise_free_words = [word for word in words if word not in stopwords.words('english')] 
    #noise_free_words = [word for word in words if word not in ["meta","META",'a','the','an']] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    words = words.split()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in words])
    # for word in words:
    #     word = lemmatizer.lemmatize(word, pos='v')
    #     #lemmas.append(word)
    #     #word = word.lemmatize()
    # words = " ".join(words) 
    return lemmatized_output

def normalize(words):
    #words = remove_non_ascii(words)
    #print(words)
    words = remove_URL(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    #print("--------"+words+"-------\n")
    return words

def NLPpipeLine(df_data,col1,col2):
    #Tokenize
    #df_data[col1] = df_data[col1].apply(nltk.word_tokenize)
    #df_data[col2] = df_data[col2].apply(nltk.word_tokenize)
    
    # Normalize
    #words = normalize(words)
    df_data[col1] = df_data[col1].apply(normalize)
    df_data[col2] = df_data[col2].apply(normalize)
    #print(df_data['req1'].head())
    #input("hit enter")

    #print(words)

    return df_data



