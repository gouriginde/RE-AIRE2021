import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE 
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings('ignore')
import os
from nltk.tokenize import word_tokenize 


# extracting the stopwords from nltk library
def remove_stopwords(text):
    '''a function for removing the stopword'''
    sw = stopwords.words('english')
    # removing the stop words
    words = [w for w in text if w not in sw]
    return words

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


def tokenize(text):
    # instantiate tokenizer  I use regex to define my pattern
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [t.lower() for t in tokens]
    return tokens

def my_tokenizer(arr):
    '''
    Returns a tokenized version of input array, used in Count Vectorizer
    '''
    return (arr[0]+" "+arr[1]).split()

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    words = words 
    noise_free_words = [word for word in words if not word.isdigit()] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

def word_lemmatizer(text):
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text

def normalize(words):
    #print(words)
    words = remove_punctuation(words)
    words = tokenize(words)
    words = remove_stopwords(words)
    words = replace_numbers(words)
    words = word_lemmatizer(words)
    #print("--------"+words+"-------\n")
    return words

def NLPpipeLine(df_data,col1,col2):
    df_data[col1] = df_data[col1].apply(normalize)
    df_data[col2] = df_data[col2].apply(normalize)
    return df_data

def read_file():
    file_train = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/RedMine_Train.csv"
    file_test = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/RedMine_Test.csv"
    
    df_BERT_test = pd.read_csv(file_test)
    df_BERT_train = pd.read_csv(file_train)
    
    # print(df_BERT_test.head())
    # print(df_BERT_train.head())
    
    df_data_test = df_BERT_test
    col1_test = 'Description1'
    col2_test = 'Description2'
    
    df_data_train = df_BERT_train
    col1_train = 'Description1'
    col2_train = 'Description2'
    
    # preprocess both the trainning ant test datasets
    df_Pro_data_test = NLPpipeLine(df_data_test, col1_test, col2_test)
    df_Pro_data_train = NLPpipeLine(df_data_train, col1_train, col2_train)
    
    df_Pro_data_test.dependency[df_Pro_data_test.dependency == 'relate'] = 1
    df_Pro_data_test.dependency[df_Pro_data_test.dependency == 'independent'] = 2
    
    df_Pro_data_train.dependency[df_Pro_data_train.dependency == 'relate'] = 1
    df_Pro_data_train.dependency[df_Pro_data_train.dependency == 'independent'] = 2
    
    print(df_Pro_data_test.head())
    print(df_Pro_data_train.head())
    
    # save preprocessed datasets to new csv files
    df_Pro_data_test.to_csv("C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Test.csv")
    df_Pro_data_train.to_csv("C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Train.csv")
    

# Random Forest classifier without hyper parameter tunning
def classify_new():
    file_train = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Train.csv" 
    file_test = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Test.csv"
    
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)
    
    train_size = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400] #, 2700, 3000, 3300, 3600, 3900 4200, 4500, 4800, 5100, 5192
    
    #Innitiallize count vestorizer
    cv = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    tfidf_transformer = TfidfTransformer()
    
    ##train
    l = len(df_train)
    print(l)
    
    for i in train_size:
        print(i)
        fac = i/l  #to select 300 it is 300/8304 = 0.036
        ds_train, dummy = train_test_split(df_train, test_size=(1-fac),stratify=df_train['Label'])
        # print(ds_train.shape)
        
        x_trainn = ds_train.loc[:,['Description1', 'Description2']] #Using Description1, Description2 because they have been cleaned - lower case+punctuations
        y_train = ds_train['Label'].astype("int")
       
        #Perform Bag of Words on trainned data
        x_trainf = cv.fit_transform(x_trainn.values.astype('U'))
        x_traint = tfidf_transformer.fit_transform(x_trainf)     
        print(x_traint.shape)
        
        #generating my model
        model = RandomForestClassifier(random_state=0)
        model.fit(x_traint, y_train)
        
        #predict your data
        x_testn = df_test.loc[:,['Description1', 'Description2']]
        y_testn = df_test['Label'].astype("int")
    
        print(len(x_testn))
        
        #Perform Bag of Words on test data
        x_testf = cv.transform(x_testn.values.astype('U'))
        x_testt = tfidf_transformer.fit_transform(x_testf) 
        
        prediction_test = model.predict(x_testt)
        # print(prediction_test)
        
        print("Accuracy = ", metrics.accuracy_score(y_testn, prediction_test))
        print('\n')
        print("=== Classification report ===")
        print(metrics.classification_report(y_testn, prediction_test))
        print('\n')
        print("=== Confusion matrix ===")
        print(metrics.confusion_matrix(y_testn,prediction_test,labels=[0,1]))
        print('\n')
        
# Random Forest classifier with hyper parameter tunning
def classify_hyp():
    file_train = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Train.csv" 
    file_test = "C:/Users/aolt/Desktop/Courses/SENG 607 Software Analytics/project/new data/processed/RedMine_Pro_Test.csv"
    
    df_trainn = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)
     
    train_size = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400] #, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400
    
    #Innitiallize count vestorizer
    cv = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    tfidf_transformer = TfidfTransformer()
    
    #generating my model
    model = RandomForestClassifier(random_state=0)
    
    # Create the parameter random search 
    # #################RF@@############################################## 
    # Number of trees in random forest
    n_estimators = [100, 300, 500, 800, 1200]
    # Maximum number of levels in tree
    max_depth = [5, 8, 15, 25, 30]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10] 

    # Create the random grid
    random_grid = dict(n_estimators = n_estimators, 
                  max_depth = max_depth,  
                  min_samples_split = min_samples_split, 
                  min_samples_leaf = min_samples_leaf)
    
    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation, 
    # search across 10 different combinations, and use all available cores rf_ramdom.fit(X_train, Y_train)
    # rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    # n_iter = 10 by default
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, cv = 3, verbose=2, random_state=0, n_jobs = -1)  
    
    ##train
    l = len(df_trainn)
    print(l)
    
    for i in train_size:
        print(i)
        fac = i/l  #to select 300 it is 300/8304 = 0.036
        df_train, dummy = train_test_split(df_trainn, test_size=(1-fac),stratify=df_trainn['Label'])
        print(len(df_train))
        
        x_trainn = df_train.loc[:,['Description1', 'Description2']]  #Using Description1, Description2 because they have been cleaned - lower case+punctuations
        y_train = df_train['Label'].astype("int")
        
        #Perform Bag of Words on trainned data
        x_trainf = cv.fit_transform(x_trainn.values.astype('U'))
        x_traint = tfidf_transformer.fit_transform(x_trainf)     
        print(x_traint.shape)
        # print(y_train.shape)
        
        # Fitting data in my random search model
        rf_random.fit(x_traint, y_train)
        print(rf_random.best_params_)
        print(rf_random.best_score_)
        params = rf_random.best_params_
        
        # build model with best estimate
        lr_clf =RandomForestClassifier(n_estimators=params['n_estimators'], 
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'], 
                                         min_samples_leaf=params['min_samples_leaf'])
        
        # Fitting data in my new random search model
        lr_clf.fit(x_traint, y_train)
        
        #predict your data
        x_testn = df_test.loc[:,['Description1', 'Description2']]
        y_testn = df_test['Label'].astype("int")
        print(len(x_testn))
        
        #Perform Bag of Words on test data
        x_testf = cv.transform(x_testn.values.astype('U'))
        x_testt = tfidf_transformer.fit_transform(x_testf)
        # print(x_testf)
        # print(x_testf.shape)
        
        prediction_test = lr_clf.predict(x_testt)
        # print(prediction_test)
        
        print("=== Confusion metrx ===")
        cm = metrics.confusion_matrix(y_testn, prediction_test, labels=[0,1])
        print(cm)
        print('\n')
        print('True positive = ', cm[0][0])
        print('False positive = ', cm[0][1])
        print('False negative = ', cm[1][0])
        print('True negative = ', cm[1][1])
        print('\n')
        print("Accuracy = ", metrics.accuracy_score(y_testn, prediction_test))
        print('\n')
        print("=== Classification report ===")
        print(metrics.classification_report(y_testn, prediction_test))
        print('\n')
        
        
    
# read_file()
# classify_new()
classify_hyp()