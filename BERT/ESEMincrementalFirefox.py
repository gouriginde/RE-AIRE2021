import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
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
from imblearn.over_sampling import ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
import os

from nltk.corpus import stopwords 
from sklearn.metrics import jaccard_similarity_score
from nltk.tokenize import word_tokenize 

def my_tokenizer(arr):
    '''
    Returns a tokenized version of input array, used in Count Vectorizer
    '''
    return (arr[0]+" "+arr[1]).split()

def getNeg(df):
    df_neg = pd.DataFrame(columns=df.columns)
    df_neg['req1'] = df['req2']
    
    df_neg['req2'] = df['req1']
    
    df_neg['Label'] = 0

    print("neg samples: ",len(df_neg))
    input("Done with generating negative samples: Hit enter to proceed")
    return df_neg


from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import cross_val_score,train_test_split
#inputFile = "../../processedGenData/All_requires_Pos_Neg_Pairs.csv"
inputFile = "preetham_Typo3_Pos.csv"
def FirefoxTrainTestSplit():
    df_firefox = pd.read_csv(inputFile)
    #df_firefox = df_firefox[(df_firefox['req1Product'] == 'Firefox') & (df_firefox['req2Product'] == 'Firefox')]
    
    print(df_firefox.columns) 
    df_PosChangedColNames = pd.DataFrame()
    #(['Unnamed: 0', 'id1', 'Description1', 'id2', 'Description2', 'Type of dependency']
    df_PosChangedColNames['req1'] = df_firefox['Description1']
    df_PosChangedColNames['req2'] = df_firefox['Description2']
    df_PosChangedColNames['id1'] = df_firefox['id1']
    df_PosChangedColNames['id2'] = df_firefox['id2']
    df_PosChangedColNames['Label'] = df_firefox['Type of dependency']
    df_firefox = df_PosChangedColNames
    print(df_firefox['Label'].value_counts())
    print(df_firefox.columns) 
    
    #encode labels
    # label encoding the data 
    le = LabelEncoder() 
    ### encoding 1: requires, 0: independent
    df_firefox['Label']= le.fit_transform(df_firefox['Label']) 
    print("Data in hand\n", df_firefox['Label'].value_counts()) 
    input("hit enter")
    #get half of -ve samples from inverse of Requires
    poslable_id = 4
    neglable_id = 0
    df_neg = getNeg(df_firefox[df_firefox['Label']==poslable_id]) #4 is for relates
    
    df_firefox_1 =  df_firefox[df_firefox['Label']==poslable_id]
    df_firefox_0 =  df_firefox[df_firefox['Label']!=4]
    # make the id's 0: independent and 1:dependent
    df_firefox_1['Label'] = 1
    df_firefox_0['Label'] = 0
    #df_firefox_0 = df_firefox_0.sample(len(df_firefox_1))

    #pump pure black n white a bit
    lengthIs = len(df_firefox_1)
    df_dummy = pd.concat([df_neg.sample(int(lengthIs*0.7)),df_firefox_0.sample(int(lengthIs*0.3))])
    

    df_firefox = pd.concat([df_dummy, df_firefox_1])
    print("extract balanced\n", df_firefox['Label'].value_counts())
     
    ######Shuffle##############
    df_firefox = df_firefox.sample(frac=1).reset_index(drop=True)
    ###########################

    df_train, df_test = train_test_split(df_firefox, test_size=.2, stratify=df_firefox['Label'])
    print("train set\n", df_train['Label'].value_counts()) 
    print("test set\n", df_test['Label'].value_counts()) 

    df_train.to_csv("Firefox_Train.csv")
    df_test.to_csv("Firefox_Test.csv")
    

'''
train set
1    1834
0    1834

test set
1    459
0    459

'''    
def Classify():
    df = pd.read_csv("Firefox_Train.csv")
    ######Shuffle##############
    df = df.sample(frac=1).reset_index(drop=True)
    ###########################
    
    df_test = pd.read_csv("Firefox_Test.csv")

    train_size = [400, 800, 1200, 1600, 2000,2400, 2800, 3200]#[400, 800, 1200, 1600, 2000]

    #Perform Bag of Words
    count_vect = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    #Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()

    
    ##train
    x = len(df)
    lr_clf = RandomForestClassifier(random_state=0)#, class_weight={0:10,1:20,2:100})
    #lr_clf = MultinomialNB() 
    
    # Create the parameter grid based on the results of random search 
    # #################RF@@############################################## 
    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 

    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

    gridF = GridSearchCV(lr_clf, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)  
    # ################################################################# 

    ##########NB######################################################
    # grid_params = {
    #     'alpha': np.linspace(0.5, 1.5, 6),
    #     'fit_prior': [True, False],
    #     }
    # gridF = GridSearchCV(lr_clf, grid_params, cv = 3, verbose = 1, 
    #                   n_jobs = -1)
    
    # #######################################################################
    for i in train_size:
        print(i)
        fac = i/x  #to select 400 it is 400/1834 = 0.21
        df_train, dummy = train_test_split(df,test_size=(1-fac),stratify=df['Label'])
        print(len(df_train))
        X_train = df_train.loc[:,['req1','req2']]  #Using req_1,req_2 rather than req1,req2 because req_1,req_2 have been cleaned - lower case+punctuations
        y_train = df_train['Label'].astype("int")
    
        X_train_counts = count_vect.fit_transform(np.array(X_train))
        X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
        features = X_train_tfidf
        labels = y_train

        #Hyper param tuning############## 
        bestF = gridF.fit(features, labels)
        print(bestF.best_params_)
        print(bestF.best_score_)
        params = bestF.best_params_
        
        #RF#
        lr_clf =RandomForestClassifier(n_estimators=params['n_estimators'], 
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'], 
                                         min_samples_leaf=params['min_samples_leaf'])

        #NB#
        #lr_clf = MultinomialNB(alpha = params['alpha'],
        #                        fit_prior= params['fit_prior'] )
        # ##################################
        #input("Hit enter")

        
        #model and predict
        #######prep Test : one time activity #####
        #print("Test set is:\n", df_test['Label'].value_counts())
        y_test = df_test.loc[:,'Label'].astype("int")
        #df_test = df_test.drop(['Label'], axis=1)
        X_test = np.array(df_test.loc[:,['req1','req2']])  #Using req_1,req_2 rather than req1,req2 because req_1,req_2 have been cleaned - lower case+punctuations
        X_test_counts = count_vect.transform(np.array(X_test))
        X_test_tfidf= tfidf_transformer.transform(X_test_counts)
        features_test = X_test_tfidf
        ###############################################
    
    
        lr_clf.fit(features, y_train)
        print(lr_clf.score(features_test, y_test))
        scores = cross_val_score(lr_clf, features, labels)
        print("MY classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        predictions = lr_clf.predict(features_test)
        print(metrics.classification_report(y_test,predictions))
        print(metrics.confusion_matrix(y_test,predictions,labels=[0,1]))

        

def diagnostics():
    file = "../../processedGenData/Firefox_requires_Pos_Neg_Pairs.csv"
    df_firefox = pd.read_csv(file)
    print(df_firefox['Label'].value_counts())
    df_firefox = df_firefox[(df_firefox['req1Product'] == 'Firefox') & (df_firefox['req2Product'] == 'Firefox')]
    print(df_firefox['Label'].value_counts())

    
FirefoxTrainTestSplit()
#Classify()
#diagnostics()
