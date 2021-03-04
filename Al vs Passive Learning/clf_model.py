import logs
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
from sklearn.model_selection import cross_val_predict
from scipy.stats import sem
from numpy import mean
from numpy import std


label = 'label'
req1 = 'summary1'
req2 = 'summary2'
req1Id = 'id1'
req2Id = 'id2'
annStatus = 'AnnotationStatus'



def computeGridnget(X,y):
    print("Grid searching")
    rf = RandomForestClassifier()
    #weights = np.linspace(0.005, 0.05, 10)
    params = {'class_weight':{2:10}}
    gsc = GridSearchCV(param_grid = params, estimator=rf)
    grid_result = gsc.fit(X, y)

    print("Best parameters : %s" % grid_result.best_params_)
    return

def createClassifier(clf,df_trainSet,resampling_type):
    df_trainSet= df_trainSet.sample(frac=1)
    '''
    Passes the dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Performs Synthetic Monitoring Over-Sampling after performing TFIDF transformation (ONLY when resampling_type is over_sampling)
    Trains the classifier (Random Forest / Naive Bayes / SVM / Ensemble using Voting Classifier)

    Parameters : 
    clf (str) : Name of classifier (options - RF, NB, SVM , ensemble)
    df_trainSet (DataFrame) : Training Data
    df_testSet (DataFrame) : Test Data

    Returns : 
    count_vect : Count Vectorizer Model
    tfidf_transformer : TFIDF Transformer Model
    clf_model : Trained Model 
    clf_test_score (float) : Accuracy achieved on Test Set 
    f1/precision/recall (float) : F1, Precision and Recall scores (macro average)
    '''

    #df_trainSet = shuffle(df_trainSet)
    #df_testSet = shuffle(df_testSet)

    #Convert dataframes to numpy array's
    X_train = df_trainSet.loc[:,[req1,req2]]  #Using req_1,req_2 rather than req1,req2 because req_1,req_2 have been cleaned - lower case+punctuations
    y_train = df_trainSet.loc[:,label].astype("int")

    logs.writeLog("\nTraining Set Size : "+str(len(X_train)))
    logs.writeLog("\nTrain Set Value Count : \n"+str(df_trainSet[label].value_counts()))

    logs.writeLog("\n\nTraining Model....")
    
    #Perform Bag of Words
    count_vect = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    X_train_counts = count_vect.fit_transform(np.array(X_train))
    
    #Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
    
    #######################################################################################
    # if resampling_type == "over_sampling":
    #     logs.writeLog("\n\nValue Count for each class in training set."+str(Counter(y_train)))
        
    #     logs.writeLog("\n\nPerforming Over Sampling")
    #     sm = SMOTE()#k_neighbors=3) #ADASYN(sampling_strategy="minority")
    #     X_train_tfidf, y_train = sm.fit_resample(X_train_tfidf, y_train)
    #     logs.writeLog("\n\nValue Count for each class in training set."+str(Counter(y_train)))
    ######################################################################################

    #Initiate Classifiers
    rf_model = RandomForestClassifier(random_state=0, class_weight={0:10,1:20,2:100})
    nb_model = MultinomialNB()
    svm_model = SVC(random_state = 0, probability=True)  #predict_proba not available if probability = False

    #Random Forest Classifier Creation
    if clf == "RF" :
        clf_model = rf_model.fit(X_train_tfidf, np.array(y_train).astype('int'))
        
    #Naive Bayes Classifier Creation
    elif clf == "NB":
        clf_model = nb_model.fit(X_train_tfidf,np.array(y_train).astype('int'))

    #Support Vector Machine Classifier Creation.
    elif clf == "SVM":
        clf_model = svm_model.fit(X_train_tfidf,np.array(y_train).astype('int'))
    
    #Ensemble Creation
    elif clf == "ensemble":
        #Predict_proba works only when Voting = 'soft'
        #n_jobs = -1 makes allows models to be created in parallel (using all the cores, else we can mention 2 for using 2 cores)
        #clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model),('SVM',svm_model)], voting='soft',n_jobs=1)  
        clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model), ('SVM',svm_model) ], voting='soft',n_jobs=1)#,weights=[30,10])  
        clf_model.fit(X_train_tfidf,np.array(y_train).astype('int'))
 
    
    #perform cross validation instead of train and test split
    clf_test_score=evaluate_model(X_train_tfidf,np.array(y_train).astype('int'),clf_model)
    print(str(round(clf_test_score.mean(),2)) +"(+/- "+str(round(clf_test_score.std()*2,2))+")")
    #input("hit enter")
    return count_vect, tfidf_transformer, clf_model, clf_test_score

# evaluate a model

def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # summarize
    #print(mean(scores), sem(scores))
    #input("hit enter")
    #y_pred = cross_val_predict(model, X, y, cv=cv)
    #conf_mat = confusion_matrix(y, y_pred)
	return scores


def predictLabels(cv,tfidf,clf,df_toBePredictedData):
    '''
    Passes the to be predicted dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Predicts and returns the labels for the input data in a form of DataFrame.

    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_toBePredictedData (DataFrame) : To Be Predicted Data (Unlabelled Data)

    Returns : 
    df_toBePredictedData (DataFrame) : Updated To Be Predicted Data (Unlabelled Data), including prediction probabilities for different labels
    '''
    predictData = np.array(df_toBePredictedData.loc[:,[req1,req2]])
    #logs.writeLog(str(df_toBePredictedData))
    
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    predict_labels = clf.predict(predict_tfidf)
    predict_prob = clf.predict_proba(predict_tfidf)
    
    logs.writeLog ("\nTotal Labels Predicted : "+ str(len(predict_labels)))

    df_toBePredictedData['predictedProb'] = predict_prob.tolist() 
    df_toBePredictedData['maxProb'] = np.amax(predict_prob,axis=1)
    
    return df_toBePredictedData    

def validateClassifier(cv,tfidf,clf_model,df_validationSet):
    '''
    Passes the validation dataset (Unseen data) via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Calculate the accuracy and other metrics to evaluate the performance of the model on validation set (unseen data)
    
    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_validationSet (DataFrame) : Validation Data (Unseen Data)

    Returns : 
    clf_val_score/f1/precision/recall (float) : Accuracy Value on Validation Data / F1 score / Precision / Recall
    '''
    
    predictData = np.array(df_validationSet.loc[:,[req1,req2]])
    actualLabels = np.array(df_validationSet.loc[:,label]).astype('int')
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    
    predict_labels = clf_model.predict(predict_tfidf)
    clf_val_score = clf_model.score(predict_tfidf,actualLabels)
    precisionArr,recallArr,fscoreArr,supportArr=score(actualLabels,predict_labels,average=None)

    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    
    labelClasses = list(set(actualLabels))   #np.array(y_train).astype('int')
    logs.writeLog ("\n\nClassification Report On Validation Set: \n\n"+str(classification_report(actualLabels,predict_labels)))
    cm = confusion_matrix(actualLabels,predict_labels,labels=labelClasses)    
    logs.writeLog ("\n\nConfusion Matrix : \n"+str(cm)+"\n")
    
    return clf_val_score,f1,precision,recall, precisionArr,recallArr,fscoreArr,supportArr, cm

def my_tokenizer(arr):
    '''
    Returns a tokenized version of input array, used in Count Vectorizer
    '''
    return (arr[0]+" "+arr[1]).split()


