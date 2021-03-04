import os
import warnings
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import logs
import argparse

#project = "Firefox Build System" #input("Enter the project name: ")
#lables are converted as {0: 'incorporates', 1: 'independent', 2: 'relates to'}

path = "../ExtractedData/"
negFile = "Processed_Solr_NegativePairs.csv"
posFile = "Processed_Solr_PositivePairs.csv"
fields = ['summary1','summary2','dependency','id1', 'id2','label']
fullFile = path+"Processed_Solr_concatenated_posNeg_5types.csv"#"Processed_Solr_concatenated_posNeg.csv"
splitIs = .15 #this is x%
#Sample Command to execute this file
#python3 splitLabelledData.py -i "/data/preprocessed/firefox_data.csv" -o "/data/processed/firefox"

label = 'dependency'
req1 = 'summary1'
req2 = 'summary2'
req1Id = 'id1'
req2Id = 'id2'
annStatus = 'AnnotationStatus'

def diagnostics(df):
    print(df[label].value_counts())
    print(df[annStatus].value_counts())
    print(df[df[annStatus]=='M'].groupby(df[label]))

    print("size after removing duplicates: ",len(df))
    # sorting by first name 
    df.sort_values(req1Id, inplace = True)
    df.drop_duplicates(subset=[req1Id,req2Id], inplace=True)
    print("size after removing duplicates: ",len(df))

    pass


def splitDataSet(df_rqmts,frac,balancedClass):
    '''
    Performs data balancing by undersampling.
        1. Does a value count for each class label and extacts the minimum value.
        2. Selects fraction * minimum value (Undersampling) of the data points from each class.

    Parameters : 
    df_rqmts (DataFrame) : Dataframe containing requirement combinations and corresponding labels.
    colName (str) : Name of the column on which value counts/ data balancing is to be performed
    frac (str) : Fraction of requirement combinations that need to separated.

    Returns 
        df_sampledCombinations (DataFrame) : Sampled Balanced Dataset 
        df_rqmts (DataFrame) :  Remaining Dataset
    '''

    print("\nSplitting Data :-")
    print ("\nOriginal Size of Dataset: "+str(len(df_rqmts)))
    #df_rqmts[label] = df_rqmts[label].astype('int')
    stats = df_rqmts[label].value_counts()  #Returns a series of number of different types of TargetLabels (values) available with their count.
    print ("Value Count : \n"+str(stats))

    #if resampleType == "under_sampling":
    #    count = int(stats.min()*float(frac))
    #print ("\nSampled Combinations for each class : "+str(count) + " ("+str(frac)+" of the total combinations)") 
        
    df_sampledCombinations = pd.DataFrame(columns=df_rqmts.columns)
    df_rqmts = df_rqmts.sample(frac=1) #shuffling
    for key in stats.keys():
        #Sample out some values for df_data Set for each label 0,1,2,3
        if (balancedClass == False):
            sample_count = int(stats[key]*float(frac))  
            if (sample_count>1000):  #Limiting Sample Count to 500
                sample_count = 1000
        elif (balancedClass == True):
            sample_count = sample_count = int(stats.min()*float(frac))
        else:
            raise ("Invalid Input. Please enter True/False for balancedClass.")
        df_sample = df_rqmts[df_rqmts[label]==key].sample(sample_count)
        print ("\nSampled "+str(len(df_sample))+" Combinations for class "+str(key) + " ("+str(frac)+" of the total combinations)") 
     
        df_sample[annStatus] = "M" #Mark the sampled values as Annotation 'M' - these combinations will be the inital training dataset.
        df_rqmts = df_rqmts[~df_rqmts.isin(df_sample)].dropna()  #Remove Sampled Values from original data set.
        df_rqmts[annStatus] = "" #mark remaning nothing
        df_sampledCombinations = pd.concat([df_sampledCombinations,df_sample],axis=0)   #Add sampled values into the Test Set

    print ("\nSize of Sampled Combinations : "+str(len(df_sampledCombinations)))
    print ("\nSize of Remaining Combinations : "+str(len(df_rqmts)))
    #input("hit enter")    
    #again concat the two and return just one file
    df_combined = pd.concat([df_sampledCombinations,df_rqmts],axis=0)
    #diagnostics(df_cobined)
    #return df_sampledCombinations,df_rqmts   
    return df_combined

def main():
    #Ignore Future warnings if any occur. 
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    
    #To make sure all the columns are visible in the logs.
    pd.set_option('display.max_columns', 500)   
    pd.set_option('display.width', 1000)

    
    print("\nGenerating dataframe from the input file.")
    df_data = pd.read_csv(fullFile,',',encoding="utf-8",error_bad_lines=False)
       
    df_data[annStatus] = ""

    print(df_data.keys())
    #input("hit enter")
    
    #diagnostics
    #diagnostics(df_data)
    #input("hit enter")

    print ("\nPreparing Data.......")
    print ("\n"+"-"*150)
    print("Fetching data from the input file and Marking 10% of the combinations as Manually Annotated. Setting 'Annotation Status' as 'M'")
    print ("-"*150)

    df_changedData = splitDataSet(df_data,splitIs, balancedClass = True) 
    
    #print ("\nSaving the datasets after splitting at : "+odirName)
    
    df_changedData.to_csv(path+"Solr_Processed_SplitAndTaggedData_posNeg_5types.csv",index=False)
    print("stats as follows")
    print("Split and changed data: ",len(df_changedData))
    diagnostics(df_changedData)
    
if __name__ == "__main__":
   main()

def balancedClasses(df):
    stats = df["label"].value_counts()
    print(stats)
    df_sampledData= pd.DataFrame(columns=df.columns)
    sampleSize = int(input("enter the size"))
    for key in stats.keys():
        allData = df[df["label"]==key]
        if sampleSize > len(allData):
            sampleSize = len(allData)
            df_sample = allData.sample(sampleSize)
        else:
            df_sample = allData.sample(sampleSize)
        
        df_sampledData = pd.concat([df_sampledData,df_sample],axis=0)   #Add sampled values into the Test Set
    return df_sampledData

def combineData():
    dfPos = pd.read_csv(path+posFile)
    dfNeg = pd.read_csv(path+negFile)
    selectTYpes = ['relates to', 'incorporates',
                   'depends upon', 'duplicates',
                   'blocks']
    #df = pd.concat([dfPos,dfNeg])
    #print(df['dependency'].value_counts(), len(df))
    #input("hit enter")
    #selecting just top 3 dependency types for processing further
    # independent       50001
    # relates to         1650
    # incorporates        266
    # depends upon        228
    # duplicates          208
    # blocks              174
    # supercedes           90
    # requires             68
    # contains             52
    # is a clone of        20
    # is a parent of       12
    # breaks               10
    # Blocked               4
    
    #dfPos['dependency'] = dfPos['dependency'].astype('str')
    dfPos = dfPos[dfPos['dependency'].isin(selectTYpes)]
    print(dfPos['dependency'].value_counts(), len(dfPos))
    #print(dfPos.dependency.unique())
    #input("hit enter")
    
    df = pd.concat([dfPos,dfNeg])
    df['label'] = df['dependency']
    print(df['dependency'].value_counts(), len(df))
    c = df.label.astype('category')
    d = dict(enumerate(c.cat.categories))
    print (d)
    #{0: 'english', 1: 'spanish'}
    df['label'] = df.label.astype('category').cat.codes
    print(df.head(), df.columns)

    df = balancedClasses(df)
    print(df['dependency'].value_counts(), len(df))
    #input("hit enter")
    #AFter selecting almost balanced data
    # relates to      266
    # independent     266
    # incorporates    266
    # depends upon    228
    # duplicates      208
    # blocks          174

        
    df.to_csv(path+"Processed_Solr_concatenated_posNeg_5types.csv")

#combineData()