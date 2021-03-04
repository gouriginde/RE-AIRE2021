import os
import numpy as np
import pandas as pd
import warnings
import logs
import clf_model
import annotate
#from splitLabelledData import splitDataSet
# lables are converted as 
# {0: 'incorporates', 1: 'independent', 2: 'relates to'}
path = "../ExtractedData/"
fields = ['summary1','summary2','dependency','id1', 'id2','label']
#fullFile = path+"Solr_Processed_SplitAndTaggedData.csv"
fullFile = path+"Solr_Processed_SplitAndTaggedData_posNeg_5types.csv"
depType =  {'incorporates':0, 'independent':1, 'relates to':2}
#print(depType['independent'])
#input("hit enter")
label = 'label'
req1 = 'summary1'
req2 = 'summary2'
req1Id = 'id1'
req2Id = 'id2'
annStatus = 'AnnotationStatus'

projectName = "Solr"


def learnTargetLabel(args):
    '''
    Active Learning iterative process
    1. Prepare Data
    2. Create Classifier
    3. Evaluate Classifier
    4. Select Uncertain Samples and get them annotated by Oracle
    5. Update Data Set (Merge newly annotated samples to original dataset) 
    6. Repeat steps 1-5 until stopping condition is reached.

    Parameters : 
    args (dataframe) : Run-time arguments in a dataframe.

    Returns :
    df_rqmts (dataframe) : Updated / Final requirements dataset, included the prediction values at the last iteration of Active learning process. 
    df_resultTracker (dataframe) : Results for tracking purpose

    '''
    #Read run time arguments
    idir = os.getcwd()+args.loc[0,'input']    
    splitratio = float(args.loc[0,'testsize']) 
    maxIterations = int(args.loc[0,'maxIterations'])
    resamplingTechnique = args.loc[0,'resampling']
    print(idir,splitratio,maxIterations,resamplingTechnique)
    #input("hit enter")
    
    logs.writeLog("Fetching data from the input directory.")
    #Read To be Annotated, Training, Test and Validation Sets generated after executing splitData.py
    try:
        df_rqmts = pd.read_csv(idir+fullFile) #this has training data with Annotated = 'M' and rest with Nothing ''
        print(df_rqmts[annStatus].value_counts())

    except FileNotFoundError as err:
        logs.writeLog ("File Not Found! Please provide correct path of the directory containing Training, Test, Validation, ToBeAnnotated and Manually Annotated DataSet.")
        print (err)
        exit()

    
    #Create a dataframe to track the results
    df_resultTracker = pd.DataFrame()
    iteration = 0
    df_rqmts=df_rqmts.sample(frac=1) #shuffulles
    df_rqmts[label] = df_rqmts[label].astype('int')
  
    df_training = df_rqmts[df_rqmts[annStatus]=='M']
    df_testing = df_rqmts[df_rqmts[annStatus]!='M']
    
    #these are two df's for local model(LM) training for first iteration
    df_LM_training = df_training
    df_LM_testing = df_testing

    while True:
        iteration+=1
        logs.writeLog("\n"+100*"-")
        logs.writeLog("\n\nIteration : "+str(iteration)+"\n")
        #####run it multiple times say 10 and accumulate average results
        V_f1=[]
        V_prec = []
        V_rcl=[]
        V_indPrec = []
        V_indRcl=[]
        V_indF1=[]
        V_ReqPre=[]
        V_ReqRcl=[]
        V_ReqF1=[]
        V_SimPrec=[]
        V_SimRcl=[]
        V_SimF1=[]
        V_confusioMatrix=[]

        LM_f1=[]
        LM_prec=[]
        LM_rcl=[]
        LM_indPrec=[]
        LM_indRcl=[]
        LM_indF1=[]
        LM_ReqPre=[]
        LM_ReqRcl=[]
        LM_ReqF1=[]
        LM_SimPrec=[]
        LM_SimRcl=[]
        LM_SimF1=[]
        LM_confusionMatrix = []
        for i in range(9):
            #-----------------------------------------AL model -------------------------------------#
            logs.writeLog("\nCreating Classifier...")
            #just pass the ones with AnnotationStatus = 'M'
            countVectorizer, tfidfTransformer, classifier, classifierTestScore = clf_model.createClassifier(args.loc[0,'classifier'],df_training,resamplingTechnique)  
            logs.writeLog("\n\n5 fold Cross Validation Score : "+str(classifierTestScore))
            logs.writeLog ("\n\nValidating Classifier...")
            
            #pass the rest as testing data annStatus]!='M'
            classifierValidationScore,v_f1Score,v_precisionScore,v_recallScore,v_precisionArr,v_recallArr,v_fscoreArr,v_supportArr,v_confusionMatrix = clf_model.validateClassifier(countVectorizer,tfidfTransformer,classifier,df_testing)
            logs.writeLog("\n\nClassifier Validation Set Score : "+str(classifierValidationScore))
            
            #Update Analysis DataFrame (For tracking purpose)
            df_training[label] = df_training[label].astype('int')
            independentCount = len(df_training[df_training[label]==depType['independent']])
            requiresCount = len(df_training[df_training[label]==depType['relates to']])
            similarCount = len(df_training[df_training[label]==depType['incorporates']])
            #----------------------------------------------AL ends-----------------------------------------

            
            #-----------------------------------------LM starts-------------------------------------------
            LM_countVectorizer, LM_tfidfTransformer, LM_classifier, LM_classifierTestScore= clf_model.createClassifier(args.loc[0,'classifier'],df_LM_training,resamplingTechnique)
                
            #pass the rest as testing data annStatus]!='M'
            LM_classifierValidationScore,LM_f1Score,LM_precisionScore,LM_recallScore,LM_precisionArr,LM_recallArr,LM_fscoreArr,LM_supportArr,lm_confusionMatrix = clf_model.validateClassifier(LM_countVectorizer,LM_tfidfTransformer,LM_classifier,df_LM_testing)
            logs.writeLog("\n\nClassifier Validation Set Score for LM: "+str(LM_classifierValidationScore))
            
            #Update Analysis DataFrame (For tracking purpose)
            df_LM_training[label] = df_LM_training[label].astype('int')
            LM_independentCount = len(df_LM_training[df_LM_training[label]==depType['independent']])
            LM_requiresCount = len(df_LM_training[df_LM_training[label]==depType['relates to']])
            LM_similarCount = len(df_LM_training[df_LM_training[label]==depType['incorporates']])
            
            #-----------------------------------------LM Ends--------------------------------------------
            #store results to average later
            V_f1.append(v_f1Score)
            V_prec.append(v_precisionScore)
            V_rcl.append(v_recallScore)
            V_indPrec.append(v_precisionArr[0])
            V_indRcl.append(v_recallArr[0])
            V_indF1.append(v_fscoreArr[0])
            V_ReqPre.append(v_precisionArr[1])
            V_ReqRcl.append(v_recallArr[1])
            V_ReqF1.append(v_fscoreArr[1])
            V_SimPrec.append(v_precisionArr[2])
            V_SimRcl.append(v_recallArr[2])
            V_SimF1.append(v_fscoreArr[2])
            V_confusioMatrix.append(v_confusionMatrix)

            LM_f1.append(LM_f1Score)
            LM_prec.append(LM_precisionScore)
            LM_rcl.append(LM_recallScore)
            LM_indPrec.append(LM_precisionArr[0])
            LM_indRcl.append(LM_recallArr[0])
            LM_indF1.append(LM_fscoreArr[0])
            LM_ReqPre.append(LM_precisionArr[1])
            LM_ReqRcl.append(LM_recallArr[1])
            LM_ReqF1.append(LM_fscoreArr[1])
            LM_SimPrec.append(LM_precisionArr[2])
            LM_SimRcl.append(LM_recallArr[2])
            LM_SimF1.append(LM_fscoreArr[2])
            # 
            
        tempList = (lm_confusionMatrix.tolist())
        LM_confusionMatrix.append(tempList)
        #print(tempList[-1],"\n", tempList)
        #input("hit enter")    
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,
                                                    'Total data':len(df_rqmts),
                                                    'TraiAKA_ManlyAntd':len(df_training),
                                                    'Testing':len(df_testing),
                                                    'CV':classifierTestScore,
                                                    '#Ind':independentCount,
                                                    '#Req':requiresCount,
                                                    '#Sim':similarCount,
                                                    'f1':"{:.2f}".format(np.average(V_f1)),
                                                    'prec':"{:.2f}".format(np.average(V_prec)),
                                                    'rcl':"{:.2f}".format(np.average(V_rcl)),
                                                    'indPrec': "{:.2f}".format(np.average(V_indPrec)),
                                                    'indRcl':"{:.2f}".format(np.average(V_indRcl)),
                                                    'indF1':"{:.2f}".format(np.average(V_indF1)),
                                                    'indSup':v_supportArr[0],
                                                    'ReqPre': "{:.2f}".format(np.average(V_ReqPre)),
                                                    'ReqRcl':"{:.2f}".format(np.average(V_ReqRcl)),
                                                    'ReqF1':"{:.2f}".format(np.average(V_ReqF1)),
                                                    'ReqSup':v_supportArr[1],
                                                    'SimPrec': "{:.2f}".format(np.average(V_SimPrec)),
                                                    'SimRcl':"{:.2f}".format(np.average(V_SimRcl)),
                                                    'SimF1':"{:.2f}".format(np.average(V_SimF1)),
                                                    'SimSup':v_supportArr[2],
                                                    'ConfusionM':v_confusionMatrix,
                                                    
                                                    '#LM Training':len(df_LM_training),
                                                    '#LM Testing':len(df_LM_testing),
                                                    'LM CV':LM_classifierTestScore,
                                                    '#LM Ind':LM_independentCount,
                                                    '#LM Req':LM_requiresCount,
                                                    '#LM Sim':LM_similarCount,
                                                    'LM f1':"{:.2f}".format(np.average(LM_f1)),
                                                    'LM prec':"{:.2f}".format(np.average(LM_prec)),
                                                    'LM rcl':"{:.2f}".format(np.average(LM_rcl)),
                                                    'LM indPre': "{:.2f}".format(np.average(LM_indPrec)),
                                                    'LM indRcl':"{:.2f}".format(np.average(LM_indRcl)),
                                                    'LM indF1':"{:.2f}".format(np.average(LM_indF1)),
                                                    'LM indSup':LM_supportArr[0],
                                                    'LM ReqPre': "{:.2f}".format(np.average(LM_ReqPre)),
                                                    'LM ReqRcl':"{:.2f}".format(np.average(LM_ReqRcl)),
                                                    'LM ReqF1':"{:.2f}".format(np.average(LM_ReqF1)),
                                                    'LM ReqSup':LM_supportArr[1],
                                                    'LM SimPrec': "{:.2f}".format(np.average(LM_SimPrec)),
                                                    'LM SimRcl':"{:.2f}".format(np.average(LM_SimRcl)),
                                                    'LM SimF1':"{:.2f}".format(np.average(LM_SimF1)),
                                                    'LM SimSup':LM_supportArr[2],
                                                    'LM ConfusionM':LM_confusionMatrix[-1]
                                                    

                                                    },ignore_index=True)


        logs.writeLog("\n\nAnalysis DataFrame : \n"+str(df_resultTracker))
        print("-----------Before-----------")
        print(len(df_rqmts),"=",len(df_training),"+", len(df_testing))
        logs.writeLog ("\n\nPredicting Labels....")
        df_predictionResults = clf_model.predictLabels(countVectorizer,tfidfTransformer,classifier,df_testing)   #operate on the df_rqmts only
        
        logs.writeLog("\n\nFinding Uncertain Samples and Annotating them.....")
        df_finalPredictions, df_remaining_Testing = annotate.analyzePredictions(args,df_predictionResults)
        logs.writeLog("\n\nMerging Newly Labelled Data Samples....")
        df_rqmts = pd.concat([df_training,df_finalPredictions],axis=0,ignore_index=True)
        df_rqmts = pd.concat([df_rqmts,df_remaining_Testing],axis=0,ignore_index=True)
        print("After")
        print(len(df_rqmts),"=", len(df_training),"+", len(df_remaining_Testing),"+", len(df_finalPredictions))
        print(df_rqmts[annStatus].value_counts())
        print("-"*100)
        #Remove unwanted columns
        df_rqmts = df_rqmts[[req1,req2,label,annStatus]]#df_rqmts[['req_2','req_1',label,annStatus]]#df_rqmts[['comboId','req1Id','req1','req_1','req2Id','req2','req_2',label,annStatus]]

        #input("hit enter to proceed")
        
        
        #if iteration >=maxIterations:
        #new stopping condition is if the validation set is more than equal to or less than 30% of training size
        #df_validation = df_rqmts[df_rqmts[annStatus]!='M']
        if int(len(df_testing)) <= int(0.3*(int(len(df_training)))) or iteration >=maxIterations:
            logs.writeLog("\n\nStopping Condition Reached... Exiting the program."+str(len(df_testing))+str(len(df_training)))
            break
        
        #----------------for next iteration-------------------------------------#
        df_rqmts = df_rqmts.sample(frac=1)
        df_training = df_rqmts[df_rqmts[annStatus]=='M']
        df_testing = df_rqmts[df_rqmts[annStatus]!='M']
        
        #add equal amount of randomly selected labels to LM models
        #for this extract the number from df_finalPredictions
        stats=df_finalPredictions[label].value_counts()

        print("-------------Before LM---------------")
        print(len(df_rqmts),len(df_LM_training), len(df_LM_testing))
        print(stats)
        for key in stats.keys():
            #fetch the sample of size values for each class randomly and add tp LM training set
            #and remove from testing set
            sampleCount = int(stats[key])
            print(type(sampleCount), sampleCount)
            #df_temp = df_LM_testing[df_LM_testing["Label"]==key].sample(sampleCount)
            df_temp= df_LM_testing[df_LM_testing[label]==key].sample(sampleCount)
            df_temp[annStatus] == 'M'
            df_LM_training = pd.concat([df_LM_training,df_temp],axis=0,ignore_index=True)
            #print(sampleCount, len(df_temp), len(df_LM_testing), len(df_LM_training))
            df_LM_testing = df_LM_testing[~df_LM_testing.isin(df_temp)].dropna(how='any', subset=[req1, req2]) #df_LM_testing[~df_LM_testing.isin(df_temp)].dropna(how='any', subset=['req1', 'req2']) 
            df_LM_testing[annStatus] = ""
            #print(sampleCount, len(df_temp), len(df_LM_testing))

        print("-------------After LM---------------")
        print(len(df_rqmts),len(df_LM_training), len(df_LM_testing))
        #input ("Hit enter")
        #house keeping    
        df_LM_training = df_LM_training.sample(frac=1)
        df_LM_training[label] = df_LM_training[label].astype('int')
        df_LM_testing[label] = df_LM_testing[label].astype('int')
        #---------------------------------------Next iteration ends ---------------- #
        
    return df_rqmts,df_resultTracker


def main():
    #Ignore Future warnings if any occur. 
    warnings.simplefilter(action='ignore', category=FutureWarning)  
    
    pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
    pd.set_option('display.width', 1000)

    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    
    #Reads run time arguments
    args = logs.getArguments(currentFileDir+"/ALParams.txt") 
    comments = args.loc[0,'comments']
    #print(args)
    #code glueing to not disturb wht ikagarjot did
    #df_myargs = pd.DataFrame()
    #df_myargs['comments'] = args.loc[0,'comments']
    #df_myargs['input'] = args.loc[0,'input']
    #print(df_myargs)
    
    inputPath = str(args.loc[0,'input'])
    comment = str(args.loc[0,'comments'])
    
    args['input'] = inputPath #+ str(i) #"FirefoxBuildSystem"
    args['comments'] = projectName + comment
    #Creates Logs folder structure
    logFilePath,OFilePath = logs.createLogs(currentFileDir+"/Logs",args)   

    df_rqmtComb,df_Analysis = learnTargetLabel(args)

    #Adds the Analysis DataFrame to Output File
    logs.addOutputToExcel(df_Analysis,"\nAnalysis of  Label Classification  : \n")

    #logs.updateResults(df_rqmtComb,args)   #Update Results in excel....

    logs.writeLog("\nOutput Analysis is available at : "+str(OFilePath))
    logs.writeLog("\nLogs are available at : "+str(logFilePath))

    print(args)
    #input("hit enter")

    
if __name__ == '__main__':
    main()