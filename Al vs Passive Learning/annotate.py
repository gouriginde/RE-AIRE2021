import os
import pandas as pd
import numpy as np
import logs
from uncertaintySampling import leastConfidenceSampling,minMarginSampling,entropySampling

label = 'label'
req1 = 'summary1'
req2 = 'summary2'
req1Id = 'id1'
req2Id = 'id2'
annStatus = 'AnnotationStatus'
fields = ['summary1','summary2','dependency','id1', 'id2','label']
def analyzePredictions(args,df_predictions):
    '''
    Analyzis the predictions, samples the most uncertain data points and queries it from the oracle (original database/file) and updates dataframe accordingly.
    '''
    #df_manuallyAnnotated = pd.DataFrame(columns=['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus])#Create an empty Dataframe to store the manually annotated Results

    queryType = args.loc[0,'samplingType']
    df_userAnnot = pd.DataFrame(columns = fields)
    
    for field in [0,1,2,3,4,5]:
        iteration = 0
        logs.writeLog("\n\nIteration for field: "+str(field))
        #input("hit enter to proceed")
        while iteration<int(args.loc[0,'manualAnnotationsCount']):  #while iteration is less than number of annotations that need to be done.
            if (len(df_predictions[df_predictions[label]==field ])>0):
                logs.writeLog("\n\nIteration : "+str(iteration+1))
                if queryType == 'leastConfidence':
                    indexValue = leastConfidenceSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'minMargin':
                    indexValue = minMarginSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'entropy':
                    indexValue =entropySampling(df_predictions[df_predictions[label]==field ])
            
                sample = df_predictions.loc[indexValue,:]
                logs.writeLog("\n\nMost Uncertain Sample : \n"+str(sample))
                df_userAnnot = df_userAnnot.append({req1:sample[req1],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)#df_userAnnot.append({'comboId':sample['comboId'],'req1Id':sample['req1Id'],'req1':sample['req1'],req1:sample[req1],'req2Id':sample['req2Id'],'req2':sample['req2'],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)  #Added AnnotationStatus as M 
                #logs.createAnnotationsFile(df_userAnnot)
                
                #Remove the selected sample from the original dataframe
                df_predictions.drop(index=indexValue,inplace=True)   
                df_predictions.reset_index(inplace=True,drop=True)
            else:
                print("All of unlabelled data is over")            
                    
                #df_manuallyAnnotated = pd.concat([df_manuallyAnnotated,df_userAnnot])
                
            iteration+=1
        
    #Remove all the extra columns. df now contains only combinations marked 'A'
    df_predictions=df_predictions[[req1,req2,label,annStatus]]#df_predictions[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    df_remaining = df_predictions
    df_remaining[annStatus] = ''
    #df_manuallyAnnotated=df_manuallyAnnotated[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    logs.writeLog("\n\nManually Annotated Combinations... "+str(len(df_predictions))+"Rows \n"+str(df_predictions[:10]))
    
    return df_userAnnot, df_remaining