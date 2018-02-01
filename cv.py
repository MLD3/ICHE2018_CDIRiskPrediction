'''
Updated: February 1, 2018
Written by: Jeeheh Oh

Purpose: Performs stratified, kfold, cross validation on the training set in order to determine optimal value for C hyperparameter that controls L2 regularization strength in logistic regression.  

How to use:
-- a. Complete reference.py
-- b. Run code once for each potential hyperparameter value. (eg input into terminal: python cv.py cindex=0, python cv.py cindex=1,..., python cv.py cindex=14). This doesn't have to be done via the terminal. The cindex variable can be changed in the ## Set Variables ## section of this code. Results will be aggregated in model.py. 

Saved Outputs:
-- auc_cv: Cross Validation AUC, shape = len(c_param) x nfolds. Only the column corresponding to c is completed.
-- auc_train_ef: Training AUC, scalar
-- auc_test_ef: Test AUC, scalar
-- c: C parameter used
-- c_param: List of C parameters in grid search
'''

import numpy as np
import pandas as pd
import pickle
import sys
import reference as reference
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix


## Set Variables ##
pathout,randseed,c_param,nfolds=reference.returnVars()
np.random.seed(randseed)

# Which C parameter tested in this run
cindex=0 #cindex[0-14]

# Used to set any variables from terminal
# eg: python cv.py cindex=1
list=sys.argv[1:]
for i in list:
    exec(i)
    
############################################################################################
############################################################################################

# Import data 
xtrain,ytrain,xtest,ytest,eid_train,eid_test,day_train,day_test=reference.loadData()

 
# Returns AUC score using the maximum estimate per patient.
def scorer(estimator,X,y,eid):
    proba=estimator.decision_function(X)
    df=pd.DataFrame({'eid':eid,'proba':proba,'label':y})
    df=df.groupby(df['eid'],as_index=False).max()
    fpr, tpr, thresholds=metrics.roc_curve(df.label,df.proba,pos_label=1)            
    return metrics.auc(fpr,tpr)


# Stratified K Fold, Clustered on Patient
from sklearn.model_selection import StratifiedKFold
dfskf=pd.DataFrame({'eid':eid_train,'y':ytrain}) #create dataframe of eid and y
dfskf.drop_duplicates(subset='eid',inplace=True) #make it unique by eid
skf=StratifiedKFold(n_splits=nfolds) 
skf.get_n_splits(dfskf.eid,dfskf.y)

index=1
dfskf['fold']=np.ones(dfskf.shape[0]) #'fold' indicates which fold eid is in
for _,test_index in skf.split(dfskf.eid,dfskf.y):
    dfskf.iloc[test_index,dfskf.columns.get_loc("fold")]=index
    index=index+1

eidyear_train=pd.DataFrame({'eid':eid_train}) #now translate the 'fold' to indicies
eidyear_train['index']=eidyear_train.index
eidyear_train=eidyear_train.merge(dfskf,how='left',on='eid',indicator=True)

                
# CV
c=c_param[cindex]
auc_cv=np.empty((len(c_param),nfolds))
auc_cv[:]=np.NAN

clf=linear_model.LogisticRegression(penalty='l2',class_weight='balanced',C=c)

for foldindex, year in enumerate(np.arange(nfolds)+1):
    
    #List of Index Numbers
    cvtrain_idx=eidyear_train.loc[eidyear_train.fold!=year,:]
    cvtest_idx=eidyear_train.loc[eidyear_train.fold==year,:]
    
    cvtrain_idx2=np.array(cvtrain_idx['index'],dtype=pd.Series).tolist()
    cvtest_idx2=np.array(cvtest_idx['index'],dtype=pd.Series).tolist()
    zx,zy=reference.subsample(eid_train[cvtrain_idx2],day_train[cvtrain_idx2],xtrain[cvtrain_idx2,:],ytrain[cvtrain_idx2])
    clf.fit(zx,zy)
    auc_cv[cindex,foldindex]=scorer(clf,xtrain[cvtest_idx2,:],ytrain[cvtest_idx2],eid_train[cvtest_idx2])

# Optional: Calculates the train and test AUC in addition to the CV AUC.
clf=linear_model.LogisticRegression(penalty='l2',class_weight='balanced',C=c)
zx,zy=reference.subsample(eid_train,day_train,xtrain,ytrain)
clf.fit(zx,zy)    
auc_train_ef=scorer(clf,xtrain,ytrain,eid_train)
auc_test_ef=scorer(clf,xtest,ytest,eid_test)

pickle.dump([auc_cv,auc_train_ef,auc_test_ef,c,c_param], open(pathout+"cv_c"+str(cindex)+"_auc.pickle", "wb" ) )
