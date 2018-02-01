
'''
Updated: February 1, 2018
Written by: Jeeheh Oh

Purpose: Trains final logistic regression model on complete training set using the optimal C value.

How to use:
-- a. Complete reference.py
-- b. Run

Saved Outputs: 
-- auc_cv: Cross validation AUC, complete for all values of c, shape = len(c_param) x nfolds
-- c: optimal hyperparameter c value
-- clf: final classifier
-- df, dftr: Data frame containing columns for: eid, day, rawest(raw score from logistic regression), label, rollingscore (the rolling average of the raw score) for test and training. Data is at the day level of granularity.
-- df2,dftr2: Data frame containing eid, label and rollingscore for test and training. Data is at the admission level. Maximum rolling score is kept for each admission. 
'''

import numpy as np
import pandas as pd
import reference as reference
import pickle
import sys
from sklearn import linear_model
from sklearn import metrics
from scipy.sparse import csr_matrix


pathout,randseed,c_param,nfolds=reference.returnVars()
np.random.seed(randseed)


# Import data
xtrain,ytrain,xtest,ytest,eid_train,eid_test,day_train,day_test=maggie.loadData()


# Import data: CV
auc_cv=np.empty((len(c_param),nfolds))
auc_cv[:]=np.NAN

auc_train_ef=[]
auc_test_ef=[]

for i in range(len(c_param)):
    zauc_cv,zauc_train_ef,zauc_test_ef,_,_=pickle.load(open(pathout+"cv_c"+str(i)+"_auc.pickle","rb"))
        #auc_cv,auc_train_ef,auc_test_ef,c,c_param
    auc_cv[i,:]=zauc_cv[i,:]
    auc_train_ef.append(zauc_train_ef)
    auc_test_ef.append(zauc_test_ef)
    
    
# Find best c
c=c_param[np.argmax(np.mean(auc_cv,axis=1))]


# Learn Model 
clf=linear_model.LogisticRegression(penalty='l2',class_weight='balanced',C=c)
zx,zy=reference.subsample(eid_train,day_train,xtrain,ytrain)
clf.fit(zx,zy)


# Create Output Dataframe
idx=np.where(clf.classes_==1)[0]
test_rawest=clf.predict_proba(xtest)[:,idx]
train_rawest=clf.predict_proba(xtrain)[:,idx]
df=pd.DataFrame({'encounterID':eid_test,'day':day_test     ,'rawest':test_rawest.reshape((len(test_rawest),)),'label':ytest.reshape((len(ytest),))})
dftr=pd.DataFrame({'encounterID':eid_train,'day':day_train     ,'rawest':train_rawest.reshape((len(train_rawest),)),'label':ytrain.reshape((len(ytrain),))})


#Smoothing 
df=df.sort_values(['encounterID','day'],ascending=[1,1])
df['csum']=df['rawest'].groupby([df['encounterID']]).cumsum()
df['rollingscore']=df['csum']/df['day']

dftr=dftr.sort(['encounterID','day'],ascending=[1,1])
dftr['csum']=dftr['rawest'].groupby([dftr['encounterID']]).cumsum()
dftr['rollingscore']=dftr['csum']/dftr['day']

df2=df.loc[df.day>=2,['rollingscore','label','encounterID']].groupby([df['encounterID']]).max()
dftr2=dftr.loc[dftr.day>=2,['rollingscore','label','encounterID']].groupby([dftr['encounterID']]).max()

# Save Data
with open(pathout+'learn_model.pickle','wb') as f:
    pickle.dump([auc_cv, c, clf, df, df2,dftr,dftr2],f)