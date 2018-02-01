import pandas as pd

# Fill out desired variable settings
def returnVars():
    #Location where you want output to be saved to
    pathout='/../' 
    
    #subsample random seed
    randseed= #eg: 11
    
    # C parameter Grid Search Values
    c_param=[5**(-14),5**(-13),5**(-12),5**(-11),5**(-10),5**(-9),5**(-8),5**(-7),5**(-6),5**(-5),5**(-4),5**(-3),5**(-2),5**(-1),5**(0)]

    # Number of folds for the Cross Validation
    nfolds=5
    
    return pathout,randseed,c_param,nfolds

# Complete Function so that it returns the appropriate data variables
def loadData():
    # m = dimensionality of features. eg: If our observations consist only of Heart Rate and Blood Pressure, m=2.
    # n_train, n_test = number of patient days we have data for in the training and test set. eg: If our training dataset consists of: (Patient A, LOS=3), (Patient B, LOS=2). Then n_train=5.
    # eid = Admission ID/Encounter ID. Each admission should have a unique ID regardless if it is a returning patient. 
    # day = Day of observation, in relation to the admission day. eg: 1, 2, 3, 4, etc.
    
    xtrain= #Data: scipy sparse csr matrix of size n_train xm
    ytrain= #Labels: numpy array of size n_train 
    xtest= #Data: scipy sparse csr matrix of size n_test xm
    ytest= #Labels: numpy array of size n_test 
    eid_train= #eids corresponding to each row of xtrain: numpy array of size n_train
    eid_test= #eids corresponding to each row of xtest: numpy array of size n_test 
    day_train= #day corresponding to each row of xtrain: numpy array of size n_train 
    day_test= #day corresponding to each row of xtest: numpy array of size n_test 

    return xtrain,ytrain,xtest,ytest,eid_train,eid_test,day_train,day_test


# Returns subsampled version of data, where for eg, if size=3, each patient is only represented 3 times in the dataset, aka only 3 days from that patient are kept in the subsampled dataset. 
# Set desired subsample size

def subsample(eid,day,x,y):
    df=pd.DataFrame({'eid':eid,'day':day})
    df['index']=df.index

    size= #eg: 3
    replace=True
    subspl=lambda obj: obj.loc[np.random.choice(obj.index,size,replace),:]
    df=df.groupby('eid',as_index=False).apply(subspl)

    x=x[df['index'].tolist(),:]
    y=y[df['index'].tolist()]
    
    return x,y