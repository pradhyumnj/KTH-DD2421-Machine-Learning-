# imports

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Just reading the data
val = pd.read_csv('EvaluateOnMe.csv', dtype={'x5':str})
train = pd.read_csv('TrainOnMe.csv')

#Cleaning
train.x11[train.x11.loc[train.x11 == 'F'].index] = 'False'
train.x11[train.x11.loc[train.x11 == 'Tru'].index] = 'True'
train.x12[train.x12.loc[train.x12 == 'Flase'].index] = 'False'
train.x12[train.x12.loc[train.x12 == 'F'].index] = 'False

#Convert to numeric
train.x1 = pd.to_numeric(train.x1, errors = 'coerce')
train.x2 = pd.to_numeric(train.x2, errors = 'coerce')
train.x3 = pd.to_numeric(train.x3, errors = 'coerce')
train.x4 = pd.to_numeric(train.x4, errors = 'coerce')


#dropping junk data
train.drop(index = [487,488,489,490,639,737], inplace=True)

#Correcting names
train.y[164] = 'Dragspel'
train.y[243] = 'Serpent'
train.y[175] = 'Nyckelharpa'
train.x6[604] = 'Östra stationen'

#Outliers
train.x1[301] = np.NaN
train.x1[301] = train.x1.mean()
train.x1[721] = train.x1.mean()
train.x2[441] = np.NaN
train.x2[441] = train.x2.mean()

#x5
train.x5 = train.x5.apply(lambda x : 0 if x[0] == '-' else 1)
val.x5 = val.x5.apply(lambda x : 0 if x[0] == '-' else 1)

#x6
train['x6'] = train.x6.fillna('NA')
val['x6'] = val.x6.fillna('NA')

#x11 and x12
train.x12 = train.x12.apply(lambda x : 0 if x == 'False' else 1)
train.x11 = train.x11.apply(lambda x : 0 if x == 'False' else 1)
val.x12 = val.x12.apply(lambda x : int(x))
val.x11 = val.x11.apply(lambda x : int(x))

#x2
train.x2.fillna(train.x2.mean(), inplace = True)

#outliers
indices = set()
for col in ['x2','x3','x4','x7','x8','x9','x10']:
    temp = train.loc[(train[col] - train[col].mean()).abs() > 3.3*train[col].std()].index.values
    indices.update(set(temp))
train.drop(indices, inplace = True)

#cat to numeric
encoder = LabelEncoder()
train_X = train[list(set(train.columns) - {'y'})]
train_X['x6'] = encoder.fit_transform(train_X['x6'])
val['x6'] = encoder.fit_transform(val['x6'])
train_y = encoder.fit_transform(train.y)
val = val[list(set(train.columns) - {'y'})]

#model
qda = QuadraticDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis()
lr = LogisticRegression()
rf = RandomForestClassifier(max_depth = 9, n_jobs = -1, n_estimators = 400, min_samples_split=20, bootstrap = True, max_features = 'auto', ccp_alpha=0.003)
xgb = XGBClassifier(max_depth = 3,n_estimators = 150, learning_rate = 0.03, max_leaves = 20, n_jobs = -1)
classifiers = [('qda', qda), ('lda', lda), ('clf', rf)]    
clf = StackingClassifier(classifiers, final_estimator=xgb, cv=10, n_jobs = -1)
cross_validate(clf,train_X, train_y, cv = 6, n_jobs = -1)

#performance eval
def performace(clf,iterations = 20, split = 0.7, verbose = 0): 
    """returns test, train accuracy"""
    test = []
    train = []
    for i in tqdm(range(iterations)):
        a,b,c,d = train_test_split(train_X,train_y,train_size = split)
        clf.fit(a,c)
        test += [clf.score(b,d)]
        train += [clf.score(a,c)]
        if verbose == 1:
            print(test[i], train[i])
    return test, train

testa, traina = performace(clf,iterations = 50, split = 0.7, verbose = 1)


#predictions
clf.fit(train_X, train_y)
print('training performace =', clf.score(train_X, train_y))
predictions = encoder.inverse_transform(clf.predict(val))
with open('predictions.txt', 'w') as file:
    file.write('\n'.join(predictions))