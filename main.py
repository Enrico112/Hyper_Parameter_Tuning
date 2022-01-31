from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# create df
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
# transform species from int to cat
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# split train and tests dfs
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.2)

# fit support vector machine
model = SVC(kernel='rbf',C=30,gamma='auto')
model.fit(X_train,y_train)
model.score(X_test,y_test)

# try different parameters and splits
cross_val_score(SVC(kernel='linear',C=10,gamma='auto'), iris.data, iris.target, cv=10)
cross_val_score(SVC(kernel='rbf',C=10,gamma='auto'), iris.data, iris.target, cv=10)
cross_val_score(SVC(kernel='rbf',C=20,gamma='auto'), iris.data, iris.target, cv=10)

# do the same with for loop
kernels = ['rbf', 'linear']
C = [1,10,20]
avg_score = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(SVC(kernel=kval,C=cval,gamma='auto'), iris.data, iris.target, cv=10)
        avg_score[kval+'_'+str(cval)] = np.average(cv_scores)

avg_score

# GridSearchCV does the same thing as the above for loop
gscv = GridSearchCV(SVC(gamma='auto'),{
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=10, return_train_score=False)
# n_iter = set number of iteration with random params among inputs, to offset for lengthy computations, and still records the best score
gscv.fit(iris.data,iris.target)
gscv.cv_results_

# export gscv to df
df = pd.DataFrame(gscv.cv_results_)

# see other properties
dir(gscv)
gscv.best_score_
gscv.best_params_

#  dict with model and params
model_params = {
    'svm' : {
        'model' : SVC(gamma='auto'),
        'params' : {
            'C' : [1,10,20],
            'kernel' : ['rbf','linear']
        }
    },
    'random forest' : {
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [1,5,10]
        }
    },
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear',multi_class='auto'),
        'params' : {
            'C' : [1,5,10]
        }
    }
}

scores = []
# loop over dict
for model_name, mp in model_params.items():
    gscv = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    gscv.fit(iris.data,iris.target)
    scores.append({
        'model': model_name,
        'best_score': gscv.best_score_,
        'best_params': gscv.best_params_
    })

# convert to df
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

