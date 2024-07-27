import pandas as pd
from os import getcwd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier

def load_data(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/expression_matrix.csv'
    df = pd.read_csv(file_path)
    return df

data = load_data('GSE137140')
data
#Create necessary data variables
expression_data = data.copy()
del expression_data['target']

target_data = data['target']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(expression_data, target_data, train_size=0.7, random_state=3)
y_train = pd.Series(y_train.reset_index()['target'])
y_test = pd.Series(y_test.reset_index()['target'])

file_path = getcwd() + '/data/Manipulated/GSE137140/'
data.to_csv(file_path + 'full/df.csv', index=False)
X_train.to_csv(file_path + 'split/X_train.csv', index=False)
X_test.to_csv(file_path + 'split/X_test.csv', index=False)
y_train.to_csv(file_path + 'split/y_train.csv', index=False)
y_test.to_csv(file_path + 'split/y_test.csv', index=False)

#Implementation of algorithms to make predictions
preds_alg = Perceptron()
preds_alg.fit(X_train, y_train)
preds = preds_alg.predict(X_test)
cfs = preds_alg.coef_

preds
cfs
#Analysis of Results
def get_stats(y_test, preds):
    #Get basic stats
    acc = accuracy_score(y_test, preds)
    num_wrong = round(len(X_test)*(1-acc))
    #Get true pos, false pos, etc. for calculating further stats
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(X_test)):
        if y_test[i] == 0 and preds[i] == 1:
            fp += 1
        elif y_test[i] == 1 and preds[i] == 0:
            fn += 1
        elif y_test[i] == 0 and preds[i] == 0:
            tn += 1
        else:
            tp += 1
    #Put all stats together for return val
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    data = {'Accuracy (%)': [round(acc*100,3)],
            'Precision (%)': [round(prec*100,3)],
            'Recall (%)': [round(recall*100,3)],
            'F1_Score': [round((100*(2*prec*recall)/(prec+recall)),3)]
            }
    stats_df = pd.DataFrame(data)
    #Return the stats dataframe and other info incase needed
    return stats_df, cfs, fp, fn, tp, tn

results_df, cfs, fp, fn, tp, tn = get_stats(y_test, preds)
results_df

def feature_importance_analysis(limit):
    #Determine feature importance coefficients
    cfs = np.sort(preds_alg.coef_.flatten())
    #Implement IQR Outlier rule to minimize features of interest
    q3, q1 = np.percentile(cfs, [75 ,25])
    iqr = q3 - q1
    u_bound = q3 + limit*iqr
    l_bound = q1 - limit*iqr
    #Implement IQR Rule to select upper and lower bound outliers
    l_out = []
    u_out = []
    cfs_mutable = cfs.tolist()
    for cf in cfs_mutable:
        if cf < l_bound:
            l_out.append(cf)
        elif cf > u_bound:
            u_out.append(cf)
    return l_out, u_out

lower_outliers, upper_outliers = feature_importance_analysis(3)
outliers = lower_outliers + upper_outliers
outliers

cfs1 = cfs[0].tolist()
lst = []
for i in range(len(cfs1)):
    if cfs1[i] in outliers:
        lst.append(i)
pd.DataFrame(lst).to_csv(getcwd() + '/data/Manipulated/GSE137140/outliers/indices.csv', index=False)
lst
qc_X_train = pd.DataFrame()
qc_X_test = pd.DataFrame()
for index in lst:
    qc_X_train[X_train.columns[index]] = X_train[X_train.columns[index]]
    qc_X_test[X_test.columns[index]] = X_test[X_test.columns[index]]

alg = Perceptron()
alg.fit(qc_X_train, y_train)
preds = alg.predict(qc_X_test)
accuracy_score(preds, y_test)
lst1 = []
for ind in lst:
    lst1.append(data.columns[ind])
lst1
file_path
qc_X_train.to_csv(file_path + '/qc/qc_X_train.csv', index=False)
qc_X_test.to_csv(file_path + '/qc/qc_X_test.csv', index=False)
