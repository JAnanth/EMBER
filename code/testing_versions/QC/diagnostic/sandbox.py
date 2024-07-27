!pip install qiskit-ibmq-provider
!pip install qiskit-aer

from qiskit import IBMQ

IBMQ.save_account('b6ea37d76d6622154b91cf302fefe57459c07451f3009dcbcee5e6086f1eda0e86274e6e185e052ea5cfb5f03a2789a9ac774df2cb0023bead0a6ea3bb1786fd')
IBMQ.load_account()

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from os import getcwd
import pandas as pd
from sklearn.metrics import accuracy_score


def load_split_datasets(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/'
    df = pd.read_csv(file_path + 'full/df.csv')
    outliers_lst = pd.read_csv(file_path + '/outliers/indices.csv')
    X_train = pd.read_csv(file_path + 'qc/qc_X_train.csv')
    X_test = pd.read_csv(file_path + 'qc/qc_X_test.csv')
    y_train = pd.read_csv(file_path + 'split/y_train.csv')
    y_test = pd.read_csv(file_path + 'split/y_test.csv')
    return df, outliers_lst

data, out_idxs = load_split_datasets('GSE137140')
out_idxs = np.array(out_idxs['0']).tolist()

data1 = data.sample(300)
expression_data = data1[data1.columns[0:len(data1.columns)-1]]
target_data = np.array(pd.Series(data1['target'].reset_index()['target']))

X = SelectKBest(f_classif, k=10).fit_transform(expression_data, target_data)

X_train, X_test, y_train, y_test = train_test_split(X, target_data, train_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, log_loss
import time

start = time.time()
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
classical_svc_duration = time.time() - start

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))



!pip install qiskit_algorithms


from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA, AQGD
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit import Aer
from sklearn.metrics import classification_report


feature_maps = [
    ZFeatureMap(feature_dimension=X_train.shape[1], reps=2),
    ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear'),
    PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
]
entanglements = ['linear', 'circular', 'full']

# for feature_map in feature_maps:
#     for entanglement in entanglements:
#         feature_map.entanglement = entanglement
#         quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
#
#         qsvc = QSVC(quantum_kernel=quantum_kernel)
#
#         start = time.time()
#         qsvc.fit(X_train, y_train)
#         y_pred = qsvc.predict(X_test)
#         run_time = time.time() - start
#
#         print(f'Feature Map: {type(feature_map).__name__}, Entanglement: {entanglement}')
#         print(classification_report(y_test, y_pred))
#         mins = round(run_time/60)
#         print(f'Run Time (min): {mins+ round((run_time/60-mins)*60)}')
#         print("-" * 55)

fm = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
quantum_kernel = FidelityQuantumKernel(feature_map=fm)
fm.entanglement = 'full'
fm.decompose().draw(output="mpl", fold=20)
pqsvc = PegasosQSVC(quantum_kernel=quantum_kernel)

start = time.time()
pqsvc.fit(X_train, y_train)
training_time = time.time() - start

tt_mins = round(training_time/60)
f'Training Time: {tt_mins} min, {round((training_time/60-tt_mins)*60)} sec'


start = time.time()
y_pred = pqsvc.predict(X_test)
pred_time = time.time() - start

pt_mins = round(pred_time/60)
f'Prediction Time: {pt_mins} min, {round((pred_time/60-pt_mins)*60)} sec'


f'Accuracy Score (%): {accuracy_score(y_test, y_pred)}'
print(classification_report(y_test, y_pred))

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

results_df, cfs, fp, fn, tp, tn = get_stats(y_test, y_pred)
results_df






def load_split_datasets(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/split/'
    X_train = pd.read_csv(file_path + 'X_train.csv')
    X_test = pd.read_csv(file_path + 'X_test.csv')
    y_train = pd.read_csv(file_path + 'y_train.csv')
    y_test = pd.read_csv(file_path + 'y_test.csv')
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_split_datasets('GSE137140')
y_train = np.array(pd.Series(y_train.reset_index()['target']))
y_test = np.array(pd.Series(y_test.reset_index()['target']))

def load_qc_datasets(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/qc/'
    train = pd.read_csv(file_path + 'qc_X_train.csv')
    test = pd.read_csv(file_path + 'qc_X_test.csv')
    return train, test

qc_X_train, qc_X_test = load_qc_datasets('GSE137140')
qc_X_train = qc_X_train[qc_X_train.columns[0:12]]
qc_X_test = qc_X_test[qc_X_test.columns[0:12]]

num_features = qc_X_train.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)
