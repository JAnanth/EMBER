import pandas as pd
import numpy as np
from os import getcwd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier as NNC

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

optimizer = COBYLA(maxiter=3)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        print(len(objective_func_vals))

vqc = VQC(
    sampler=Sampler(),
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=3),
)

start = time.time()
vqc.fit(qc_X_train, y_train)
elapsed = time.time() - start
elapsed/60

preds = vqc.predict(qc_X_test)
accuracy_score(y_test, preds)
balanced_accuracy_score(y_test, preds)
