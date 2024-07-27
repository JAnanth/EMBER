import pandas as pd
import numpy as np
from os import getcwd
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, ADAM
from qiskit.primitives import Sampler
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms.classifiers import VQC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

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
qc_X_train = qc_X_train[qc_X_train.columns[0:8]]
qc_X_test = qc_X_test[qc_X_test.columns[0:8]]

qc = QNNCircuit(num_qubits=8)
e_qnn = EstimatorQNN(circuit=qc)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    print(len(objective_func_vals))


e_classifier = NeuralNetworkClassifier(
    e_qnn, optimizer=COBYLA(maxiter=5), callback=callback_graph
)
e_classifier



start = time.time()
e_classifier.fit(qc_X_train, y_train)
elapsed = time.time() - start
elapsed/60

preds = e_classifier.predict(qc_X_test)
accuracy_score(y_test, preds)
