!pip install qiskit_algorithms
!pip install qiskit-ibmq-provider



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
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA, AQGD
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit import Aer
from sklearn.metrics import classification_report, auc

from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, roc_auc_score, log_loss
import time

file_path = getcwd() + '/data/cBIO_Progression/progression_data.csv'
df = pd.read_csv(file_path)
df.columns
df.drop(['Study ID', 'Patient ID', 'Sample ID'], axis=1, inplace=True)
df.dropna(subset=['Fraction Genome Altered', 'Mutation Count', 'CT Size', 'Pathologic Stage', 'Relapse Free Status (Months)', 'Person Cigarette Smoking History Pack Year Value', 'Standardized uptake values'], inplace=True)
df
input = df[['Fraction Genome Altered', 'Mutation Count', 'Relapse Free Status (Months)', 'Person Cigarette Smoking History Pack Year Value']]
stage_target = df['Pathologic Stage']
survival_target = df['Overall Survival Status']
survival_target
lst
lst = []
for val in survival_target:
    lst.append(val)
lst1 = []
for targ in lst:
    print(targ)
    if targ == '1:DECEASED':
        lst1.append(1)
    else:
        lst1.append(0)
survival_target = pd.Series(lst1)

X_train, X_test, y_s_train, y_s_test, y_ss_train, y_ss_test = train_test_split(input, stage_target, survival_target, train_size=0.75, random_state=49)

stage_clf = SVC()
stage_clf.fit(X_train, y_s_train)
accuracy_score(stage_clf.predict(X_test), y_s_test)

surv_clf = SVC()
surv_clf.fit(X_train, y_ss_train)
accuracy_score(surv_clf.predict(X_test), y_ss_test)


fm = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
quantum_kernel = FidelityQuantumKernel(feature_map=fm)
fm.entanglement = 'full'
stage_qsvc = QSVC(quantum_kernel=quantum_kernel, probability=True, random_state=49)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# We now define a two qubit unitary as defined in [3]
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


# Let's draw this circuit and see what it looks like
params = ParameterVector("θ", length=3)
circuit = conv_circuit(params)
circuit.draw("mpl", style="clifford")

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


circuit = conv_layer(4, "θ")
circuit.decompose().draw("mpl")


start = time.time()
stage_qsvc.fit(X_train, y_s_train)
accuracy_score(stage_qsvc.predict(X_test), y_s_test)
auc(stage_qsvc.predict(X_test), y_s_test)
run1 = time.time() - start
f'{round(run1/60)} minutes, {round((run1/60-round(run1/60))*60)} seconds'



s2 = time.time()
surv_qsvc = QSVC(quantum_kernel=quantum_kernel, probability=True, random_state=49)
surv_qsvc.fit(X_train, y_ss_train)



preds = stage_qsvc.predict(X_test)
preds = np.zeros(132)
accuracy_score(preds, y_ss_test)
run2 = time.time() - s2
f'{round(run2/60)} minutes, {round((run2/60-round(run2/60))*60)} seconds'
