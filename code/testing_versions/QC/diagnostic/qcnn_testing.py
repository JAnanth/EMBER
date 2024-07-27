import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tab2img.converter import Tab2Img

def load_split_datasets(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/'
    df = pd.read_csv(file_path + 'full/df.csv')
    outliers_lst = pd.read_csv(file_path + '/outliers/indices.csv')
    X_train = pd.read_csv(file_path + 'split/X_train.csv')
    X_test = pd.read_csv(file_path + 'split/X_test.csv')
    y_train = pd.read_csv(file_path + 'split/y_train.csv')
    y_test = pd.read_csv(file_path + 'split/y_test.csv')
    return df, outliers_lst

data, out_idxs = load_split_datasets('GSE137140')
out_idxs = np.array(out_idxs['0']).tolist()
expression_data = data[data.columns[0:len(data.columns)-1]]
target_data = np.array(pd.Series(data['target'].reset_index()['target']))

input = pd.DataFrame()
for index in out_idxs:
    input[expression_data.columns[index]] = expression_data[expression_data.columns[index]]

model = Tab2Img()
img_exp = model.fit_transform(np.array(input), target_data)

lst = []
for i in range(len(img_exp)):
    lst.append(img_exp[i].ravel())

X_train, X_test, y_train, y_test = train_test_split(np.array(lst), target_data, train_size=0.7, random_state=3)

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
circuit.draw("mpl")


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

circuit = conv_layer(3, "θ")
circuit.decompose().draw("mpl")

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


params = ParameterVector("θ", length=3)
circuit = pool_circuit(params)
circuit.draw("mpl")

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

#
# sources = [0, 1]
# sinks = [2, 3]
# circuit = pool_layer(sources, sinks, "θ")
# circuit.decompose().draw("mpl")

feature_map = ZFeatureMap(16)
feature_map.decompose().draw("mpl")

feature_map = ZFeatureMap(16)

ansatz = QuantumCircuit(16, name="Ansatz")

# First Convolutional Layer
ansatz.compose(conv_layer(16, "с1"), list(range(16)), inplace=True)

# First Pooling Layer
ansatz.compose(pool_layer([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15], "p1"), list(range(16)), inplace=True)

# Second Convolutional Layer
ansatz.compose(conv_layer(12, "c2"), list(range(4, 16)), inplace=True)

# Second Pooling Layer
ansatz.compose(pool_layer([0,1,2,3,4,5], [6,7,8,9,10,11], "p2"), list(range(4, 16)), inplace=True)

# Third Convolutional Layer
ansatz.compose(conv_layer(10, "c3"), list(range(6, 16)), inplace=True)

# Third Pooling Layer
ansatz.compose(pool_layer([0,1,2,3,4], [5,6,7,8,9], "p3"), list(range(6, 16)), inplace=True)

# Fourth Convolutional Layer
ansatz.compose(conv_layer(8, "c4"), list(range(8, 16)), inplace=True)

# Fourth Pooling Layer
ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p4"), list(range(8, 16)), inplace=True)

# Fifth Convolutional Layer
ansatz.compose(conv_layer(6, "c5"), list(range(10, 16)), inplace=True)

# Fifth Pooling Layer
ansatz.compose(pool_layer([0,1,2], [3,4,5], "p5"), list(range(10, 16)), inplace=True)

# Sixth Convolutional Layer
ansatz.compose(conv_layer(4, "c6"), list(range(12, 16)), inplace=True)

# Sixth Pooling Layer
ansatz.compose(pool_layer([0,1], [1,2], "p6"), list(range(12, 16)), inplace=True)

# Sevent Convolutional Layer
ansatz.compose(conv_layer(2, "c7"), list(range(14, 16)), inplace=True)

# Sevent Pooling Layer
ansatz.compose(pool_layer([0], [1], "p7"), list(range(14, 16)), inplace=True)

# Combining the feature map and ansatz
circuit = QuantumCircuit(16)
circuit.compose(feature_map, range(16), inplace=True)
circuit.compose(ansatz, range(16), inplace=True)

# observable = SparsePauliOp.from_list([("Z" + "I" * 9, 1)])

# we decompose the circuit for the QNN to avoid additional data copying
qnn = EstimatorQNN(
    circuit=circuit.decompose(),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)


circuit.draw('mpl')

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    print(len(objective_func_vals))

classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=15),  # Set max iterations here
    callback=callback_graph,
)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
start = time.time()
classifier.fit(X_train, y_train)
elapsed = time.time() - start
elapsed/60

preds = classifier.predict(X_test)
accuracy_score(y_test, preds)
balanced_accuracy_score(y_test, preds)
