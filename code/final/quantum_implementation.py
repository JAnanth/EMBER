# Necessary Module Imports
import pandas as pd
import numpy as np
import time
from os import getcwd, listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

class Quantum_Implementation:
    ''''''
    def __init__(self, study_name):
        self.study_name = study_name

    def load_split_datasets(self):
        ''' This method loads in the data that has already been feature selected. The code is all ORIGINAL, though it does make use
        of the Python Pandas podule in addition to the os module.'''
        file_path = f'{getcwd()}/data/Manipulated/{self.study_name}/'

        X_train = pd.read_csv(file_path + 'qc/qc_X_train.csv')
        X_test = pd.read_csv(file_path + 'qc/qc_X_test.csv')
        y_train = pd.read_csv(file_path + 'split/y_train.csv')
        y_test = pd.read_csv(file_path + 'split/y_test.csv')
        return X_train, X_test, y_train['target'], y_test['target']

    def classical_implementation(self, X_train, X_test, y_train, y_test):
        '''This method constructs and implements the classical support vector classifier (SVC) baseline model. All code is ORIGINAL, but does
        use the scikit learn module. '''
        start = time.time()
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train, y_train)
        classical_svc_duration = time.time() - start

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))
        return y_pred

    def qsvc_implementation(X_train, X_test, y_train, y_test):
        ''' This method implements the first quantum classifier: the quantum support vector classifier. This classifier is the one
        that EMBER uses. All code in this method is original, although several classes of the qiskit python module are implemented. '''
        fm = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
        quantum_kernel = FidelityQuantumKernel(feature_map=fm)
        fm.entanglement = 'full'
        qsvc = QSVC(quantum_kernel=quantum_kernel)

        start = time.time()
        qsvc.fit(X_train, y_train)
        training_time = time.time() - start

        start = time.time()
        y_pred = qsvc.predict(X_test)
        pred_time = time.time() - start

        print(classification_report(y_test, y_pred))
        return qsvc, y_pred, training_time, pred_time


    def vqc_implementation(self, X_train, X_test, y_train, y_test):
        ''' This method is part of the quantum classifier comparitive analysis. The code to initialize the variational quantum classifier (VQC)
        was pulled from the Qiskit documentation website. The code was modified to fit the purposes of EMBER, but the code was NOT original.'''

        num_features = X_train.shape[1]

        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        feature_map.decompose().draw(output="mpl", fold=20)

        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        ansatz.decompose().draw(output="mpl", fold=20)

        vqc = VQC(
            sampler=Sampler(),
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=3),
        )

        start = time.time()
        vqc.fit(X_train, y_train)
        training_time = time.time() - start

        start = time.time()
        y_pred = vqc.predict(X_test)
        pred_time = time.time() - start

        print(classification_report(y_test, y_pred))
        return vqc, y_pred, training_time, pred_time

    def qnn_implementation(self, X_train, X_test, y_train, y_test):
        ''' Similar to the vqc_implmentation method, this method was part of the quantum classifier comparative analysis. The code was also pulled
        from the Qiskit documentation website and is thus NOT original. It was modified from the pulled code to fit the purposes of EMBER.'''

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
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=15),  # Set max iterations here
            callback=callback_graph,
        )

        start = time.time()
        classifier.fit(X_train, y_train)
        training_time = time.time() - start

        start2 = time.time()
        preds = classifier.predict(X_test)
        pred_time = time.time()-start2

        print(classification_report(preds, y_test))
        return NeuralNetworkClassifier, preds, training_time, pred_time


#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#IMPLEMENTATION OF THE CONSTRUCTED CLASSES (All ORIGINAL code)
cls = Quantum_Implementation('GSE137140')
X_train, X_test, y_train, y_test = cls.load_split_datasets()
y_pred = cls.classical_implementation(X_train, X_test, y_train, y_test)
qsvc, qsvc_y_pred, training_time, pred_time = cls.qsvc_implmentation(X_train, X_test, y_train, y_test)
vqc, vqc_y_pred, training_time, pred_time = cls.qsvc_implmentation(X_train, X_test, y_train, y_test)
qnn, qnn_y_pred, training_time, pred_time = cls.qsvc_implmentation(X_train, X_test, y_train, y_test)

tt_mins = round(training_time/60)
print (f'Training Time: {tt_mins} min, {round((training_time/60-tt_mins)*60)} sec')

pt_mins = round(pred_time/60)
print (f'Prediction Time: {pt_mins} min, {round((pred_time/60-pt_mins)*60)} sec')
