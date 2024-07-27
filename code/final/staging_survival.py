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

class Staging_And_Survival:
    '''This class is for the staging and survival components of EMBER with methods meant to provide different aspects of lung cancer patients' evaluations.
    All code in the class is ORIGINAL.'''
    def __init__(self, file_path):
        self.file_path = file_path

    def load_split_datasets(self):
        '''This method loads in the Memorial Sloan Kettering (MSK) dataset, which has both staging and survival data for lung cancer patients.
        The mthod standardizes the data and drops any null values so that the models work without error. All code is ORIGINAL.'''
        df = pd.read_csv(file_path)
        df.drop(['Study ID', 'Patient ID', 'Sample ID'], axis=1, inplace=True)
        df.dropna(subset=['Fraction Genome Altered', 'Mutation Count', 'CT Size', 'Pathologic Stage', 'Relapse Free Status (Months)', 'Person Cigarette Smoking History Pack Year Value', 'Standardized uptake values'], inplace=True)

        input = df[['Fraction Genome Altered', 'Mutation Count', 'Relapse Free Status (Months)', 'Person Cigarette Smoking History Pack Year Value']]
        stage_target = df['Pathologic Stage']
        survival_target = df['Overall Survival Status']

        surv_lst = []
        for val in survival_target:
            surv_lst.append(val)

        targ_lst = []
        for targ in surv_lst:
            print(targ)
            if targ == '1:DECEASED':
                targ_lst.append(1)
            else:
                targ_lst.append(0)
        survival_target = pd.Series(targ_lst)

        X_train, X_test, y_s_train, y_s_test, y_ss_train, y_ss_test = train_test_split(input, stage_target, survival_target, train_size=0.75, random_state=49)
        return X_train, X_test, y_st_train, y_st_test, y_su_train, y_su_test

    def staging_qsvc(X_train, X_test, y_st_train, y_st_test):
        ''' This method implements a QSVC to predict the staging of patients' lung cancers using the MSK dataset. All code
        is ORIGINAL, though it does make use of the qiskit and time python modules. '''
        fm = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
        quantum_kernel = FidelityQuantumKernel(feature_map=fm)
        fm.entanglement = 'full'
        stage_qsvc = QSVC(quantum_kernel=quantum_kernel, probability=True, random_state=42)

        start = time.time()
        stage_qsvc.fit(X_train, y_st_train)
        training_time = time.time() - start

        start = time.time()
        st_preds = qsvc.predict(X_test)
        pred_time = time.time() - start

        print(classification_report(y_st_test, st_preds))
        return stage_qsvc, st_preds, training_time, pred_time

    def survival_qsvc(X_train, X_test, y_su_train, y_su_test):
        ''' This method implements a QSVC to predict the survival outcome of lung cancer patients using the MSK dataset. All code
        is ORIGINAL, though it does make use of the qiskit and time python modules. '''
        fm = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X', 'Y', 'Z'])
        quantum_kernel = FidelityQuantumKernel(feature_map=fm)
        fm.entanglement = 'full'
        stage_qsvc = QSVC(quantum_kernel=quantum_kernel, probability=True, random_state=42)

        start = time.time()
        stage_qsvc.fit(X_train, y_su_train)
        training_time = time.time() - start

        start = time.time()
        su_preds = qsvc.predict(X_test)
        pred_time = time.time() - start

        print(classification_report(y_su_test, su_preds))
        return survival_qsvc, su_preds, training_time, pred_time
#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#IMPLEMENTATION OF THE CONSTRUCTED CLASSES (All ORIGINAL code)
file_path = file_path = getcwd() + '/data/cBIO_Progression/progression_data.csv'
cls = Quantum_Implementation(file_path)
X_train, X_test, y_train, y_test = cls.load_split_datasets()

stage_qsvc, st_preds, st_training_time, st_pred_time = cls.qsvc_implmentation(X_train, X_test, y_train, y_test)
survival_qsvc, su_preds, su_training_time, su_pred_time = cls.qsvc_implmentation(X_train, X_test, y_train, y_test)


tt_mins = round(training_time/60)
print (f'Training Time: {tt_mins} min, {round((training_time/60-tt_mins)*60)} sec')

pt_mins = round(pred_time/60)
print (f'Prediction Time: {pt_mins} min, {round((pred_time/60-pt_mins)*60)} sec')
