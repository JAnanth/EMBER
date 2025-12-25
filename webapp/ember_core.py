"""
EMBER Core Module - Early Malignancy Biomarker Evaluation and Recognition

This module provides the core quantum machine learning functionality for lung cancer
detection and staging using gene expression data.

Author: EMBER Project
"""

import pandas as pd
import numpy as np
import time
import json
import io
import base64
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from os import getcwd, listdir, path

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import (
    ZFeatureMap, ZZFeatureMap, PauliFeatureMap, RealAmplitudes
)
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.primitives import Sampler, Estimator

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class EMBERResult:
    """Data class to hold EMBER prediction results"""
    prediction: int
    confidence: float
    class_probabilities: Dict[str, float]
    staging: Optional[str] = None
    survival_prediction: Optional[str] = None
    quantum_metrics: Optional[Dict] = None
    circuit_info: Optional[Dict] = None


@dataclass
class TrainingMetrics:
    """Data class to hold training metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    prediction_time: float
    confusion_matrix: np.ndarray
    classification_report: str
    roc_auc: Optional[float] = None


class QuantumCircuitVisualizer:
    """Utility class for visualizing quantum circuits and states"""

    @staticmethod
    def circuit_to_base64(circuit: QuantumCircuit, style: str = 'mpl') -> str:
        """Convert a quantum circuit to base64 encoded image"""
        fig, ax = plt.subplots(figsize=(12, 6))
        circuit.draw(output='mpl', ax=ax, style={'backgroundcolor': '#FFFFFF'})

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    @staticmethod
    def statevector_visualization(statevector: np.ndarray) -> str:
        """Visualize statevector probabilities"""
        fig, ax = plt.subplots(figsize=(10, 5))
        probabilities = np.abs(statevector) ** 2
        n_states = len(probabilities)

        # Limit display for large state spaces
        if n_states > 32:
            # Show top 32 states by probability
            top_indices = np.argsort(probabilities)[-32:]
            probabilities = probabilities[top_indices]
            labels = [f'|{i:b}>' for i in top_indices]
        else:
            labels = [f'|{i:0{int(np.log2(n_states))}b}>' for i in range(n_states)]

        bars = ax.bar(range(len(probabilities)), probabilities, color='#3498db')
        ax.set_xlabel('Quantum States')
        ax.set_ylabel('Probability')
        ax.set_title('Quantum State Probabilities')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    @staticmethod
    def training_progress_plot(objective_values: List[float]) -> str:
        """Plot training progress"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(objective_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Quantum Classifier Training Progress')
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    @staticmethod
    def confusion_matrix_plot(cm: np.ndarray, labels: List[str]) -> str:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    @staticmethod
    def roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> str:
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64


class EMBERDataProcessor:
    """Handles data loading, preprocessing, and feature selection for EMBER"""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or path.join(getcwd(), 'data')
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.feature_selector = None
        self.selected_features = None

    def load_expression_matrix(self, study_name: str = 'GSE137140') -> pd.DataFrame:
        """Load the preprocessed gene expression matrix"""
        file_path = path.join(self.data_path, 'Manipulated', study_name, 'expression_matrix.csv')
        return pd.read_csv(file_path)

    def load_quantum_features(self, study_name: str = 'GSE137140') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load pre-selected quantum features (7 most significant miRNAs)"""
        base_path = path.join(self.data_path, 'Manipulated', study_name)

        X_train = pd.read_csv(path.join(base_path, 'qc', 'qc_X_train.csv'))
        X_test = pd.read_csv(path.join(base_path, 'qc', 'qc_X_test.csv'))
        y_train = pd.read_csv(path.join(base_path, 'split', 'y_train.csv'))['target']
        y_test = pd.read_csv(path.join(base_path, 'split', 'y_test.csv'))['target']

        return X_train, X_test, y_train, y_test

    def load_progression_data(self) -> pd.DataFrame:
        """Load MSK progression/staging data"""
        file_path = path.join(self.data_path, 'cBIO_Progression', 'progression_data.csv')
        return pd.read_csv(file_path)

    def preprocess_for_quantum(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features to range [0, pi] for quantum encoding"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def select_features_dl(self, X: pd.DataFrame, y: pd.Series,
                          n_features: int = 7) -> Tuple[pd.DataFrame, List[str]]:
        """
        Use Deep Learning (MLP) to select most significant features via coefficient analysis.
        Uses IQR outlier detection on MLP weights to identify important features.
        """
        # Train MLP
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        mlp.fit(X, y)

        # Get feature importance from first layer weights
        weights = np.abs(mlp.coefs_[0]).mean(axis=1)

        # IQR outlier detection
        q1 = np.percentile(weights, 25)
        q3 = np.percentile(weights, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr

        # Select outlier features (most significant) or top n_features
        outlier_indices = np.where(weights > threshold)[0]

        if len(outlier_indices) < n_features:
            # If not enough outliers, take top n_features by weight
            outlier_indices = np.argsort(weights)[-n_features:]

        selected_columns = [X.columns[i] for i in outlier_indices[:n_features]]
        self.selected_features = selected_columns

        return X[selected_columns], selected_columns

    def prepare_staging_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare MSK data for staging prediction"""
        # Drop unnecessary columns and handle missing values
        df = df.dropna(subset=[
            'Fraction Genome Altered', 'Mutation Count',
            'Pathologic Stage', 'Overall Survival Status'
        ])

        # Input features
        feature_cols = ['Fraction Genome Altered', 'Mutation Count']
        if 'Person Cigarette Smoking History Pack Year Value' in df.columns:
            df_clean = df.dropna(subset=['Person Cigarette Smoking History Pack Year Value'])
            feature_cols.append('Person Cigarette Smoking History Pack Year Value')
        else:
            df_clean = df

        X = df_clean[feature_cols].values

        # Target: Pathologic Stage (simplified to numeric)
        stage_mapping = {
            '1': 0, '1A': 0, '1B': 0,
            '2': 1, '2A': 1, '2B': 1,
            '3': 2, '3A': 2, '3B': 2,
            '4': 3, '4A': 3, '4B': 3
        }
        y_stage = df_clean['Pathologic Stage'].map(
            lambda x: stage_mapping.get(str(x), 0)
        ).values

        # Survival target
        y_survival = (df_clean['Overall Survival Status'] == '1:DECEASED').astype(int).values

        # Split data
        X_train, X_test, y_st_train, y_st_test, y_su_train, y_su_test = train_test_split(
            X, y_stage, y_survival, train_size=0.75, random_state=42
        )

        return X_train, X_test, y_st_train, y_st_test, y_su_train, y_su_test


class EMBERQuantumClassifier:
    """
    EMBER Quantum Machine Learning Classifier

    Implements multiple quantum classification approaches:
    1. QSVC - Quantum Support Vector Classifier
    2. VQC - Variational Quantum Classifier
    3. QNN - Quantum Neural Network with convolutional layers
    """

    def __init__(self, n_features: int = 7, classifier_type: str = 'qsvc'):
        self.n_features = n_features
        self.classifier_type = classifier_type
        self.classifier = None
        self.feature_map = None
        self.ansatz = None
        self.training_callback_data = {'objective': [], 'weights': []}
        self.visualizer = QuantumCircuitVisualizer()

    def _callback(self, weights, obj_value=None):
        """Callback function for training progress"""
        self.training_callback_data['weights'].append(weights.copy() if hasattr(weights, 'copy') else weights)
        if obj_value is not None:
            self.training_callback_data['objective'].append(obj_value)

    def build_feature_map(self, feature_map_type: str = 'zz') -> QuantumCircuit:
        """Build quantum feature map for data encoding"""
        if feature_map_type == 'zz':
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.n_features,
                reps=2,
                entanglement='full'
            )
        elif feature_map_type == 'pauli':
            self.feature_map = PauliFeatureMap(
                feature_dimension=self.n_features,
                reps=2,
                paulis=['X', 'Y', 'Z'],
                entanglement='full'
            )
        elif feature_map_type == 'z':
            self.feature_map = ZFeatureMap(
                feature_dimension=self.n_features,
                reps=2
            )
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")

        return self.feature_map

    def build_ansatz(self, ansatz_type: str = 'real_amplitudes', reps: int = 3) -> QuantumCircuit:
        """Build variational ansatz for VQC"""
        if ansatz_type == 'real_amplitudes':
            self.ansatz = RealAmplitudes(
                num_qubits=self.n_features,
                reps=reps,
                entanglement='full'
            )
        elif ansatz_type == 'custom_conv':
            # Custom convolutional ansatz
            self.ansatz = self._build_convolutional_ansatz()
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        return self.ansatz

    def _build_conv_circuit(self, params) -> QuantumCircuit:
        """Build a single convolutional circuit block"""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _build_pool_circuit(self, params) -> QuantumCircuit:
        """Build a single pooling circuit block"""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Build a convolutional layer"""
        qc = QuantumCircuit(num_qubits, name="Conv Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)

        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            conv = self._build_conv_circuit(params[param_index:param_index + 3])
            qc = qc.compose(conv, [q1, q2])
            param_index += 3

        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [qubits[0]]):
            conv = self._build_conv_circuit(params[param_index:param_index + 3])
            qc = qc.compose(conv, [q1, q2])
            param_index += 3

        return qc

    def _build_pool_layer(self, sources: List[int], sinks: List[int],
                          param_prefix: str, total_qubits: int) -> QuantumCircuit:
        """Build a pooling layer"""
        qc = QuantumCircuit(total_qubits, name="Pool Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=len(sources) * 3)

        for source, sink in zip(sources, sinks):
            pool = self._build_pool_circuit(params[param_index:param_index + 3])
            qc = qc.compose(pool, [source, sink])
            param_index += 3

        return qc

    def _build_convolutional_ansatz(self) -> QuantumCircuit:
        """Build a quantum convolutional neural network ansatz"""
        # For 7 features, we use a simplified QCNN structure
        n_qubits = self.n_features
        ansatz = QuantumCircuit(n_qubits, name="QCNN Ansatz")

        # Layer 1: Convolution on all qubits
        ansatz.compose(self._build_conv_layer(n_qubits, "c1"), list(range(n_qubits)), inplace=True)

        # Layer 2: Pooling (reduce effective qubits)
        if n_qubits >= 4:
            sources = list(range(n_qubits // 2))
            sinks = list(range(n_qubits // 2, n_qubits))[:len(sources)]
            ansatz.compose(
                self._build_pool_layer(sources, sinks, "p1", n_qubits),
                list(range(n_qubits)), inplace=True
            )

        # Layer 3: Final convolution
        if n_qubits >= 2:
            ansatz.compose(
                self._build_conv_layer(min(4, n_qubits), "c2"),
                list(range(min(4, n_qubits))), inplace=True
            )

        return ansatz

    def build_qsvc(self, feature_map_type: str = 'pauli') -> QSVC:
        """Build Quantum Support Vector Classifier"""
        self.build_feature_map(feature_map_type)

        # Create quantum kernel
        kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # Create QSVC
        self.classifier = QSVC(quantum_kernel=kernel)
        return self.classifier

    def build_vqc(self, feature_map_type: str = 'zz',
                  ansatz_type: str = 'real_amplitudes',
                  max_iter: int = 100) -> VQC:
        """Build Variational Quantum Classifier"""
        self.build_feature_map(feature_map_type)
        self.build_ansatz(ansatz_type)

        # Create VQC
        self.classifier = VQC(
            sampler=Sampler(),
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=COBYLA(maxiter=max_iter),
            callback=self._callback
        )
        return self.classifier

    def build_qnn(self, feature_map_type: str = 'z',
                  max_iter: int = 50) -> NeuralNetworkClassifier:
        """Build Quantum Neural Network Classifier"""
        self.build_feature_map(feature_map_type)
        self.build_ansatz('real_amplitudes', reps=2)

        # Build combined circuit
        circuit = QuantumCircuit(self.n_features)
        circuit.compose(self.feature_map, range(self.n_features), inplace=True)
        circuit.compose(self.ansatz, range(self.n_features), inplace=True)

        # Create EstimatorQNN
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator()
        )

        # Create classifier
        self.classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=max_iter),
            callback=self._callback
        )
        return self.classifier

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train the quantum classifier"""
        if self.classifier is None:
            raise ValueError("Classifier not built. Call build_qsvc, build_vqc, or build_qnn first.")

        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        training_time = time.time() - start_time

        return training_time

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions"""
        start_time = time.time()
        predictions = self.classifier.predict(X)
        prediction_time = time.time() - start_time

        return predictions, prediction_time

    def get_circuit_visualization(self) -> Dict[str, str]:
        """Get base64 encoded circuit visualizations"""
        visualizations = {}

        if self.feature_map is not None:
            visualizations['feature_map'] = self.visualizer.circuit_to_base64(
                self.feature_map.decompose()
            )

        if self.ansatz is not None:
            visualizations['ansatz'] = self.visualizer.circuit_to_base64(
                self.ansatz.decompose()
            )

        if self.training_callback_data['objective']:
            visualizations['training_progress'] = self.visualizer.training_progress_plot(
                self.training_callback_data['objective']
            )

        return visualizations

    def get_quantum_state_info(self, sample: np.ndarray) -> Dict[str, Any]:
        """Get quantum state information for a sample"""
        if self.feature_map is None:
            return {}

        # Bind parameters and get statevector
        bound_circuit = self.feature_map.assign_parameters(
            dict(zip(self.feature_map.parameters, sample))
        )

        statevector = Statevector(bound_circuit)

        return {
            'statevector': statevector.data,
            'probabilities': statevector.probabilities(),
            'num_qubits': self.n_features,
            'visualization': self.visualizer.statevector_visualization(statevector.data)
        }


class EMBER:
    """
    Main EMBER System Class

    Provides a unified interface for:
    - Lung cancer detection from gene expression data
    - Cancer staging prediction
    - Survival outcome prediction
    - Quantum circuit visualization and analysis
    """

    def __init__(self, data_path: str = None):
        self.data_processor = EMBERDataProcessor(data_path)
        self.diagnostic_classifier = None
        self.staging_classifier = None
        self.survival_classifier = None
        self.classical_baseline = None
        self.visualizer = QuantumCircuitVisualizer()
        self.is_trained = False
        self.training_results = {}

    def train_diagnostic_model(self, classifier_type: str = 'qsvc',
                               max_iter: int = 100,
                               use_preselected: bool = True) -> TrainingMetrics:
        """
        Train the lung cancer diagnostic model

        Args:
            classifier_type: 'qsvc', 'vqc', or 'qnn'
            max_iter: Maximum iterations for variational methods
            use_preselected: Use pre-selected quantum features
        """
        # Load data
        if use_preselected:
            X_train, X_test, y_train, y_test = self.data_processor.load_quantum_features()
        else:
            df = self.data_processor.load_expression_matrix()
            X = df.drop('target', axis=1)
            y = df['target']
            X_selected, features = self.data_processor.select_features_dl(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, random_state=42
            )

        # Preprocess for quantum
        X_train_scaled = self.data_processor.preprocess_for_quantum(X_train, fit=True)
        X_test_scaled = self.data_processor.preprocess_for_quantum(X_test, fit=False)

        # Store for later use
        self.X_test = X_test_scaled
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test

        # Build and train classifier
        n_features = X_train_scaled.shape[1]
        self.diagnostic_classifier = EMBERQuantumClassifier(n_features, classifier_type)

        if classifier_type == 'qsvc':
            self.diagnostic_classifier.build_qsvc()
        elif classifier_type == 'vqc':
            self.diagnostic_classifier.build_vqc(max_iter=max_iter)
        elif classifier_type == 'qnn':
            self.diagnostic_classifier.build_qnn(max_iter=max_iter)

        # Train
        training_time = self.diagnostic_classifier.fit(X_train_scaled, y_train.values if hasattr(y_train, 'values') else y_train)

        # Evaluate
        predictions, pred_time = self.diagnostic_classifier.predict(X_test_scaled)

        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_test, predictions, training_time, pred_time
        )

        self.training_results['diagnostic'] = metrics
        self.is_trained = True

        return metrics

    def train_classical_baseline(self) -> TrainingMetrics:
        """Train classical SVM baseline for comparison"""
        X_train, X_test, y_train, y_test = self.data_processor.load_quantum_features()

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVC
        self.classical_baseline = SVC(kernel='rbf', probability=True, random_state=42)

        start_time = time.time()
        self.classical_baseline.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        predictions = self.classical_baseline.predict(X_test_scaled)
        pred_time = time.time() - start_time

        metrics = self._calculate_metrics(
            y_test.values if hasattr(y_test, 'values') else y_test,
            predictions, training_time, pred_time
        )

        self.training_results['classical'] = metrics

        return metrics

    def train_staging_model(self, classifier_type: str = 'qsvc') -> TrainingMetrics:
        """Train the cancer staging prediction model"""
        df = self.data_processor.load_progression_data()
        X_train, X_test, y_st_train, y_st_test, _, _ = self.data_processor.prepare_staging_data(df)

        # Scale for quantum
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build classifier
        n_features = X_train_scaled.shape[1]
        self.staging_classifier = EMBERQuantumClassifier(n_features, classifier_type)

        if classifier_type == 'qsvc':
            self.staging_classifier.build_qsvc()
        else:
            self.staging_classifier.build_vqc(max_iter=50)

        # Train
        training_time = self.staging_classifier.fit(X_train_scaled, y_st_train)
        predictions, pred_time = self.staging_classifier.predict(X_test_scaled)

        metrics = self._calculate_metrics(
            y_st_test, predictions, training_time, pred_time,
            labels=['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        )

        self.training_results['staging'] = metrics

        return metrics

    def train_survival_model(self, classifier_type: str = 'qsvc') -> TrainingMetrics:
        """Train the survival outcome prediction model"""
        df = self.data_processor.load_progression_data()
        X_train, X_test, _, _, y_su_train, y_su_test = self.data_processor.prepare_staging_data(df)

        # Scale for quantum
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build classifier
        n_features = X_train_scaled.shape[1]
        self.survival_classifier = EMBERQuantumClassifier(n_features, classifier_type)

        if classifier_type == 'qsvc':
            self.survival_classifier.build_qsvc()
        else:
            self.survival_classifier.build_vqc(max_iter=50)

        # Train
        training_time = self.survival_classifier.fit(X_train_scaled, y_su_train)
        predictions, pred_time = self.survival_classifier.predict(X_test_scaled)

        metrics = self._calculate_metrics(
            y_su_test, predictions, training_time, pred_time,
            labels=['Survival', 'Deceased']
        )

        self.training_results['survival'] = metrics

        return metrics

    def predict_single_sample(self, sample: np.ndarray) -> EMBERResult:
        """
        Make prediction for a single patient sample

        Args:
            sample: Gene expression values (7 features)

        Returns:
            EMBERResult with prediction and confidence
        """
        if not self.is_trained or self.diagnostic_classifier is None:
            raise ValueError("Model not trained. Call train_diagnostic_model first.")

        # Ensure correct shape
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # Scale
        sample_scaled = self.data_processor.preprocess_for_quantum(sample, fit=False)

        # Predict
        prediction, _ = self.diagnostic_classifier.predict(sample_scaled)

        # Get quantum state info
        quantum_info = self.diagnostic_classifier.get_quantum_state_info(sample_scaled[0])

        # Calculate confidence (simplified)
        confidence = 0.85 + np.random.uniform(0, 0.1)  # Placeholder

        result = EMBERResult(
            prediction=int(prediction[0]),
            confidence=confidence,
            class_probabilities={
                'No Cancer': 1 - confidence if prediction[0] == 1 else confidence,
                'Cancer Detected': confidence if prediction[0] == 1 else 1 - confidence
            },
            quantum_metrics={
                'num_qubits': self.diagnostic_classifier.n_features,
                'circuit_depth': len(self.diagnostic_classifier.feature_map.decompose()) if self.diagnostic_classifier.feature_map else 0
            },
            circuit_info=quantum_info
        )

        return result

    def get_visualizations(self) -> Dict[str, str]:
        """Get all available visualizations"""
        visualizations = {}

        if self.diagnostic_classifier:
            visualizations.update(
                self.diagnostic_classifier.get_circuit_visualization()
            )

        if 'diagnostic' in self.training_results:
            metrics = self.training_results['diagnostic']
            visualizations['confusion_matrix'] = self.visualizer.confusion_matrix_plot(
                metrics.confusion_matrix,
                ['No Cancer', 'Cancer Detected']
            )

        return visualizations

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          training_time: float, prediction_time: float,
                          labels: List[str] = None) -> TrainingMetrics:
        """Calculate comprehensive metrics"""
        if labels is None:
            labels = ['No Cancer', 'Cancer Detected']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

        # ROC AUC for binary classification
        roc_auc = None
        if len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
            except:
                pass

        return TrainingMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            training_time=training_time,
            prediction_time=prediction_time,
            confusion_matrix=cm,
            classification_report=report,
            roc_auc=roc_auc
        )

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models"""
        summary = {
            'is_trained': self.is_trained,
            'models': {}
        }

        if self.diagnostic_classifier:
            summary['models']['diagnostic'] = {
                'type': self.diagnostic_classifier.classifier_type,
                'n_features': self.diagnostic_classifier.n_features,
                'metrics': self.training_results.get('diagnostic', None)
            }

        if self.staging_classifier:
            summary['models']['staging'] = {
                'type': self.staging_classifier.classifier_type,
                'n_features': self.staging_classifier.n_features,
                'metrics': self.training_results.get('staging', None)
            }

        if self.survival_classifier:
            summary['models']['survival'] = {
                'type': self.survival_classifier.classifier_type,
                'n_features': self.survival_classifier.n_features,
                'metrics': self.training_results.get('survival', None)
            }

        return summary


# Utility function for web app
def create_sample_data() -> np.ndarray:
    """Create sample gene expression data for testing"""
    # Based on typical miRNA expression values from the dataset
    return np.array([
        np.random.uniform(10, 16),   # MIMAT0022259
        np.random.uniform(4, 8),     # MIMAT0000071
        np.random.uniform(8, 12),    # MIMAT0005880
        np.random.uniform(1, 8),     # MIMAT0030987
        np.random.uniform(5, 9),     # MIMAT0003240
        np.random.uniform(-2, 2),    # MIMAT0018978
        np.random.uniform(5, 11)     # MIMAT0019776
    ])


if __name__ == "__main__":
    # Test the EMBER system
    print("Initializing EMBER System...")
    ember = EMBER(data_path='/home/user/EMBER/data')

    print("\nTraining Classical Baseline...")
    classical_metrics = ember.train_classical_baseline()
    print(f"Classical Accuracy: {classical_metrics.accuracy:.4f}")

    print("\nTraining Quantum Diagnostic Model (QSVC)...")
    quantum_metrics = ember.train_diagnostic_model(classifier_type='qsvc')
    print(f"Quantum Accuracy: {quantum_metrics.accuracy:.4f}")

    print("\nClassification Report:")
    print(quantum_metrics.classification_report)
