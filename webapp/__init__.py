"""
EMBER Web Application Package

Early Malignancy Biomarker Evaluation and Recognition
Quantum-Enhanced Lung Cancer Detection System
"""

from .ember_core import (
    EMBER,
    EMBERResult,
    TrainingMetrics,
    EMBERDataProcessor,
    EMBERQuantumClassifier,
    QuantumCircuitVisualizer,
    create_sample_data
)

__version__ = '1.0.0'
__author__ = 'EMBER Project'

__all__ = [
    'EMBER',
    'EMBERResult',
    'TrainingMetrics',
    'EMBERDataProcessor',
    'EMBERQuantumClassifier',
    'QuantumCircuitVisualizer',
    'create_sample_data'
]
