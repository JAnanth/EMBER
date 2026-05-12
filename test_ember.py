#!/usr/bin/env python3
"""
EMBER System Test Script

This script tests the core functionality of the EMBER system.
Run this to verify the installation and basic functionality.
"""

import sys
import os
import numpy as np

# Add webapp to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp'))


def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")

    try:
        import pandas as pd
        print("  pandas: OK")
    except ImportError as e:
        print(f"  pandas: FAILED - {e}")
        return False

    try:
        import numpy as np
        print("  numpy: OK")
    except ImportError as e:
        print(f"  numpy: FAILED - {e}")
        return False

    try:
        from sklearn.svm import SVC
        print("  sklearn: OK")
    except ImportError as e:
        print(f"  sklearn: FAILED - {e}")
        return False

    try:
        from qiskit import QuantumCircuit
        print("  qiskit: OK")
    except ImportError as e:
        print(f"  qiskit: FAILED - {e}")
        return False

    try:
        from qiskit_machine_learning.algorithms.classifiers import QSVC
        print("  qiskit_machine_learning: OK")
    except ImportError as e:
        print(f"  qiskit_machine_learning: FAILED - {e}")
        return False

    try:
        import streamlit
        print("  streamlit: OK")
    except ImportError as e:
        print(f"  streamlit: FAILED - {e}")
        return False

    try:
        import plotly
        print("  plotly: OK")
    except ImportError as e:
        print(f"  plotly: FAILED - {e}")
        return False

    print("All imports successful!")
    return True


def test_data_loading():
    """Test that data files can be loaded"""
    print("\nTesting data loading...")

    from ember_core import EMBERDataProcessor

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    processor = EMBERDataProcessor(data_path)

    try:
        X_train, X_test, y_train, y_test = processor.load_quantum_features()
        print(f"  Quantum features: OK (Train: {len(X_train)}, Test: {len(X_test)})")
    except Exception as e:
        print(f"  Quantum features: FAILED - {e}")
        return False

    try:
        df = processor.load_expression_matrix()
        print(f"  Expression matrix: OK ({len(df)} samples)")
    except Exception as e:
        print(f"  Expression matrix: FAILED - {e}")
        return False

    try:
        df = processor.load_progression_data()
        print(f"  Progression data: OK ({len(df)} samples)")
    except Exception as e:
        print(f"  Progression data: FAILED - {e}")
        return False

    print("Data loading successful!")
    return True


def test_quantum_classifier():
    """Test quantum classifier initialization"""
    print("\nTesting quantum classifier...")

    from ember_core import EMBERQuantumClassifier

    try:
        # Test QSVC
        classifier = EMBERQuantumClassifier(n_features=7, classifier_type='qsvc')
        classifier.build_qsvc()
        print("  QSVC initialization: OK")
    except Exception as e:
        print(f"  QSVC initialization: FAILED - {e}")
        return False

    try:
        # Test VQC
        classifier = EMBERQuantumClassifier(n_features=7, classifier_type='vqc')
        classifier.build_vqc(max_iter=5)
        print("  VQC initialization: OK")
    except Exception as e:
        print(f"  VQC initialization: FAILED - {e}")
        return False

    try:
        # Test feature map generation
        classifier = EMBERQuantumClassifier(n_features=4, classifier_type='qsvc')
        fm = classifier.build_feature_map('pauli')
        print(f"  Feature map (Pauli): OK ({fm.num_qubits} qubits)")
    except Exception as e:
        print(f"  Feature map: FAILED - {e}")
        return False

    print("Quantum classifier tests successful!")
    return True


def test_ember_system():
    """Test the main EMBER system"""
    print("\nTesting EMBER system...")

    from ember_core import EMBER, create_sample_data

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    try:
        ember = EMBER(data_path=data_path)
        print("  EMBER initialization: OK")
    except Exception as e:
        print(f"  EMBER initialization: FAILED - {e}")
        return False

    try:
        sample = create_sample_data()
        print(f"  Sample data generation: OK (shape: {sample.shape})")
    except Exception as e:
        print(f"  Sample data generation: FAILED - {e}")
        return False

    print("EMBER system tests successful!")
    return True


def test_visualization():
    """Test visualization utilities"""
    print("\nTesting visualization...")

    from ember_core import QuantumCircuitVisualizer
    from qiskit import QuantumCircuit

    try:
        visualizer = QuantumCircuitVisualizer()

        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Generate visualization
        img_base64 = visualizer.circuit_to_base64(qc)

        if len(img_base64) > 100:
            print(f"  Circuit visualization: OK (image size: {len(img_base64)} bytes)")
        else:
            print("  Circuit visualization: FAILED - Image too small")
            return False

    except Exception as e:
        print(f"  Circuit visualization: FAILED - {e}")
        return False

    print("Visualization tests successful!")
    return True


def run_full_training_test(quick=True):
    """Run a full training test (optional, takes longer)"""
    print("\nRunning full training test...")

    from ember_core import EMBER

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    ember = EMBER(data_path=data_path)

    try:
        # Train classical baseline first (faster)
        print("  Training classical baseline...")
        classical_metrics = ember.train_classical_baseline()
        print(f"  Classical accuracy: {classical_metrics.accuracy:.2%}")

        if not quick:
            # Train quantum model (slower)
            print("  Training quantum QSVC (this may take a few minutes)...")
            quantum_metrics = ember.train_diagnostic_model(classifier_type='qsvc')
            print(f"  Quantum accuracy: {quantum_metrics.accuracy:.2%}")

            # Make a prediction
            sample = np.array([15.0, 6.5, 10.8, 7.3, 6.8, 0.7, 10.0])
            result = ember.predict_single_sample(sample)
            print(f"  Sample prediction: {'Cancer' if result.prediction else 'No Cancer'} "
                  f"(confidence: {result.confidence:.1%})")

        print("Full training test successful!")
        return True

    except Exception as e:
        print(f"  Full training test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("EMBER System Test Suite")
    print("=" * 60)

    all_passed = True

    # Run basic tests
    all_passed &= test_imports()
    all_passed &= test_data_loading()
    all_passed &= test_quantum_classifier()
    all_passed &= test_ember_system()
    all_passed &= test_visualization()

    # Optional: run full training test
    if '--full' in sys.argv:
        all_passed &= run_full_training_test(quick=False)
    elif '--quick-train' in sys.argv:
        all_passed &= run_full_training_test(quick=True)

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
        print("EMBER system is ready to use.")
        print("\nTo start the web application:")
        print("  python run_app.py")
    else:
        print("Some tests FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
