# EMBER - Early Malignancy Biomarker Evaluation and Recognition

A quantum computing-based system for early lung cancer detection using gene expression data from blood tests.

## Overview

EMBER leverages quantum machine learning to analyze miRNA (microRNA) expression patterns and predict:
- **Lung Cancer Presence**: Binary classification detecting cancer vs. non-cancer
- **Cancer Staging**: Predicting pathological stage (I-IV)
- **Survival Outcomes**: Estimating patient prognosis

### Key Features

- **Quantum Advantage**: Uses quantum superposition and entanglement for enhanced pattern recognition
- **93% Accuracy**: Outperforms classical deep learning baseline (85%)
- **Accessible Testing**: Works with simple, inexpensive blood tests
- **Interactive Web Interface**: Real-time quantum simulation visualization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/EMBER.git
cd EMBER

# Install dependencies
pip install -r requirements.txt

# Run the web application
python run_app.py
```

### Access the Application

Open your browser and navigate to: `http://localhost:8501`

## Project Structure

```
EMBER/
├── webapp/                    # Web application
│   ├── app.py                # Streamlit application
│   └── ember_core.py         # Core quantum ML module
├── code/
│   ├── final/                # Production code
│   │   ├── quantum_implementation.py
│   │   ├── preprocessing.py
│   │   ├── DL_FeatureSelection.py
│   │   └── staging_survival.py
│   └── testing_versions/     # Development code
├── data/
│   ├── DirectPull/           # Raw data files
│   ├── Manipulated/          # Processed datasets
│   └── cBIO_Progression/     # Clinical data
├── requirements.txt
├── run_app.py               # Application launcher
└── README.md
```

## Technology Stack

### Quantum Computing
- **Framework**: IBM Qiskit
- **Classifiers**: QSVC, VQC, QNN
- **Feature Maps**: Pauli, ZZ, Z Feature Maps
- **Simulator**: Qiskit Aer

### Machine Learning
- **Classical Baseline**: SVM with RBF kernel
- **Feature Selection**: MLP-based coefficient analysis
- **Preprocessing**: Min-Max scaling to [0, π]

### Web Application
- **Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Real-time**: Interactive quantum circuit visualization

## Quantum Classifiers

### 1. QSVC (Quantum Support Vector Classifier)
- Uses quantum kernel estimation
- Pauli Feature Map (X, Y, Z gates)
- Full entanglement connectivity
- Best for smaller datasets

### 2. VQC (Variational Quantum Classifier)
- Parameterized quantum circuit
- ZZ Feature Map with RealAmplitudes ansatz
- COBYLA optimizer
- Good balance of speed and accuracy

### 3. QNN (Quantum Neural Network)
- Quantum convolutional layers
- Pooling layers for dimensionality reduction
- Most powerful but computationally intensive

## Data

### GSE137140 Dataset
- **Source**: Gene Expression Omnibus (GEO)
- **Samples**: 3,924 patients
- **Features**: 2,550 miRNA biomarkers → 7 selected features
- **Classes**: Cancer (1,746) vs. Control (1,774)

### MSK Clinical Data
- **Source**: Memorial Sloan Kettering cBioPortal
- **Samples**: 604 patients
- **Features**: Mutation count, genome alteration, smoking history
- **Targets**: Pathological stage, survival status

## Usage Examples

### Python API

```python
from webapp.ember_core import EMBER

# Initialize EMBER
ember = EMBER(data_path='./data')

# Train quantum classifier
metrics = ember.train_diagnostic_model(
    classifier_type='qsvc',  # or 'vqc', 'qnn'
    max_iter=100
)

print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"F1 Score: {metrics.f1:.2%}")

# Make prediction
import numpy as np
sample = np.array([15.0, 6.5, 10.8, 7.3, 6.8, 0.7, 10.0])
result = ember.predict_single_sample(sample)

print(f"Prediction: {'Cancer' if result.prediction else 'No Cancer'}")
print(f"Confidence: {result.confidence:.1%}")
```

### Web Application

1. Navigate to **Train Model** page
2. Select classifier type (QSVC recommended)
3. Click **Train Quantum Model**
4. Go to **Make Prediction** page
5. Enter gene expression values or use sample data
6. View results and quantum analysis

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Classical SVM | 85% | 84% | 86% | 85% |
| **Quantum QSVC** | **93%** | **92%** | **94%** | **93%** |
| Quantum VQC | 91% | 90% | 92% | 91% |
| Quantum QNN | 90% | 89% | 91% | 90% |

### Key Findings

1. **Quantum Advantage**: 8% improvement over classical baseline
2. **Feature Efficiency**: Only 7 miRNA biomarkers needed
3. **Computational Cost**: QSVC fastest, QNN most accurate (with more iterations)

## How It Works

### 1. Data Encoding
Gene expression values are encoded into quantum states using parameterized rotation gates:
```
|ψ⟩ = U(x)|0⟩^⊗n
```

### 2. Quantum Entanglement
CNOT gates create entanglement between qubits, capturing feature correlations:
```
|ψ⟩ = Σ αᵢ|basis states⟩
```

### 3. Measurement
Quantum states are measured to produce classification probabilities:
```
P(class) = |⟨class|ψ⟩|²
```

## Future Work

- [ ] Integration with IBM Quantum hardware
- [ ] Additional cancer type support
- [ ] Mobile application
- [ ] Clinical validation studies

## References

1. GEO Dataset: GSM 4067570 (GSE137140)
2. cBioPortal: MSK Lung Adenocarcinoma Study
3. Qiskit Machine Learning Documentation
4. OWASP Secure Coding Guidelines

## License

This project is for educational and research purposes.

## Contact

For questions about EMBER or quantum computing in healthcare, please open an issue or contact the development team.

---

*EMBER: Enabling early detection through quantum innovation*
