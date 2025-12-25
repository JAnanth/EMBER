"""
EMBER Web Application - Early Malignancy Biomarker Evaluation and Recognition

A Streamlit-based web application for quantum-enhanced lung cancer detection
using gene expression data.

Author: EMBER Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ember_core import EMBER, EMBERResult, create_sample_data, QuantumCircuitVisualizer

# Page configuration
st.set_page_config(
    page_title="EMBER - Quantum Lung Cancer Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .quantum-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .result-positive {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .result-negative {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'ember' not in st.session_state:
        st.session_state.ember = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = None
    if 'classical_metrics' not in st.session_state:
        st.session_state.classical_metrics = None
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'quantum_visualizations' not in st.session_state:
        st.session_state.quantum_visualizations = {}


def create_header():
    """Create application header"""
    st.markdown('<h1 class="main-header">EMBER</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Early Malignancy Biomarker Evaluation and Recognition<br>'
        'Quantum-Enhanced Lung Cancer Detection System</p>',
        unsafe_allow_html=True
    )


def create_sidebar():
    """Create sidebar with navigation and options"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=EMBER", width=150)
        st.markdown("---")

        page = st.selectbox(
            "Navigation",
            ["Home", "Train Model", "Make Prediction", "Quantum Analysis", "About"]
        )

        st.markdown("---")

        st.markdown("### Model Configuration")
        classifier_type = st.selectbox(
            "Quantum Classifier",
            ["QSVC", "VQC", "QNN"],
            help="Select the quantum classification algorithm"
        )

        if classifier_type in ["VQC", "QNN"]:
            max_iter = st.slider(
                "Max Iterations",
                min_value=10,
                max_value=200,
                value=50,
                help="Maximum optimization iterations"
            )
        else:
            max_iter = 100

        st.markdown("---")

        st.markdown("### System Status")
        if st.session_state.is_trained:
            st.success("Model Trained")
        else:
            st.warning("Model Not Trained")

        return page, classifier_type.lower(), max_iter


def page_home():
    """Home page content"""
    st.markdown("## Welcome to EMBER")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Problem
        Lung cancer is the most deadly form of cancer, responsible for over
        **2 million deaths annually**. Early detection is crucial, with survival
        rates dropping **57%** when detected late.
        """)

    with col2:
        st.markdown("""
        ### Solution
        EMBER leverages **quantum computing** to analyze gene expression data
        from simple blood tests, achieving **93% accuracy** in lung cancer
        detection - outperforming classical methods.
        """)

    with col3:
        st.markdown("""
        ### Impact
        Enable **rapid, inexpensive, and accessible** lung cancer diagnostics
        for patients of all backgrounds, demographics, and socioeconomic statuses.
        """)

    st.markdown("---")

    # Feature highlights
    st.markdown("## Key Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        #### Quantum Advantage
        Utilizes quantum superposition and entanglement for enhanced pattern recognition
        """)

    with col2:
        st.markdown("""
        #### Gene Expression
        Analyzes miRNA biomarkers from accessible blood tests
        """)

    with col3:
        st.markdown("""
        #### Cancer Staging
        Predicts cancer progression and staging
        """)

    with col4:
        st.markdown("""
        #### Survival Prediction
        Estimates patient outcomes for treatment planning
        """)

    st.markdown("---")

    # Quick start
    st.markdown("## Quick Start")
    st.markdown("""
    1. **Train Model**: Navigate to 'Train Model' to train the quantum classifier
    2. **Make Prediction**: Upload gene expression data or use sample data
    3. **View Analysis**: Explore quantum circuit visualizations and metrics
    """)

    if st.button("Start Training Model", type="primary"):
        st.session_state.page = "Train Model"
        st.rerun()


def page_train_model(classifier_type: str, max_iter: int):
    """Training page content"""
    st.markdown("## Model Training")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Training Configuration
        The EMBER system will train on the GSE137140 dataset containing
        **3,924 patient samples** with **7 selected miRNA biomarkers**.
        """)

        st.markdown(f"""
        **Selected Configuration:**
        - Classifier: `{classifier_type.upper()}`
        - Max Iterations: `{max_iter}`
        - Feature Map: `Pauli (X, Y, Z)` for QSVC, `ZZ` for VQC/QNN
        - Qubits: `7` (one per feature)
        """)

    with col2:
        st.markdown("### Dataset Info")
        st.metric("Total Samples", "3,924")
        st.metric("Selected Features", "7 miRNAs")
        st.metric("Train/Test Split", "70/30")

    st.markdown("---")

    # Training controls
    col1, col2, col3 = st.columns(3)

    with col1:
        train_quantum = st.button("Train Quantum Model", type="primary")
    with col2:
        train_classical = st.button("Train Classical Baseline")
    with col3:
        train_all = st.button("Train Both Models")

    if train_quantum or train_all:
        with st.spinner("Initializing EMBER System..."):
            st.session_state.ember = EMBER(data_path='/home/user/EMBER/data')

        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Loading and preprocessing data...")
        progress_bar.progress(20)
        time.sleep(0.5)

        status_text.text(f"Building {classifier_type.upper()} quantum classifier...")
        progress_bar.progress(40)

        status_text.text("Training quantum model (this may take a few minutes)...")
        progress_bar.progress(60)

        try:
            metrics = st.session_state.ember.train_diagnostic_model(
                classifier_type=classifier_type,
                max_iter=max_iter
            )
            st.session_state.training_metrics = metrics
            st.session_state.is_trained = True

            progress_bar.progress(100)
            status_text.text("Training complete!")

            # Display results
            st.success("Quantum model training completed successfully!")
            display_training_results(metrics, "Quantum Classifier")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    if train_classical or train_all:
        if st.session_state.ember is None:
            st.session_state.ember = EMBER(data_path='/home/user/EMBER/data')

        with st.spinner("Training classical SVM baseline..."):
            try:
                classical_metrics = st.session_state.ember.train_classical_baseline()
                st.session_state.classical_metrics = classical_metrics
                st.success("Classical baseline training completed!")
                display_training_results(classical_metrics, "Classical SVM")
            except Exception as e:
                st.error(f"Classical training failed: {str(e)}")

    # Show comparison if both trained
    if st.session_state.training_metrics and st.session_state.classical_metrics:
        st.markdown("---")
        st.markdown("## Model Comparison")
        display_model_comparison()


def display_training_results(metrics, model_name: str):
    """Display training results"""
    st.markdown(f"### {model_name} Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics.accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{metrics.precision:.2%}")
    with col3:
        st.metric("Recall", f"{metrics.recall:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics.f1:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Time", f"{metrics.training_time:.2f}s")
    with col2:
        st.metric("Prediction Time", f"{metrics.prediction_time:.4f}s")

    # Classification report
    with st.expander("View Classification Report"):
        st.text(metrics.classification_report)

    # Confusion matrix visualization
    fig = go.Figure(data=go.Heatmap(
        z=metrics.confusion_matrix,
        x=['Predicted: No Cancer', 'Predicted: Cancer'],
        y=['Actual: No Cancer', 'Actual: Cancer'],
        colorscale='Blues',
        text=metrics.confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def display_model_comparison():
    """Display comparison between quantum and classical models"""
    q_metrics = st.session_state.training_metrics
    c_metrics = st.session_state.classical_metrics

    # Comparison chart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    quantum_values = [q_metrics.accuracy, q_metrics.precision, q_metrics.recall, q_metrics.f1]
    classical_values = [c_metrics.accuracy, c_metrics.precision, c_metrics.recall, c_metrics.f1]

    fig = go.Figure(data=[
        go.Bar(name='Quantum Classifier', x=metrics_names, y=quantum_values,
               marker_color='#1E88E5'),
        go.Bar(name='Classical SVM', x=metrics_names, y=classical_values,
               marker_color='#FFA726')
    ])

    fig.update_layout(
        title="Quantum vs Classical Performance Comparison",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Improvement metrics
    improvement = (q_metrics.accuracy - c_metrics.accuracy) / c_metrics.accuracy * 100

    if improvement > 0:
        st.success(f"Quantum classifier shows **{improvement:.1f}%** improvement over classical baseline!")
    else:
        st.info(f"Classical baseline performs {-improvement:.1f}% better in this run.")


def page_make_prediction():
    """Prediction page content"""
    st.markdown("## Make Prediction")

    if not st.session_state.is_trained:
        st.warning("Please train the model first before making predictions.")
        if st.button("Go to Training"):
            st.rerun()
        return

    st.markdown("""
    ### Input Gene Expression Data
    Enter the expression values for the 7 selected miRNA biomarkers, or use sample data.
    """)

    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Upload CSV", "Use Sample Data"],
        horizontal=True
    )

    sample_data = None

    if input_method == "Manual Entry":
        st.markdown("#### Enter miRNA Expression Values")

        feature_names = [
            "MIMAT0022259", "MIMAT0000071", "MIMAT0005880", "MIMAT0030987",
            "MIMAT0003240", "MIMAT0018978", "MIMAT0019776"
        ]

        cols = st.columns(4)
        values = []

        for i, feature in enumerate(feature_names):
            with cols[i % 4]:
                val = st.number_input(
                    feature,
                    min_value=-10.0,
                    max_value=30.0,
                    value=10.0,
                    step=0.1,
                    key=f"feature_{i}"
                )
                values.append(val)

        sample_data = np.array(values)

    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file with gene expression data",
            type=['csv']
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            if len(df.columns) >= 7:
                sample_data = df.iloc[0, :7].values
            else:
                st.error("CSV must have at least 7 columns for the miRNA features")

    else:  # Use Sample Data
        st.markdown("#### Sample Data")
        if st.button("Generate Random Sample"):
            sample_data = create_sample_data()
            st.session_state.sample_data = sample_data

        if 'sample_data' in st.session_state:
            sample_data = st.session_state.sample_data

            feature_names = [
                "MIMAT0022259", "MIMAT0000071", "MIMAT0005880", "MIMAT0030987",
                "MIMAT0003240", "MIMAT0018978", "MIMAT0019776"
            ]

            df_display = pd.DataFrame([sample_data], columns=feature_names)
            st.dataframe(df_display)

    # Make prediction
    if sample_data is not None:
        st.markdown("---")

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running quantum prediction..."):
                try:
                    result = st.session_state.ember.predict_single_sample(sample_data)
                    st.session_state.current_prediction = result

                    # Display result
                    display_prediction_result(result, sample_data)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.exception(e)


def display_prediction_result(result: EMBERResult, sample_data: np.ndarray):
    """Display prediction result with visualizations"""
    st.markdown("---")
    st.markdown("## Prediction Results")

    # Main result
    if result.prediction == 1:
        st.markdown("""
        <div class="result-positive">
            <h3>Result: Lung Cancer Indicators Detected</h3>
            <p>The gene expression pattern suggests potential malignancy.
            Further clinical evaluation is recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-negative">
            <h3>Result: No Cancer Indicators Detected</h3>
            <p>The gene expression pattern appears normal.
            Regular screening is still recommended.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Confidence metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "Cancer" if result.prediction == 1 else "No Cancer")

    with col2:
        st.metric("Confidence", f"{result.confidence:.1%}")

    with col3:
        if result.quantum_metrics:
            st.metric("Qubits Used", result.quantum_metrics.get('num_qubits', 'N/A'))

    # Probability distribution
    st.markdown("### Class Probabilities")

    fig = go.Figure(data=[
        go.Bar(
            x=list(result.class_probabilities.keys()),
            y=list(result.class_probabilities.values()),
            marker_color=['#4CAF50', '#F44336']
        )
    ])
    fig.update_layout(
        yaxis_title="Probability",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance visualization
    st.markdown("### Input Feature Analysis")

    feature_names = [
        "MIMAT0022259", "MIMAT0000071", "MIMAT0005880", "MIMAT0030987",
        "MIMAT0003240", "MIMAT0018978", "MIMAT0019776"
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=sample_data,
            marker_color='#1E88E5'
        )
    ])
    fig.update_layout(
        title="Gene Expression Values",
        xaxis_title="miRNA Biomarker",
        yaxis_title="Expression Level",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quantum state visualization
    if result.circuit_info and 'visualization' in result.circuit_info:
        st.markdown("### Quantum State Visualization")
        st.image(
            f"data:image/png;base64,{result.circuit_info['visualization']}",
            caption="Quantum State Probability Distribution"
        )


def page_quantum_analysis():
    """Quantum analysis page content"""
    st.markdown("## Quantum Analysis")

    if not st.session_state.is_trained:
        st.warning("Please train the model first to view quantum analysis.")
        return

    # Get visualizations
    visualizations = st.session_state.ember.get_visualizations()

    st.markdown("""
    ### Understanding Quantum Computation in EMBER

    EMBER uses quantum computing principles to analyze gene expression patterns:

    1. **Feature Encoding**: Gene expression values are encoded into quantum states
    2. **Quantum Entanglement**: Captures correlations between biomarkers
    3. **Measurement**: Quantum states collapse to classification results
    """)

    st.markdown("---")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quantum Circuit",
        "Training Progress",
        "Performance Metrics",
        "Technical Details"
    ])

    with tab1:
        st.markdown("### Quantum Feature Map Circuit")
        st.markdown("""
        The feature map encodes classical gene expression data into quantum states
        using parameterized rotation gates. Each qubit represents one biomarker.
        """)

        if 'feature_map' in visualizations:
            st.image(
                f"data:image/png;base64,{visualizations['feature_map']}",
                caption="Feature Map Circuit"
            )
        else:
            # Generate a simple circuit diagram description
            st.info("Circuit visualization will be available after training with VQC or QNN classifiers.")

        if 'ansatz' in visualizations:
            st.markdown("### Variational Ansatz Circuit")
            st.image(
                f"data:image/png;base64,{visualizations['ansatz']}",
                caption="Variational Ansatz"
            )

    with tab2:
        st.markdown("### Training Progress")

        if 'training_progress' in visualizations:
            st.image(
                f"data:image/png;base64,{visualizations['training_progress']}",
                caption="Objective Function During Training"
            )
        else:
            st.info("Training progress visualization available for VQC and QNN classifiers.")

        # Show training metrics over time
        if st.session_state.ember and st.session_state.ember.diagnostic_classifier:
            callback_data = st.session_state.ember.diagnostic_classifier.training_callback_data

            if callback_data['objective']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=callback_data['objective'],
                    mode='lines+markers',
                    name='Objective Value'
                ))
                fig.update_layout(
                    title="Training Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Objective Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Performance Metrics")

        if 'confusion_matrix' in visualizations:
            st.image(
                f"data:image/png;base64,{visualizations['confusion_matrix']}",
                caption="Confusion Matrix"
            )

        # Additional metrics visualization
        if st.session_state.training_metrics:
            metrics = st.session_state.training_metrics

            # Radar chart of metrics
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill='toself',
                name='Quantum Classifier'
            ))

            if st.session_state.classical_metrics:
                c_metrics = st.session_state.classical_metrics
                c_values = [c_metrics.accuracy, c_metrics.precision, c_metrics.recall, c_metrics.f1]
                fig.add_trace(go.Scatterpolar(
                    r=c_values + [c_values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Classical SVM'
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Performance Comparison Radar Chart",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Technical Details")

        st.markdown("""
        #### Quantum Algorithm Details

        **QSVC (Quantum Support Vector Classifier)**
        - Uses quantum kernel estimation
        - Feature map: Pauli Feature Map (X, Y, Z gates)
        - Entanglement: Full connectivity
        - Complexity: O(n^2) kernel evaluations

        **VQC (Variational Quantum Classifier)**
        - Parameterized quantum circuit
        - Feature map: ZZ Feature Map
        - Ansatz: RealAmplitudes with full entanglement
        - Optimizer: COBYLA

        **QNN (Quantum Neural Network)**
        - Quantum convolutional layers
        - Pooling layers for dimensionality reduction
        - EstimatorQNN with classical optimizer
        """)

        # System information
        st.markdown("#### System Configuration")

        if st.session_state.ember and st.session_state.ember.diagnostic_classifier:
            classifier = st.session_state.ember.diagnostic_classifier

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                - **Classifier Type**: {classifier.classifier_type.upper()}
                - **Number of Qubits**: {classifier.n_features}
                - **Simulator**: Qiskit Aer
                """)

            with col2:
                st.markdown(f"""
                - **Feature Map**: {'Pauli' if classifier.classifier_type == 'qsvc' else 'ZZ'}
                - **Entanglement**: Full
                - **Shots**: 1024 (default)
                """)


def page_about():
    """About page content"""
    st.markdown("## About EMBER")

    st.markdown("""
    ### Project Overview

    **EMBER (Early Malignancy Biomarker Evaluation and Recognition)** is a quantum
    computing-based system for early lung cancer detection using gene expression data.

    ### The Problem

    Lung cancer is the most deadly form of cancer, responsible for over 2 million deaths
    annually. The challenge lies in early detection:

    - **75%** of lung cancer cases go undetected until late stages
    - Survival rates drop **57%** when detected late
    - CT scans lack efficiency and accessibility
    - Traditional ML methods suffer from overfitting and feature isolation

    ### Our Solution

    EMBER leverages quantum computing to overcome these limitations:

    1. **Quantum Superposition**: Allows simultaneous evaluation of multiple feature combinations
    2. **Quantum Entanglement**: Captures complex correlations between biomarkers
    3. **Probabilistic Output**: Provides confidence scores for diagnoses
    4. **Unitarity**: Ensures proper probability normalization, reducing overfitting

    ### Results

    On a diverse dataset of 3,924 patients:
    - **93% accuracy** in lung cancer detection
    - **8% improvement** over classical deep learning baseline (85%)
    - Works with simple, inexpensive blood tests

    ### Impact

    EMBER enables:
    - Rapid diagnosis (minutes vs. weeks for traditional staging)
    - Affordable testing (blood test vs. CT scan)
    - Accessible care for underserved populations
    - Early intervention leading to better outcomes

    ### Technology Stack

    - **Quantum Framework**: IBM Qiskit
    - **Classical ML**: Scikit-learn
    - **Web Framework**: Streamlit
    - **Visualization**: Plotly

    ### Data Sources

    1. **GEO Database (GSE137140)**: 3,924 patient miRNA expression profiles
    2. **MSK Cancer Center**: Clinical staging and survival data

    ### References

    This project was developed as part of research into quantum machine learning
    applications in healthcare. The system demonstrates the potential of quantum
    computing for medical diagnostics.
    """)

    st.markdown("---")

    st.markdown("""
    ### Contact & Resources

    For more information about EMBER or quantum computing in healthcare:

    - GitHub Repository: [EMBER Project]
    - Documentation: [Technical Documentation]
    - Research Paper: [EMBER: Quantum Computing for Lung Cancer Detection]
    """)


def main():
    """Main application entry point"""
    init_session_state()
    create_header()
    page, classifier_type, max_iter = create_sidebar()

    # Page routing
    if page == "Home":
        page_home()
    elif page == "Train Model":
        page_train_model(classifier_type, max_iter)
    elif page == "Make Prediction":
        page_make_prediction()
    elif page == "Quantum Analysis":
        page_quantum_analysis()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
