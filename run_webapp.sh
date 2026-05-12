#!/bin/bash
# EMBER Web Application Startup Script

echo "=============================================="
echo "   EMBER - Quantum Lung Cancer Detection"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import streamlit; import qiskit; import plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
echo ""
echo "Starting EMBER Web Application..."
echo "Access the application at: http://localhost:8501"
echo ""

cd webapp
streamlit run app.py --server.headless=true
