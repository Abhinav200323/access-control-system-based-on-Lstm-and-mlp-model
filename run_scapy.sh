#!/bin/bash
# macOS Run Script for Attack Detection System with Scapy
# This script runs Streamlit with sudo to enable Scapy packet capture

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛰️  Attack Detection System - Scapy Mode (Requires Sudo)${NC}"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found.${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_streamlit.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
fi

# Check if Scapy is installed
source venv/bin/activate
if python3 -c "import scapy" 2>/dev/null; then
    echo -e "${GREEN}✅ Scapy is installed${NC}"
else
    echo -e "${YELLOW}⚠️  Scapy not found. Installing...${NC}"
    pip install scapy
    echo -e "${GREEN}✅ Scapy installed${NC}"
fi

# Check if models exist
if [ ! -f "att_det_mlp.h5" ]; then
    echo -e "${YELLOW}⚠️  Warning: att_det_mlp.h5 not found${NC}"
    echo "   You can train models in the Training tab after starting the app"
fi

echo ""
echo -e "${YELLOW}⚠️  Note: This requires sudo for network packet capture${NC}"
echo "   You will be prompted for your admin password"
echo ""
echo "=========================================="
echo -e "${GREEN}Starting Streamlit with Scapy support...${NC}"
echo "=========================================="
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo -e "${BLUE}In the Live Capture tab:${NC}"
echo "  - Select 'Scapy (Python)' as the capture method"
echo "  - Choose your network interface (e.g., en0)"
echo "  - Click '▶️ Start Sniffing'"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Get the Python path from venv
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Run Streamlit with sudo, preserving PATH
sudo env PATH="$PATH" "$VENV_PYTHON" -m streamlit run streamlit_app.py
