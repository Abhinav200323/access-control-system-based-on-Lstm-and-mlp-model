#!/bin/bash
# macOS Run Script for Attack Detection System
# This script makes it easy to run the application on macOS

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Attack Detection System - macOS Launcher${NC}"
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

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if models exist
if [ ! -f "att_det_mlp.h5" ]; then
    echo -e "${YELLOW}⚠️  Warning: att_det_mlp.h5 not found${NC}"
    echo "   You can train models in the Training tab after starting the app"
fi

# Check for Wireshark
if command -v tshark &> /dev/null; then
    echo -e "${GREEN}✅ Wireshark (tshark) found${NC}"
else
    echo -e "${YELLOW}⚠️  Wireshark not found (optional)${NC}"
    echo "   Install with: brew install wireshark"
    echo "   Or use Scapy method (requires sudo)"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Starting Streamlit application...${NC}"
echo "=========================================="
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run Streamlit
streamlit run streamlit_app.py
