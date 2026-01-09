#!/bin/bash
# Setup script for Attack Detection System
# Handles dependency installation and sudo requirements

set -e

echo "🔧 Attack Detection System - Setup Script"
echo "=========================================="
echo ""

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    OS="Unknown"
    echo "⚠️  Warning: Unknown OS type. Proceeding with generic setup."
fi

echo "Detected OS: $OS"
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    PYTHON_PATH=$(which python3)
    echo "✅ Found: $PYTHON_VERSION"
    echo "   Location: $PYTHON_PATH"
    
    # Check if it's Homebrew Python (common on macOS)
    if [[ "$PYTHON_PATH" == *"brew"* ]] || [[ "$PYTHON_PATH" == *"Cellar"* ]]; then
        echo "   ℹ️  Using Homebrew Python (recommended for macOS)"
    fi
else
    echo "❌ Python 3 not found."
    if [[ "$OS" == "macOS" ]]; then
        echo ""
        echo "To install Python on macOS:"
        echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "  2. Install Python: brew install python3"
        echo "  3. Or download from: https://www.python.org/downloads/"
    fi
    exit 1
fi

# Check pip
echo ""
echo "Checking pip installation..."
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "✅ pip found"
    PIP_CMD="pip"
else
    echo "❌ pip not found. Installing pip..."
    if [[ "$OS" == "macOS" ]]; then
        echo "Please install pip manually or use: python3 -m ensurepip --upgrade"
    elif [[ "$OS" == "Linux" ]]; then
        echo "Installing pip (may require sudo)..."
        sudo apt-get update
        sudo apt-get install -y python3-pip
        PIP_CMD="pip3"
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This typically does NOT require sudo (uses --user flag if needed)"
echo ""

# Check if we're on Apple Silicon
if [[ "$OS" == "macOS" ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo "🍎 Detected Apple Silicon (M1/M2/M3/M4)"
        echo "   Installing TensorFlow for macOS first..."
        if $PIP_CMD install tensorflow-macos; then
            echo "   ✅ TensorFlow for macOS installed"
            # Optional: Install Metal for GPU acceleration
            read -p "   Install TensorFlow Metal (GPU acceleration)? [y/N]: " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                $PIP_CMD install tensorflow-metal
                echo "   ✅ TensorFlow Metal installed"
            fi
        else
            echo "   ⚠️  TensorFlow installation failed, will try standard TensorFlow"
        fi
    fi
fi

# Try installing without sudo first
if $PIP_CMD install -r requirements_streamlit.txt; then
    echo "✅ Dependencies installed successfully!"
else
    echo "⚠️  Installation failed. Trying with --user flag..."
    if $PIP_CMD install --user -r requirements_streamlit.txt; then
        echo "✅ Dependencies installed successfully (user install)!"
    else
        echo "❌ Installation failed. You may need to:"
        echo "   1. Check your internet connection"
        echo "   2. Update pip: $PIP_CMD install --upgrade pip"
        echo "   3. Use a virtual environment: python3 -m venv venv && source venv/bin/activate"
        if [[ "$OS" == "macOS" && "$ARCH" == "arm64" ]]; then
            echo "   4. For Apple Silicon, try: pip install tensorflow-macos tensorflow-metal"
        fi
        exit 1
    fi
fi

# Check for sudo (for live capture only)
echo ""
echo "Checking sudo access (needed only for live network capture)..."
if sudo -n true 2>/dev/null; then
    echo "✅ Sudo access available (passwordless)"
elif sudo -v 2>/dev/null; then
    echo "✅ Sudo access available (will prompt for password when needed)"
else
    echo "⚠️  Sudo access not available. This is OK for training and batch prediction."
    echo "   Live network capture will require sudo privileges."
fi

# Create virtual environment (optional but recommended)
echo ""
read -p "Create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
        echo ""
        echo "To activate:"
        echo "  source venv/bin/activate  # macOS/Linux"
        echo "  venv\\Scripts\\activate     # Windows"
        echo ""
        echo "Then install dependencies:"
        echo "  pip install -r requirements_streamlit.txt"
    else
        echo "✅ Virtual environment already exists"
    fi
fi

# macOS-specific network permissions info
if [[ "$OS" == "macOS" ]]; then
    echo ""
    echo "📱 macOS Network Permissions:"
    echo "=============================="
    echo "For live network capture, you have two options:"
    echo ""
    echo "Option 1: Grant network permissions (recommended) ⭐"
    MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
    if [[ "$MACOS_VERSION" -ge 13 ]]; then
        echo "  1. Open: System Settings → Privacy & Security → Network"
    else
        echo "  1. Open: System Preferences → Security & Privacy → Privacy → Network"
    fi
    echo "  2. Click the lock icon (bottom left) and enter your password"
    echo "  3. Check the box next to your Terminal app (Terminal, iTerm2, etc.)"
    echo "  4. Then run: streamlit run streamlit_app.py"
    echo ""
    echo "Option 2: Use sudo (alternative)"
    echo "  sudo env PATH=\"\$PATH\" streamlit run streamlit_app.py"
    echo ""
    echo "Note: Training models does NOT require network permissions or sudo!"
    echo ""
fi

# Instructions for running Streamlit
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the Streamlit app:"
if [[ "$OS" == "macOS" ]]; then
    echo "  streamlit run streamlit_app.py"
    echo ""
    echo "If you granted network permissions (see above), you can use live capture without sudo."
    echo "Otherwise, for live network capture:"
    echo "  sudo env PATH=\"\$PATH\" streamlit run streamlit_app.py"
else
    echo "  streamlit run streamlit_app.py"
    echo ""
    echo "For live network capture (requires sudo):"
    echo "  sudo env PATH=\"\$PATH\" streamlit run streamlit_app.py"
fi
echo ""
echo "Note: Training models does NOT require sudo."
echo "      Only live network capture needs elevated privileges."
echo ""

