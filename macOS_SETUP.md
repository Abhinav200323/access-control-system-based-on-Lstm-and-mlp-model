# macOS Setup Guide - Complete Instructions

This guide provides step-by-step instructions to set up and run the Attack Detection System on macOS.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Running the Application](#running-the-application)
4. [Network Permissions for Live Capture](#network-permissions-for-live-capture)
5. [Troubleshooting](#troubleshooting)
6. [Quick Reference](#quick-reference)

---

## Prerequisites

### 1. Check macOS Version
- **macOS 10.15 (Catalina)** or later recommended
- Check your version: `sw_vers`

### 2. Check Python Installation

```bash
# Check if Python 3 is installed
python3 --version
```

**Expected output:** Python 3.8, 3.9, 3.10, or 3.11

**If Python is not installed:**

#### Option A: Install via Homebrew (Recommended)
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3
```

#### Option B: Download from Python.org
1. Visit: https://www.python.org/downloads/
2. Download Python 3.11 for macOS
3. Run the installer

### 3. Check if Homebrew is Installed (Optional but Recommended)

```bash
brew --version
```

If not installed, install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## Step-by-Step Setup

### Step 1: Navigate to Project Directory

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**You should see `(venv)` in your terminal prompt.**

**Note:** Always activate the virtual environment before running the application:
```bash
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

#### For Intel Macs (x86_64):

```bash
pip install -r requirements_streamlit.txt
```

#### For Apple Silicon (M1/M2/M3/M4):

TensorFlow installation may take longer. If you encounter issues:

```bash
# First, install TensorFlow for macOS
pip install tensorflow-macos

# Optional: For GPU acceleration (Metal)
pip install tensorflow-metal

# Then install other dependencies
pip install -r requirements_streamlit.txt
```

**Alternative (if above fails):**
```bash
# Install TensorFlow 2.13 or later (supports Apple Silicon natively)
pip install "tensorflow>=2.13.0"

# Then install other dependencies
pip install -r requirements_streamlit.txt
```

### Step 5: Install Wireshark (Optional - for Live Capture)

Wireshark is recommended for live network capture (no sudo needed):

```bash
# Install via Homebrew
brew install wireshark

# Verify installation
tshark --version
```

**Note:** If you prefer not to use Wireshark, you can use Scapy (requires sudo - see Network Permissions section).

### Step 6: Verify Installation

```bash
# Check installed packages
pip list | grep -E "streamlit|tensorflow|scapy|numpy|pandas"

# Expected output should show:
# streamlit
# tensorflow (or tensorflow-macos)
# scapy
# numpy
# pandas
```

---

## Running the Application

### Basic Run (No Live Capture)

```bash
# 1. Navigate to project directory
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run Streamlit
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### With Live Capture (See Network Permissions below)

**Option 1: With Network Permissions (Recommended)**
```bash
# After granting network permissions (see below)
streamlit run streamlit_app.py
```

**Option 2: With Sudo**
```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

---

## Network Permissions for Live Capture

Live network capture requires special permissions. You have two options:

### Option 1: Grant Network Permissions (Recommended) ⭐

This avoids needing sudo every time:

1. **Open System Settings** (macOS Ventura+) or **System Preferences** (older macOS)
   - Press `⌘ + Space` and search for "System Settings" or "System Preferences"

2. **Navigate to Privacy Settings**
   - **macOS Ventura+:** System Settings → Privacy & Security → Network
   - **Older macOS:** System Preferences → Security & Privacy → Privacy → Network

3. **Enable Network Access**
   - Click the lock icon (bottom left) and enter your password
   - Check the box next to your Terminal app:
     - **Terminal** (default macOS terminal)
     - **iTerm2** (if you use iTerm2)
     - **VS Code** (if running from VS Code terminal)

4. **Run Streamlit Normally**
   ```bash
   streamlit run streamlit_app.py
   ```

**Benefits:**
- ✅ No sudo password prompts
- ✅ More secure (only grants network access, not full admin)
- ✅ Works seamlessly with Streamlit
- ✅ No need to remember sudo commands

### Option 2: Use Sudo (Alternative)

If you prefer not to grant network permissions:

```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

You'll be prompted for your admin password each time.

**Note:** Training models does NOT require sudo or network permissions!

---

## Troubleshooting

### Issue 1: "Command not found: python3"

**Solution:**
```bash
# Check if Python is installed
which python3

# If not found, install via Homebrew
brew install python3

# Or add to PATH (if installed but not in PATH)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Issue 2: "Module not found" errors

**Solution:**
```bash
# 1. Ensure virtual environment is activated
source venv/bin/activate

# 2. Verify you're in the project directory
pwd
# Should show: /Users/abhin/Desktop/ai cyber/Major/Major

# 3. Reinstall dependencies
pip install --upgrade -r requirements_streamlit.txt
```

### Issue 3: TensorFlow Installation Fails (Apple Silicon)

**Solution:**
```bash
# For Apple Silicon (M1/M2/M3/M4)
pip install tensorflow-macos tensorflow-metal

# Then install other dependencies
pip install -r requirements_streamlit.txt
```

### Issue 4: "Permission denied" for Live Capture

**Solution:**
- **Option A:** Grant network permissions (see Network Permissions section above)
- **Option B:** Use sudo: `sudo env PATH="$PATH" streamlit run streamlit_app.py`

### Issue 5: "tshark not found" (Wireshark)

**Solution:**
```bash
# Install Wireshark
brew install wireshark

# Verify installation
tshark --version
```

### Issue 6: Network Interface Not Found

**Solution:**
```bash
# List available network interfaces
ifconfig | grep "^[a-z]"

# Common macOS interfaces:
# en0 - Primary Ethernet/WiFi
# en1 - Secondary interface
# lo0 - Loopback

# In Streamlit, use the interface name (e.g., "en0")
```

### Issue 7: Homebrew Python Path Issues

**Solution:**
```bash
# Add Homebrew to PATH (for Apple Silicon)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc

# For Intel Macs
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Verify
which python3
```

### Issue 8: Virtual Environment Not Activating

**Solution:**
```bash
# Remove old virtual environment
rm -rf venv

# Create new one
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python
```

### Issue 9: Port 8501 Already in Use

**Solution:**
```bash
# Option 1: Use different port
streamlit run streamlit_app.py --server.port 8502

# Option 2: Kill process using port 8501
lsof -ti:8501 | xargs kill -9

# Then run normally
streamlit run streamlit_app.py
```

### Issue 10: "Model not found" errors

**Solution:**
1. Check that model files exist:
   ```bash
   ls -la *.h5 *.pkl
   ```

2. If missing, train models:
   - Run Streamlit: `streamlit run streamlit_app.py`
   - Go to **"🎓 Training"** tab
   - Train MLP model (required)
   - Train LSTM model (optional)

---

## Quick Reference

### Daily Usage Commands

```bash
# 1. Navigate to project
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run application
streamlit run streamlit_app.py
```

### Check Installation

```bash
# Check Python version
python3 --version

# Check packages
pip list | grep -E "streamlit|tensorflow"

# Check models
ls -la *.h5 *.pkl
```

### Common Interface Names (macOS)

- `en0` - Primary Ethernet/WiFi (most common)
- `en1` - Secondary interface
- `lo0` - Loopback (localhost)
- `bridge0` - Bridge interface

### File Structure

```
Major/
├── venv/                    # Virtual environment
├── streamlit_app.py         # Main application
├── *.h5                     # Trained models
├── *.pkl                    # Preprocessing artifacts
├── archive/                 # Training data
└── requirements_streamlit.txt
```

---

## Next Steps

1. ✅ **Setup Complete** - Dependencies installed
2. ✅ **Network Permissions** - Granted (if using live capture)
3. 🚀 **Run Application** - `streamlit run streamlit_app.py`
4. 📦 **Test Batch CSV** - Upload a CSV file in Batch CSV tab
5. 🛰️ **Test Live Capture** - Try capturing network traffic
6. 🎓 **Train Models** - Train custom models in Training tab

---

## Additional Resources

- `QUICK_START.md` - Quick start guide
- `README.md` - Complete project documentation
- `TRAINING_GUIDE.md` - Detailed training instructions
- `LIVE_WIRESHARK_GUIDE.md` - Live capture setup
- `WIRESHARK_GUIDE.md` - Process PCAP files

---

## Support

If you encounter issues not covered here:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all prerequisites are met
4. Ensure virtual environment is activated
5. Check that models are present (or train them)

---

**Ready to start?** Run: `streamlit run streamlit_app.py` 🚀
