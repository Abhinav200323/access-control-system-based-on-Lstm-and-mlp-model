# Steps to Run on macOS - Complete Guide

## 🚀 Quick Start (3 Steps)

```bash
# Step 1: Navigate to project
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# Step 2: Run setup (first time only)
./setup.sh

# Step 3: Run application
./run_mac.sh
```

**That's it!** The app opens at `http://localhost:8501`

---

## 📋 Detailed Steps

### Step 1: Check Prerequisites

```bash
# Check Python version (need 3.8-3.11)
python3 --version

# If not installed, install via Homebrew:
brew install python3
```

### Step 2: Setup Project

**Option A: Automated Setup (Recommended)**
```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
./setup.sh
```

**Option B: Manual Setup**
```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements_streamlit.txt
```

**For Apple Silicon (M1/M2/M3/M4):**
```bash
# After activating venv
pip install tensorflow-macos tensorflow-metal
pip install -r requirements_streamlit.txt
```

### Step 3: Grant Network Permissions (Optional - for Live Capture)

1. Open **System Settings** (⌘ + Space → "System Settings")
2. Go to **Privacy & Security** → **Network**
3. Click lock icon, enter password
4. Check box next to **Terminal**
5. Done! Now you can use live capture without sudo

**Note:** This is optional. Training and batch CSV don't need this.

### Step 4: Run Application

**Option A: Using Run Script (Easiest)**
```bash
./run_mac.sh
```

**Option B: Manual Run**
```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run streamlit_app.py
```

**Option C: Run with Scapy Support (Requires Sudo)**
```bash
# Using the Scapy script (recommended)
./run_scapy.sh

# Or manually:
source venv/bin/activate
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**Note:** 
- Use `./run_mac.sh` for normal operation (Wireshark method recommended)
- Use `./run_scapy.sh` if you want to use Scapy for live capture (requires sudo)
- Scapy requires sudo because it needs access to network interfaces

### Step 5: Use the Application

1. **Batch CSV Tab:**
   - Upload a CSV file
   - Click "Run Prediction"
   - View and download results

2. **Live Capture Tab:**
   - Select capture method (Wireshark recommended)
   - Choose interface (e.g., `en0`)
   - Click "Start Sniffing"
   - Click "Refresh now" to see traffic

3. **Training Tab:**
   - Train MLP, LSTM, or Behavioral LSTM models
   - Configure parameters
   - Click "Start Training"

---

## 🔧 Troubleshooting

### Issue: "Command not found: python3"
```bash
brew install python3
```

### Issue: "Module not found"
```bash
source venv/bin/activate
pip install -r requirements_streamlit.txt
```

### Issue: TensorFlow fails on Apple Silicon
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements_streamlit.txt
```

### Issue: "Permission denied" for live capture
- Grant network permissions (see Step 3 above)
- OR use: `sudo env PATH="$PATH" streamlit run streamlit_app.py`

### Issue: "tshark not found"
```bash
brew install wireshark
```

### Issue: Port 8501 in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📁 Project Structure

```
Major/
├── run_mac.sh              # Quick run script
├── setup.sh                # Setup script
├── streamlit_app.py        # Main application
├── venv/                   # Virtual environment (created)
├── *.h5                    # Trained models
├── *.pkl                   # Preprocessing files
├── archive/                # Training data
└── requirements_streamlit.txt
```

---

## ✅ Checklist

- [ ] Python 3.8-3.11 installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Models present (or ready to train)
- [ ] Network permissions granted (for live capture)
- [ ] Application runs successfully

---

## 📚 Additional Resources

- `README_MACOS.md` - Quick macOS reference
- `macOS_SETUP.md` - Complete macOS setup guide
- `QUICK_START.md` - General quick start
- `README.md` - Full documentation

---

## 🎯 Common Commands

```bash
# Daily use
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
source venv/bin/activate
streamlit run streamlit_app.py

# Or use the run script
./run_mac.sh

# Check installation
pip list | grep streamlit
python3 --version

# List network interfaces
ifconfig | grep "^[a-z]"
```

---

**Ready to start?** Run `./run_mac.sh` 🚀
