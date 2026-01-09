# Quick Start for macOS Users 🍎

## Fastest Way to Get Started

### Option 1: Automated Setup & Run (Easiest) ⭐

```bash
# 1. Navigate to project directory
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# 2. Run setup script
./setup.sh

# 3. Run the application
./run_mac.sh
```

That's it! The app will open in your browser.

### Option 2: Manual Setup

```bash
# 1. Navigate to project
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements_streamlit.txt

# 4. Run application
streamlit run streamlit_app.py
```

## For Apple Silicon (M1/M2/M3/M4) Users

If you have an Apple Silicon Mac, TensorFlow installation is slightly different:

```bash
# After creating virtual environment
source venv/bin/activate

# Install TensorFlow for macOS
pip install tensorflow-macos

# Optional: For GPU acceleration
pip install tensorflow-metal

# Then install other dependencies
pip install -r requirements_streamlit.txt
```

## Using Scapy (Requires Sudo)

If you want to use Scapy for live capture:

```bash
# Run with Scapy support
./run_scapy.sh

# Or manually:
source venv/bin/activate
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**Note:** Scapy requires sudo. For no-sudo option, use Wireshark method instead.
See `SCAPY_USAGE.md` for detailed Scapy instructions.

## Network Permissions (For Live Capture)

To use live network capture without sudo:

1. **Open System Settings** (⌘ + Space, search "System Settings")
2. Go to **Privacy & Security** → **Network**
3. Click lock icon, enter password
4. Check box next to **Terminal** (or your terminal app)
5. Run normally: `streamlit run streamlit_app.py`

**Note:** Training models does NOT need network permissions!

## What You Can Do

- ✅ **Batch CSV Analysis** - Upload CSV files for offline analysis
- ✅ **Live Network Capture** - Real-time traffic monitoring (after granting permissions)
- ✅ **Train Models** - Train MLP, LSTM, or Behavioral LSTM models
- ✅ **Continuous Learning** - Auto-retrain from live data

## Troubleshooting

### "Command not found: python3"
```bash
brew install python3
```

### "Module not found"
```bash
source venv/bin/activate
pip install -r requirements_streamlit.txt
```

### TensorFlow issues on Apple Silicon
```bash
pip install tensorflow-macos tensorflow-metal
```

### Port 8501 already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Full Documentation

For detailed instructions, see:
- `macOS_SETUP.md` - Complete setup guide
- `QUICK_START.md` - Quick start guide
- `README.md` - Full project documentation

---

**Ready?** Run: `./run_mac.sh` or `streamlit run streamlit_app.py` 🚀
