# Quick Start Guide

Get the Attack Detection System up and running in minutes!

## Prerequisites

- **Python 3.8-3.11** (check with `python3 --version`)
- **pip** (usually comes with Python)
- **macOS/Linux** (Windows works but may need adjustments)

## Step 1: Install Dependencies

### Option A: Automated Setup (Recommended) ⭐

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
chmod +x setup.sh
./setup.sh
```

The script will:
- Check Python installation
- Install all required packages
- Set up virtual environment (optional)
- Verify installation

### Option B: Manual Setup

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_streamlit.txt
```

**Note:** If TensorFlow installation fails (especially on Apple Silicon), see `macOS_SETUP.md` for specific instructions.

## Step 2: Verify Models Are Present

Check that these files exist in the project directory:
- ✅ `att_det_mlp.h5` - MLP model (required)
- ✅ `att_det_lstm.h5` - LSTM model (optional)
- ✅ `onehot_encoder.pkl` - Feature encoder
- ✅ `label_binarizer.pkl` - Label encoder
- ✅ `beh_lstm.h5` - Behavioral LSTM (optional)

If models are missing, you can train them (see Step 4).

## Step 3: Start the Application

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# Activate virtual environment if you created one
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start Streamlit
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 4: Using the Application

### Tab 1: Batch CSV Analysis

1. Go to **"📦 Batch CSV"** tab
2. Upload a CSV file with KDD features (or use `KDDTest_plus.csv`)
3. Click **"🚀 Run Prediction"**
4. View results, download predictions

### Tab 2: Live Capture

**Option A: Wireshark (Recommended - No special permissions)**

1. Install Wireshark/tshark:
   ```bash
   # macOS
   brew install wireshark
   
   # Linux
   sudo apt-get install tshark
   ```

2. In Streamlit:
   - Select **"Wireshark (tshark)"** capture method
   - Choose interface (e.g., `en0` on macOS)
   - Click **"▶️ Start Sniffing"**
   - Click **"🔄 Refresh now"** to see traffic

**Option B: Scapy (Requires sudo)**

```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

Then select "Scapy (Python)" method in the UI.

### Tab 3: Training

1. Go to **"🎓 Training"** tab
2. Choose model type (MLP, LSTM, or Behavioral LSTM)
3. Configure parameters
4. Click **"🚀 Start Training"**

## Quick Test

Test the system with sample data:

```bash
# From command line
python attack_detection_pipeline.py --artifact_dir . --csv KDDTest_plus.csv
```

Or upload `KDDTest_plus.csv` in the Batch CSV tab.

## Troubleshooting

### "Module not found" errors

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements_streamlit.txt
```

### "Model not found" errors

Train the models first:
1. Go to Training tab
2. Train MLP model (required)
3. Train LSTM model (optional)

### "tshark not found" (for Wireshark capture)

Install Wireshark:
```bash
# macOS
brew install wireshark

# Linux
sudo apt-get install tshark
```

### "Permission denied" (for live capture)

**macOS:** Grant network permissions in System Preferences → Security & Privacy → Privacy → Network

**Linux:** Add user to wireshark group:
```bash
sudo usermod -aG wireshark $USER
# Then logout and login again
```

## What's Next?

- **Batch Analysis**: Upload CSV files for offline analysis
- **Live Monitoring**: Capture and analyze network traffic in real-time
- **Train Models**: Create custom models from your data
- **Continuous Learning**: Automatically retrain from live data

## Additional Resources

- `README.md` - Complete project documentation
- `macOS_SETUP.md` - macOS-specific setup instructions
- `TRAINING_GUIDE.md` - Detailed training instructions
- `LIVE_WIRESHARK_GUIDE.md` - Live capture setup
- `DIRECT_TRAINING_GUIDE.md` - Train from live capture
- `WIRESHARK_GUIDE.md` - Process PCAP files

## Project Structure

```
Major/
├── streamlit_app.py          # Main application (START HERE)
├── attack_detection_pipeline.py  # Core detection engine
├── train_*.py                # Training scripts
├── *.h5                      # Trained models
├── *.pkl                     # Preprocessing artifacts
├── archive/                  # Training data (KDD dataset)
└── requirements_streamlit.txt # Dependencies
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the relevant guide (macOS_SETUP.md, TRAINING_GUIDE.md, etc.)
3. Verify all dependencies are installed
4. Ensure models are present or train them first

---

**Ready to start?** Run: `streamlit run streamlit_app.py` 🚀

