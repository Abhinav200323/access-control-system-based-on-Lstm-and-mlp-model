# Training Guide - Attack Detection System

This guide explains how to train models from the Streamlit interface and handle system requirements.

## Quick Start

### 1. Install Dependencies

**Option A: Using the setup script (Recommended)**
```bash
./setup.sh
```

**Option B: Manual installation**
```bash
# Install Python packages (usually no sudo needed)
pip install -r requirements_streamlit.txt

# If that fails, try user installation:
pip install --user -r requirements_streamlit.txt

# Or use a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
pip install -r requirements_streamlit.txt
```

### 2. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Navigate to the **"🎓 Training"** tab to train models.

## Training Models from Streamlit

### MLP Training

1. Go to the **Training** tab → **MLP Training** sub-tab
2. Specify your training and validation data files (KDD format)
3. Set hyperparameters:
   - Epochs (default: 15)
   - Batch size (default: 256)
   - Hidden units (default: 256)
   - Dropout rate (default: 0.3)
4. Optionally enable hyperparameter tuning
5. Click **"🚀 Start MLP Training"**

The model will be saved as `att_det_mlp.h5` (or your specified filename).

### LSTM Training

1. Go to the **Training** tab → **LSTM Training** sub-tab
2. Specify your training and validation data files
3. Set hyperparameters:
   - Epochs (default: 8)
   - Batch size (default: 256)
   - LSTM units (default: 64)
4. Click **"🚀 Start LSTM Training"**

The model will be saved as `att_det_lstm.h5` along with a threshold file `att_det_lstm.threshold.json`.

### Behavioral LSTM Training

**Prerequisites:**
1. First, build sequences from your data
2. Then train the behavioral LSTM model

**Steps:**

1. Go to the **Training** tab → **Behavioral LSTM Training** sub-tab
2. **Build Sequences First** (if you haven't already):
   - Expand the "🔨 Build Sequences First" section
   - Specify your input CSV file
   - Set sequence length (timesteps, default: 10)
   - Click **"🔨 Build Sequences"**
   - This creates `X_seq.npy` and `y_seq.npy`

3. **Train the Model:**
   - Specify the sequence files (`X_seq.npy`, `y_seq.npy`)
   - Set epochs (default: 20) and batch size (default: 64)
   - Click **"🚀 Start Behavioral LSTM Training"**

The model will be saved as `beh_lstm.h5`.

## Sudo Requirements

### When is sudo needed?

**❌ NOT needed for:**
- Training models (MLP, LSTM, Behavioral LSTM)
- Batch CSV prediction
- Installing Python packages (usually)
- Running Streamlit in normal mode

**✅ Needed for:**
- Live network capture (requires root/admin privileges to access network interfaces)

### Running with Sudo (for Live Capture)

**macOS - Option 1 (Recommended): Grant Network Permissions**
1. Open **System Preferences** (or **System Settings** on macOS Ventura+)
2. Go to **Security & Privacy** → **Privacy** → **Network**
3. Enable your Terminal app (Terminal, iTerm2, etc.) to access the network
4. Run Streamlit normally:
   ```bash
   streamlit run streamlit_app.py
   ```
   No sudo needed!

**macOS - Option 2: Use Sudo**
```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**Linux:**
```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**Why?** Network packet capture requires access to network interfaces, which typically requires elevated privileges. On macOS, granting network permissions is preferred over using sudo.

### Installing System Packages (if needed)

If you need to install system-level packages:

**macOS:**
```bash
# Check if Homebrew is installed
brew --version

# If not installed, install Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python (if needed)
brew install python3

# Note: macOS comes with Python 3, but Homebrew Python is recommended
# for better package management and compatibility
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
```

**Note:** Python package installation (`pip install`) typically does NOT require sudo. Use `--user` flag or virtual environments instead.

## Troubleshooting

### "Permission denied" errors

- **For training:** This shouldn't happen. Check file permissions on your data files.
- **For live capture:** Use sudo as described above.

### "Module not found" errors

1. Ensure dependencies are installed: `pip install -r requirements_streamlit.txt`
2. If using a virtual environment, make sure it's activated
3. Try: `pip install --upgrade pip` then reinstall

### Training fails

1. Check that your data files exist and are in the correct format (KDD format)
2. Verify file paths are correct (relative to the script directory)
3. Check the error output in the Streamlit interface
4. Ensure you have enough disk space for model files

### "Unknown layer: 'NotEqual'" or "Behavioral LSTM check failed"

This is a TensorFlow version compatibility issue. The model was saved with a different TensorFlow version than what you're using.

**Solution:** Retrain the behavioral LSTM model:
1. Go to the **Training** tab → **Behavioral LSTM Training**
2. Build sequences (if needed): Click "🔨 Build Sequences First"
3. Train the model: Click "🚀 Start Behavioral LSTM Training"

Or from command line:
```bash
python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --out beh_lstm.h5
```

**Prevention:** Always train models with the same TensorFlow version you'll use for inference.

### Sudo password prompts

- If you see frequent sudo prompts, consider setting up passwordless sudo for your user (advanced, use with caution)
- For live capture, you only need sudo when starting Streamlit with network capture enabled

## File Structure

After training, you should have:
```
.
├── att_det_mlp.h5              # MLP model
├── att_det_lstm.h5             # LSTM model
├── att_det_lstm.threshold.json # LSTM threshold
├── beh_lstm.h5                 # Behavioral LSTM model
├── onehot_encoder.pkl          # Preprocessing encoder
└── label_binarizer.pkl         # Label encoder
```

## Command Line Training (Alternative)

If you prefer command line over Streamlit:

```bash
# Train MLP
python train_mlp.py --train archive/KDDTrain+_20Percent.txt --val archive/KDDTest+.txt

# Train LSTM
python train_lstm.py --train archive/KDDTrain+_20Percent.txt --val archive/KDDTest+.txt

# Build sequences
python build_beh_sequences.py --csv KDDTest_plus.csv

# Train Behavioral LSTM
python train_beh_lstm.py --x X_seq.npy --y y_seq.npy
```

## Next Steps

After training:
1. Models are automatically saved to your artifact directory
2. Click **"Load / Reload Detector"** in the Streamlit sidebar to load new models
3. Use the **Batch CSV** or **Live Capture** tabs to test your models

