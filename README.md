## Attack Detection – NSL-KDD / KDD'99 (MLP + LSTM + Streamlit)

This project is an **intrusion / attack detection system** based on the NSL‑KDD / KDD'99 feature set.  
It provides:

- **Offline attack detection pipeline** (`attack_detection_pipeline.py`)
- **Interactive web UI** for batch CSV inference and live packet capture (`streamlit_app.py`)
- **Simple blocking policy** abstraction (`blocking_policy.py`)
- **Scapy-based feature extractor** for real-time flows (`scapy_featurizer.py`)

### 1. Project structure (key files)

- `attack_detection_pipeline.py` – core `AttackDetector` class:
  - Loads trained artifacts from an **artifact directory**:
    - `att_det_mlp.h5` – main neural MLP model (required)
    - `att_det_lstm.h5` – LSTM model (optional)
    - `onehot_encoder.pkl` – encoder for categorical features
    - `label_binarizer.pkl` – maps class indices to labels (`normal`, attack types, etc.)
    - `att_det_lstm.threshold.json` – optional learned threshold for LSTM
  - Exposes:
    - `required_input_columns()` – first 41 KDD'99 features
    - `predict(X)` – returns predicted labels and probabilities
    - `predict_with_revoke(X, threshold, risky_labels)` – adds a **revoke** flag per row

- `streamlit_app.py` – main UI:
  - **Batch CSV tab**:
    - Upload a CSV with the 41 required KDD features.
    - Get predictions, revoke flags, probability histograms, and a downloadable CSV.
  - **Live Capture tab**:
    - **Wireshark (tshark) mode**: Real-time capture using tshark (no special permissions needed) ⭐
    - **Scapy mode**: Alternative capture using Scapy (requires root/admin)
    - Converts packets to KDD features in real-time
    - Displays live flows and model decisions
    - Supports behavioral LSTM for per-user analysis
  - **Training tab** (NEW):
    - Train MLP, LSTM, and Behavioral LSTM models directly from the UI.
    - Configure hyperparameters and start training with a single click.
    - **Wireshark PCAP processing**: Upload PCAP files from Wireshark and automatically convert to KDD features for training.
    - No sudo required for training (only needed for live capture).

- `blocking_policy.py` – department-aware risk thresholds (`BlockingPolicy`).
- `scapy_featurizer.py` – Scapy-based packet featurizer.
- `KDDTest_plus.csv`, `KDDTest_plus_normal.csv` – example NSL‑KDD-derived test data.

### 2. Requirements

Use the provided `requirements_streamlit.txt`:

- **Python**: 3.8–3.11 recommended.
- Required packages (simplified):
  - `streamlit`
  - `numpy`, `pandas`, `scipy`
  - `scikit-learn`, `joblib`
  - `tensorflow==2.*` (for `att_det_mlp.h5` / `att_det_lstm.h5`)
  - `scapy` (for live capture)

### 3. Setup (recommended)

**macOS users:** See `macOS_SETUP.md` for detailed macOS-specific instructions.

**Option A: Using the setup script (easiest)**
```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
./setup.sh
```

**Option B: Manual setup**
From the project root:

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r "requirements_streamlit.txt"
```

If TensorFlow installation fails on your platform (e.g., Apple Silicon), install the appropriate wheel or use a compatible distribution following TensorFlow's official docs, then re-run:

```bash
pip install -r "requirements_streamlit.txt"
```

**Note:** Training models does NOT require sudo. Only live network capture requires elevated privileges.

### 4. Running the Streamlit app (main way to use the project)

From the project root, with the virtual environment activated:

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
streamlit run streamlit_app.py
```

In the app:

- **Artifact directory** (sidebar): leave at the default (`os.getcwd()`) so it finds:
  - `att_det_mlp.h5`
  - `att_det_lstm.h5`
  - `onehot_encoder.pkl`
  - `label_binarizer.pkl`
  - `att_det_lstm.threshold.json`

- **Batch CSV**:
  - Upload a CSV with at least the 41 required KDD features (names must match).
  - Run prediction, inspect revoke decisions, and download results.

- **Live Capture**:
  - **macOS (Recommended):** Grant network permissions in System Preferences → Security & Privacy → Privacy → Network, then run normally
  - **macOS/Linux (Alternative):** Requires sudo for network access:
    ```bash
    sudo env PATH="$PATH" streamlit run streamlit_app.py
    ```
  - Choose:
    - Interface: e.g., `en0` (macOS), `eth0`, `wlan0`
    - Optional BPF filter: e.g., `tcp`, `port 80`
  - **Continuous Learning (NEW)**: Automatically collect live data, train behavioral LSTM, and cycle continuously. See `CONTINUOUS_LEARNING.md` for details.

- **Training**:
  - Train MLP, LSTM, or Behavioral LSTM models directly from the UI.
  - Specify training/validation data files and hyperparameters.
  - Models are saved automatically to your artifact directory.
  - See `TRAINING_GUIDE.md` for detailed instructions.

### 5. Running the pipeline from the command line (offline CSV)

You can call the detector without the UI:

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
python attack_detection_pipeline.py --artifact_dir . --csv KDDTest_plus.csv
```

This will:

- Load artifacts from `--artifact_dir` (here: current folder).
- Read the CSV.
- Run `AttackDetector.predict`.
- Print a short JSON summary (predictions and probability shape) to stdout.

### 6. Training Models

You can train models in two ways:

**Option A: From Streamlit UI (Recommended)**
1. Run the Streamlit app: `streamlit run streamlit_app.py`
2. Navigate to the **"🎓 Training"** tab
3. Choose MLP, LSTM, or Behavioral LSTM training
4. Configure parameters and click "Start Training"

**Option B: From Command Line**
```bash
# Train MLP
python train_mlp.py --train archive/KDDTrain+_20Percent.txt --val archive/KDDTest+.txt

# Train LSTM
python train_lstm.py --train archive/KDDTrain+_20Percent.txt --val archive/KDDTest+.txt

# Build sequences for Behavioral LSTM
python build_beh_sequences.py --csv KDDTest_plus.csv

# Train Behavioral LSTM
python train_beh_lstm.py --x X_seq.npy --y y_seq.npy
```

For detailed training instructions, see `TRAINING_GUIDE.md`.

### 7. Troubleshooting

- **Missing required columns**:
  - The app will list any missing KDD columns. Make sure your CSV has all 41 required input features.

- **Model artifacts not found**:
  - Confirm that the `.h5` and `.pkl` files are in the folder specified by **Artifact directory**.

- **TensorFlow / GPU errors**:
  - `AttackDetector` can be forced to CPU by setting `force_cpu=True` when constructing it in your own scripts.
  - The provided Streamlit app uses the default behavior; TensorFlow will decide CPU/GPU based on your environment.

- **Training fails**:
  - Check that your data files exist and are in KDD format
  - Verify file paths are correct
  - Ensure you have write permissions in the artifact directory
  - Training does NOT require sudo (only live capture does)

- **Sudo/Privileges**:
  - Training: No sudo needed
  - Live capture: Requires sudo (see setup instructions above)
  - Package installation: Usually no sudo needed (use `--user` or virtual environments)


