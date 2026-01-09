# AI Powered Access Control System using LSTM / MLP

## Abstract
This project builds an AI-powered access control and intrusion detection system that analyzes user/network behavior using neural models. We use the first 41 NSL-KDD features, preprocess via z-score normalization (numeric) and one-hot encoding (categorical), and infer with a neural ensemble: an MLP (required) and an optional LSTM. The system supports batch CSV inference and real-time live capture (Scapy). A revoke decision is triggered when the predicted class is risky and model confidence exceeds a configurable threshold.

## Methodology
- Data schema: 41 KDD/NSL-KDD features (38 numeric + 3 categorical).
- Preprocessing: z-score on numeric; one-hot for `protocol_type`, `service`, `flag` using `onehot_encoder.pkl` (fallback: runtime one-hot). Sparse numeric+categorical features are concatenated.
- Models: `att_det_mlp.h5` (required) and `att_det_lstm.h5` (optional). Each outputs logits → softmax probabilities. Probabilities are averaged; predicted label is argmax of the mean.
- Labels: If `label_binarizer.pkl` is present, numeric predictions map back to class names (e.g., normal, attack types).
- Revoke logic: In `predict_with_revoke`, revoke=1 if the predicted label is risky and max probability ≥ threshold (default 0.90). Risky labels default to all except 'normal' when label names are available.

## Implementation
- `attack_detection_pipeline.py`:
  - Loads artifacts; preprocesses inputs; performs neural inference; provides `predict` and `predict_with_revoke(threshold)` APIs.
- `lstm_model.py`:
  - Defines `build_lstm_model` and `train_and_save_lstm` to produce `att_det_lstm.h5` compatible with the pipeline.
- `train_lstm.py`:
  - Trains an LSTM using NSL-KDD files and the same preprocessing; saves `att_det_lstm.h5` to the project root.
- `streamlit_app.py`:
  - Sidebar: artifact directory, revoke threshold.
  - Tabs: Batch CSV upload and Live Capture (Scapy). Displays predictions and revoke flags; CSV export includes `revoke`.
- `scapy_featurizer.py`:
  - Captures packets (Npcap+admin on Windows), constructs KDD-like features for live inference.

## Setup
```powershell
cd "C:\Users\abhin\Desktop\ai cyber\Major\Major"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_streamlit.txt
pip install tensorflow scapy
```
Artifacts expected in project root:
- `att_det_mlp.h5` (required), `att_det_lstm.h5` (optional)
- `onehot_encoder.pkl`, `label_binarizer.pkl`

Run app:
```powershell
streamlit run streamlit_app.py
```

Train LSTM:
```powershell
python train_lstm.py --train .\archive\KDDTrain+_20Percent.txt --val .\archive\KDDTest+.txt --artifact_dir . --epochs 8 --batch_size 256 --output att_det_lstm.h5
```

## Evaluation (starter)
- Dataset: NSL-KDD (`KDDTrain+_20Percent`, `KDDTest+`).
- Metrics to record: accuracy, precision, recall, F1; ROC-AUC per class if applicable.
- Procedure: Evaluate MLP-only, LSTM-only (if trained), and ensemble. Compare confusion matrices and false positives.

## Limitations
- LSTM currently consumes a single-step pseudo-sequence; richer temporal modeling needs true sequences (multiple events per user/flow) and pipeline changes.
- Live features are approximations; some KDD counters are simplified.
- No OAuth2/JWT integration yet; revoke is computed but not connected to an IAM/SSO system.

## Next Steps
- Sequence construction per user/session to fully leverage LSTM temporal modeling.
- Integrate with an access gateway (e.g., API gateway + IAM) to enforce revoke decisions (token invalidation, session kill, MFA challenge).
- Add calibration (temperature scaling) to improve probability thresholds.
- Evaluate on modern datasets (CIC-IDS2017, UNSW-NB15) and add domain-specific features.
- Containerize (Docker) and add CI/CD for retraining and deployment.

## References
See the provided synopsis for the literature list. Add empirical results and links to trained model cards as experiments complete.
