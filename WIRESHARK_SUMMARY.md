# Wireshark Integration - Quick Summary

## Problem Solved

The live Scapy capture feature may not work due to permission issues. **Solution**: Use Wireshark to capture traffic, then process it through our system.

## Quick Start (3 Methods)

### Method 1: Streamlit UI (Easiest) ⭐

1. Capture in Wireshark → Save as `.pcap`
2. Open Streamlit → **Training** tab → **Behavioral LSTM Training**
3. Upload PCAP file → Click "Convert PCAP to KDD CSV"
4. Click "Run Complete Workflow" → Done!

### Method 2: Command Line (One Command)

```bash
python wireshark_workflow.py --pcap your_capture.pcap --seq_len 10 --epochs 20
```

### Method 3: Step-by-Step Commands

```bash
# Step 1: Convert PCAP to CSV
python wireshark_to_kdd.py --pcap capture.pcap --output data.csv

# Step 2: Build sequences
python build_beh_sequences.py --csv data.csv --seq_len 10

# Step 3: Train model
python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --epochs 20 --out beh_lstm.h5
```

## Output Files

- `beh_lstm.h5` - Trained model
- `beh_lstm.threshold.json` - Optimal threshold and metrics
- CSV file with KDD features (intermediate)

## Using the Trained Model

1. Model is automatically loaded in Streamlit
2. Click "Load / Reload Detector" in sidebar
3. Use in Batch CSV or Live Capture tabs

## Advantages

✅ No special permissions needed  
✅ Works on all platforms  
✅ Better control over capture  
✅ Can label data manually  
✅ Reproducible results  

## Files Created

- `wireshark_to_kdd.py` - PCAP to CSV converter
- `wireshark_workflow.py` - Complete automation script
- `WIRESHARK_GUIDE.md` - Detailed documentation

## Frontend

The Streamlit frontend remains **exactly the same**. The Wireshark processing is just an alternative data source for training. All existing features (batch CSV, live capture when working, training UI) continue to work as before.

