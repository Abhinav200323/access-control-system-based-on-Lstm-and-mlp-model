# Wireshark Integration Guide

This guide explains how to use Wireshark to capture network traffic and train behavioral LSTM models when the live Scapy capture feature is not working.

## Overview

Instead of using the live Scapy capture (which may require special permissions), you can:

1. **Capture traffic in Wireshark** (works on all platforms, no special permissions needed)
2. **Export as PCAP file**
3. **Process PCAP to KDD features** using our converter
4. **Train behavioral LSTM** on the captured data
5. **Get threshold values** automatically

## Step-by-Step Workflow

### Method 1: Using Streamlit UI (Easiest)

1. **Capture in Wireshark**:
   - Open Wireshark
   - Select your network interface
   - Click "Start capturing packets"
   - Let it run for desired duration (e.g., 10-30 minutes)
   - Click "Stop capturing packets"
   - Go to **File → Export Specified Packets → Save as PCAP**

2. **Process in Streamlit**:
   - Open Streamlit app: `streamlit run streamlit_app.py`
   - Go to **Training** tab → **Behavioral LSTM Training** sub-tab
   - Expand **"📡 Process Wireshark PCAP File"** section
   - Click **"Choose File"** and select your `.pcap` file
   - Click **"🔄 Convert PCAP to KDD CSV"**
   - Wait for conversion to complete

3. **Build Sequences and Train**:
   - The CSV will be automatically selected
   - Click **"🔨 Build Sequences"** (or use the complete workflow button)
   - Set sequence length (default: 10)
   - Click **"🚀 Start Behavioral LSTM Training"**
   - Or use **"🎓 Run Complete Workflow"** to do everything at once

4. **Get Results**:
   - Model saved as `beh_lstm.h5`
   - Threshold file saved as `beh_lstm.threshold.json`
   - Click **"Load / Reload Detector"** in sidebar to use the new model

### Method 2: Using Command Line

#### Step 1: Convert PCAP to CSV

```bash
python wireshark_to_kdd.py --pcap your_capture.pcap --output wireshark_data.csv
```

**Options**:
- `--pcap`: Path to your Wireshark PCAP file
- `--output`: Output CSV filename (default: `wireshark_capture.csv`)
- `--time_window`: Time window for flow grouping in seconds (default: 2.0)
- `--min_packets`: Minimum packets per flow (default: 1)

#### Step 2: Build Sequences

```bash
python build_beh_sequences.py --csv wireshark_data.csv --seq_len 10
```

This creates:
- `X_seq.npy`: Sequence features (N_seq, T, D)
- `y_seq.npy`: Sequence labels (N_seq,)

#### Step 3: Train Behavioral LSTM

```bash
python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --epochs 20 --out beh_lstm.h5
```

This creates:
- `beh_lstm.h5`: Trained model
- `beh_lstm.threshold.json`: Optimal threshold and metrics

#### Step 4: Use in Streamlit

1. Open Streamlit app
2. Click **"Load / Reload Detector"** in sidebar
3. The behavioral LSTM will be automatically loaded

### Method 3: Complete Workflow Script (One Command)

```bash
python wireshark_workflow.py --pcap your_capture.pcap --seq_len 10 --epochs 20
```

This single command does everything:
1. Converts PCAP to CSV
2. Builds sequences
3. Trains behavioral LSTM
4. Computes threshold
5. Shows results

**Options**:
- `--pcap`: Path to PCAP file
- `--output_dir`: Output directory (default: current directory)
- `--seq_len`: Sequence length (default: 10)
- `--epochs`: Training epochs (default: 20)
- `--batch_size`: Batch size (default: 64)
- `--skip_pcap`: Skip PCAP conversion (use existing CSV)
- `--csv`: Use existing CSV file

## Wireshark Capture Tips

### Best Practices

1. **Capture Duration**:
   - Minimum: 5-10 minutes for meaningful sequences
   - Recommended: 15-30 minutes for better model training
   - Maximum: Several hours (but processing time increases)

2. **Filter Traffic** (Optional):
   - Use Wireshark filters to focus on specific traffic:
     - `tcp` - Only TCP traffic
     - `tcp.port == 80` - Only HTTP traffic
     - `ip.addr == 192.168.1.1` - Traffic to/from specific IP
   - Too much filtering may reduce data diversity

3. **Network Selection**:
   - Capture on active network interfaces
   - Monitor mode (if available) captures all traffic
   - Promiscuous mode captures traffic not destined for your machine

4. **File Size**:
   - Large PCAP files (>100MB) may take longer to process
   - Consider splitting very large captures
   - Use Wireshark's "Export Specified Packets" to extract subsets

### Labeling Captured Data (Optional)

The converted CSV will have a `label` column set to `'normal'` by default. If you know which flows are attacks:

1. Open the CSV file in Excel or a text editor
2. Update the `label` column:
   - `'normal'` for legitimate traffic
   - `'attack'` or specific attack type for malicious traffic
3. Save and use for training

## Understanding the Output

### CSV File Structure

The converted CSV contains:
- **41 KDD features**: All standard KDD99 features
- **Metadata columns**: `src_ip`, `dst_ip`, `src_port`, `dst_port`, `timestamp`
- **Label column**: `label` (default: 'normal')
- **User ID column**: `user_id` (from `src_ip`)

### Sequence Files

- **X_seq.npy**: 3D array (N_sequences, timesteps, features)
- **y_seq.npy**: 1D array (N_sequences,) with binary labels

### Model Output

- **beh_lstm.h5**: Keras/TensorFlow model file
- **beh_lstm.threshold.json**: Contains:
  ```json
  {
    "model": "beh_lstm.h5",
    "threshold": 0.65,
    "metric": "F1",
    "f1": 0.92,
    "precision": 0.89,
    "recall": 0.95
  }
  ```

## Troubleshooting

### "No valid flows extracted"

**Cause**: PCAP file has no IP packets or all packets are filtered out

**Solution**:
- Check that PCAP file contains network traffic
- Verify IP packets are present (not just link-layer frames)
- Try capturing on a different interface

### "No sequences built"

**Cause**: Not enough flows per user or sequence length too long

**Solution**:
- Reduce sequence length (try 5-7 instead of 10)
- Capture more traffic (longer duration)
- Check that `user_id` or `src_ip` column exists

### "Training fails"

**Cause**: Insufficient data or TensorFlow issues

**Solution**:
- Ensure at least 100+ sequences are built
- Check TensorFlow installation
- Reduce batch size if memory errors occur

### "PCAP file too large"

**Cause**: Very large PCAP files take long to process

**Solution**:
- Use Wireshark to export a subset of packets
- Process in chunks and combine CSVs
- Use filters to reduce capture size

## Integration with Existing System

The trained behavioral LSTM integrates seamlessly:

1. **Automatic Loading**: Streamlit automatically loads `beh_lstm.h5` if present
2. **Threshold Usage**: Threshold from JSON is used for risk scoring
3. **Live Detection**: Works with live capture (when available) or batch CSV analysis
4. **Continuous Learning**: Can be retrained periodically with new Wireshark captures

## Advantages Over Live Capture

- ✅ **No special permissions needed** (Wireshark handles capture)
- ✅ **Works on all platforms** (Windows, macOS, Linux)
- ✅ **Better control** over capture duration and filtering
- ✅ **Can label data** manually if needed
- ✅ **Reproducible** (same PCAP = same results)
- ✅ **Offline processing** (no need to keep capture running)

## Next Steps

After training:

1. **Test the model**: Use batch CSV upload in Streamlit
2. **Monitor performance**: Check threshold and metrics
3. **Retrain periodically**: Capture new traffic and retrain
4. **Combine with flow models**: Behavioral LSTM works alongside MLP/LSTM

For more details, see:
- `TRAINING_GUIDE.md` - General training instructions
- `CONTINUOUS_LEARNING.md` - Continuous learning from live data
- `README.md` - Overall system documentation

