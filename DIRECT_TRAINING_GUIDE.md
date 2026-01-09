# Direct Training from Live Capture Guide

## Overview

You can now **train behavioral LSTM models directly from live Wireshark/Scapy capture** without needing to save PCAP files first! This feature collects live traffic, builds sequences, and trains the model automatically.

## How It Works

1. **Start Live Capture** (Wireshark or Scapy)
2. **Enable Direct Training** in the Live Capture tab
3. **Collect Data** for a specified duration (e.g., 10 minutes)
4. **Automatic Training** - sequences are built and model is trained
5. **Model Ready** - automatically reloaded and ready to use

## Usage Steps

### Step 1: Start Live Capture

1. Go to **"🛰️ Live Capture"** tab in Streamlit
2. Select capture method:
   - **Wireshark (tshark)** - Recommended (no special permissions)
   - **Scapy (Python)** - Alternative
3. Configure interface and optional BPF filter
4. Click **"▶️ Start Sniffing"**

### Step 2: Enable Direct Training

1. Scroll down to **"🎓 Train Behavioral LSTM from Live Capture"** section
2. Check **"Enable Direct Training from Live Capture"**
3. Configure settings:
   - **Collection Duration**: How long to collect data (1-60 minutes)
   - **Sequence Length**: Timesteps per sequence (5-50, default: 10)
   - **Training Epochs**: Number of training epochs (5-50, default: 20)

### Step 3: Start Data Collection

1. Click **"🚀 Start Data Collection for Training"**
2. Monitor progress:
   - Elapsed time
   - Remaining time
   - Number of flows collected

### Step 4: Train Model

When collection period completes:
1. Click **"🎓 Train Model Now"** (or it auto-trains)
2. System will:
   - Build sequences from collected data
   - Train behavioral LSTM model
   - Compute optimal threshold
   - Save model as `beh_lstm.h5`
   - Reload detector automatically

### Step 5: Use the Model

The trained model is immediately available:
- Automatically loaded in the detector
- Works with live capture for behavioral analysis
- Per-user risk scores computed
- Combined with flow risk for better detection

## Features

### Real-Time Data Collection

- Collects live network flows as they're captured
- Automatically extracts KDD features
- Groups flows by user (src_ip)
- Builds sequences in real-time

### Automatic Sequence Building

- Creates per-user sequences from collected flows
- Sliding windows with 50% overlap
- Binary labels (attack vs normal)
- Handles variable-length user histories

### One-Click Training

- Single button to train model
- Automatic hyperparameter handling
- Threshold computation included
- Model validation and metrics

### Seamless Integration

- Model automatically reloaded
- Works immediately with live capture
- No manual file management
- Continuous learning support

## Configuration Options

### Collection Duration

- **Short (5-10 min)**: Quick training, less data
- **Medium (10-20 min)**: Balanced (recommended)
- **Long (20-60 min)**: More data, better model, longer wait

### Sequence Length

- **Short (5-7)**: Faster adaptation, less context
- **Medium (10-15)**: Balanced (recommended)
- **Long (20-50)**: More context, requires more data

### Training Epochs

- **Few (5-10)**: Quick training, may underfit
- **Medium (15-20)**: Balanced (recommended)
- **Many (30-50)**: Better fit, longer training time

## Requirements

- **Live capture running**: Must start capture before training
- **Sufficient data**: Need at least 50+ flows, preferably 100+
- **Multiple users**: Need flows from different source IPs
- **Minimum flows per user**: At least 5 flows per user for sequences

## Workflow Example

```
1. Start Streamlit: streamlit run streamlit_app.py
2. Go to Live Capture tab
3. Select "Wireshark (tshark)" method
4. Choose interface (e.g., en0)
5. Click "Start Sniffing"
6. Enable "Direct Training from Live Capture"
7. Set duration: 10 minutes
8. Click "Start Data Collection for Training"
9. Wait for collection to complete
10. Click "Train Model Now"
11. Model is ready! Use in live detection
```

## Advantages

✅ **No file management** - Everything happens in memory  
✅ **Real-time adaptation** - Train on current network patterns  
✅ **Automatic workflow** - No manual steps  
✅ **Immediate use** - Model ready right after training  
✅ **Continuous learning** - Can retrain periodically  

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Direct Training (Live)** | Real-time, automatic, no files | Requires active capture |
| **Wireshark PCAP** | Can label data, reproducible | Manual steps, file management |
| **Continuous Learning** | Fully automatic cycles | More complex setup |

## Tips

1. **Active Network**: Ensure network has active traffic during collection
2. **Multiple Users**: Better with traffic from multiple source IPs
3. **Balanced Data**: Mix of normal and attack traffic improves model
4. **Regular Retraining**: Retrain periodically as network patterns change
5. **Monitor Performance**: Check threshold and metrics after training

## Troubleshooting

### "Not enough data collected"

**Solution**: 
- Increase collection duration
- Ensure network has active traffic
- Check that capture is running

### "No sequences built"

**Solution**:
- Need at least 5 flows per user
- Reduce sequence length
- Collect more data

### "Training fails"

**Solution**:
- Check TensorFlow installation
- Ensure sufficient memory
- Reduce batch size or epochs

### "Model not reloading"

**Solution**:
- Manually click "Load / Reload Detector" in sidebar
- Check that `beh_lstm.h5` was created
- Verify file permissions

## Integration

Direct training works seamlessly with:
- ✅ Live Wireshark capture
- ✅ Live Scapy capture
- ✅ Behavioral LSTM inference
- ✅ Continuous learning cycles
- ✅ Batch CSV analysis

## Next Steps

After training:
1. **Test the model**: Use in live capture to see behavioral risk scores
2. **Monitor performance**: Check detection accuracy
3. **Retrain periodically**: Update model with new traffic patterns
4. **Combine with flow models**: Use ensemble for better detection

For more information:
- `LIVE_WIRESHARK_GUIDE.md` - Live capture setup
- `CONTINUOUS_LEARNING.md` - Automatic retraining cycles
- `TRAINING_GUIDE.md` - General training instructions

