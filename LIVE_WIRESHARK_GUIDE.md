# Live Wireshark Capture Guide

## Overview

You can now display **live network traffic from Wireshark (tshark) directly in the Streamlit frontend**! This provides real-time packet capture and analysis without needing special permissions that Scapy requires.

## How It Works

The system uses **tshark** (Wireshark's command-line tool) to capture live network traffic and convert it to KDD features in real-time, displaying them in the Streamlit dashboard.

## Setup

### 1. Install Wireshark/tshark

**macOS:**
```bash
brew install wireshark
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tshark
```

**Linux (RHEL/CentOS):**
```bash
sudo yum install wireshark
```

**Windows:**
- Download from https://www.wireshark.org/download.html
- Make sure `tshark.exe` is in your PATH

### 2. Verify Installation

```bash
tshark --version
```

You should see version information. If not, make sure Wireshark is installed and `tshark` is in your PATH.

## Usage in Streamlit

### Step 1: Open Live Capture Tab

1. Run Streamlit: `streamlit run streamlit_app.py`
2. Go to **"🛰️ Live Capture"** tab

### Step 2: Select Capture Method

You'll see a radio button to choose:
- **Wireshark (tshark)** - Recommended (no special permissions needed)
- **Scapy (Python)** - Alternative (requires root/admin)

Select **"Wireshark (tshark)"**.

### Step 3: Configure Capture

- **Interface**: Select from dropdown (auto-detected) or enter manually (e.g., `en0`, `eth0`, `wlan0`)
- **BPF Filter**: Optional filter (e.g., `tcp`, `port 80`, `host 192.168.1.1`)
- **Refresh**: How often to update display (seconds)

### Step 4: Start Capture

1. Click **"▶️ Start Sniffing"**
2. Traffic will start appearing in real-time
3. Click **"🔄 Refresh now"** to update the display

### Step 5: View Results

The dashboard will show:
- **Total Flows**: Number of captured flows
- **Revoked**: Flows flagged as attacks
- **Adaptive Threshold**: Current risk threshold
- **Avg Flow Risk**: Average risk score
- **Live Flow Data**: Table of recent flows with predictions

## Features

### Real-Time Display

- Packets are captured and converted to KDD features in real-time
- Flows are automatically grouped and analyzed
- Predictions are made using your trained models (MLP, LSTM, Behavioral LSTM)

### Behavioral Analysis

If you have a behavioral LSTM model (`beh_lstm.h5`):
- Per-user sequences are built automatically
- Behavioral risk scores are computed
- Combined with flow risk for better detection

### Continuous Learning

You can enable continuous learning to:
- Collect live data for a specified duration
- Automatically train behavioral LSTM on collected data
- Cycle continuously for adaptive learning

## Advantages Over Scapy

✅ **No special permissions needed** - tshark handles capture  
✅ **More reliable** - Works on all platforms  
✅ **Better performance** - Optimized C implementation  
✅ **Wider protocol support** - Handles more protocols  
✅ **Better filtering** - Advanced BPF filter support  

## Troubleshooting

### "tshark not available"

**Solution**: Install Wireshark (see Setup section above)

### "No interfaces detected"

**Solution**: 
- Enter interface name manually (e.g., `en0` on macOS, `eth0` on Linux)
- Check available interfaces: `tshark -D`

### "Permission denied"

**Solution**: 
- On Linux, you may need to add your user to `wireshark` group:
  ```bash
  sudo usermod -aG wireshark $USER
  # Then logout and login again
  ```
- On macOS, grant Terminal network permissions in System Preferences

### "No traffic appearing"

**Solution**:
- Check that interface is correct
- Verify network is active
- Try removing BPF filter
- Check that capture is actually running (status indicator)

### "High CPU usage"

**Solution**:
- Use BPF filters to reduce traffic
- Increase refresh interval
- Limit capture to specific interface

## Comparison: Wireshark vs Scapy

| Feature | Wireshark (tshark) | Scapy |
|---------|-------------------|-------|
| Permissions | Usually not needed | Root/admin required |
| Performance | Fast (C implementation) | Slower (Python) |
| Reliability | High | Medium |
| Protocol Support | Extensive | Good |
| Platform Support | All platforms | All platforms |
| Filtering | Advanced BPF | Basic |

## Best Practices

1. **Use BPF Filters**: Filter traffic to reduce load
   - `tcp` - Only TCP traffic
   - `port 80` - Only HTTP traffic
   - `host 192.168.1.1` - Specific host

2. **Selective Interface**: Capture on active interfaces only
   - Avoid capturing on loopback unless needed
   - Use specific interface instead of "any"

3. **Monitor Performance**: 
   - Watch CPU usage
   - Adjust refresh interval
   - Use filters if too much traffic

4. **Combine with Behavioral LSTM**:
   - Train behavioral model on your network
   - Enable for better detection accuracy
   - Use continuous learning for adaptation

## Integration with Existing Features

The live Wireshark capture integrates seamlessly with:

- ✅ **Batch CSV Analysis** - Still works as before
- ✅ **Model Training** - Train on captured data
- ✅ **Behavioral LSTM** - Automatic per-user sequence building
- ✅ **Continuous Learning** - Collect and retrain automatically
- ✅ **Risk Scoring** - Real-time attack detection

## Example Workflow

1. **Start Streamlit**: `streamlit run streamlit_app.py`
2. **Select Wireshark capture method**
3. **Choose interface** (e.g., `en0`)
4. **Add filter** (optional, e.g., `tcp port 80`)
5. **Start capture**
6. **View live traffic** with predictions
7. **Enable continuous learning** (optional)
8. **Train behavioral LSTM** on collected data
9. **Use trained model** for better detection

## Next Steps

- See `WIRESHARK_GUIDE.md` for processing saved PCAP files
- See `CONTINUOUS_LEARNING.md` for automatic training
- See `TRAINING_GUIDE.md` for manual training

The live Wireshark feature makes real-time network monitoring accessible without special permissions!

