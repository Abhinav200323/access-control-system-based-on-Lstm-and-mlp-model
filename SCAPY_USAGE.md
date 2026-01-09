# Using Scapy from Terminal on macOS

This guide explains how to run the Attack Detection System with Scapy for live network packet capture.

## Quick Start

```bash
# Navigate to project directory
cd "/Users/abhin/Desktop/ai cyber/Major/Major"

# Run with Scapy support (requires sudo)
./run_scapy.sh
```

You'll be prompted for your admin password. The app will open at `http://localhost:8501`

## Why Use Scapy?

Scapy is a Python library for packet manipulation and network capture. It's useful when:
- You prefer Python-based packet capture
- Wireshark is not installed
- You want more control over packet processing

**Note:** Scapy requires sudo/root privileges on macOS to access network interfaces.

## Setup Instructions

### Step 1: Ensure Scapy is Installed

```bash
# Activate virtual environment
source venv/bin/activate

# Check if Scapy is installed
python3 -c "import scapy; print('Scapy installed')"

# If not installed, install it
pip install scapy
```

### Step 2: Run with Sudo

**Option A: Using the Run Script (Recommended)**
```bash
./run_scapy.sh
```

**Option B: Manual Command**
```bash
# Activate virtual environment
source venv/bin/activate

# Run with sudo
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**Why `env PATH="$PATH"`?**
- Preserves your PATH environment variable
- Ensures Python and packages from venv are found
- Required for sudo to work correctly

### Step 3: Use Scapy in the Application

1. Open the app in your browser (usually `http://localhost:8501`)
2. Go to **"🛰️ Live Capture"** tab
3. Select **"Scapy (Python)"** as the capture method
4. Choose your network interface:
   - `en0` - Primary Ethernet/WiFi (most common)
   - `en1` - Secondary interface
   - Leave empty for default interface
5. (Optional) Add BPF filter: e.g., `tcp`, `port 80`, `host 192.168.1.1`
6. Click **"▶️ Start Sniffing"**
7. Click **"🔄 Refresh now"** to see captured traffic

## Network Interfaces on macOS

To find available interfaces:

```bash
# List all network interfaces
ifconfig | grep "^[a-z]"

# Common interfaces:
# en0 - Primary Ethernet/WiFi
# en1 - Secondary interface  
# lo0 - Loopback (localhost)
# bridge0 - Bridge interface
```

## Troubleshooting

### Issue: "Permission denied: could not open /dev/bpf0"

**Solution:** You need to run with sudo:
```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

Or use the run script:
```bash
./run_scapy.sh
```

### Issue: "Command not found: streamlit"

**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### Issue: "Module not found: scapy"

**Solution:** Install Scapy:
```bash
source venv/bin/activate
pip install scapy
```

### Issue: Sudo asks for password every time

**Solution:** This is normal macOS security behavior. Alternatively:
1. Grant network permissions in System Settings (see below)
2. Use Wireshark method instead (no sudo needed)

## Alternative: Grant Network Permissions (No Sudo)

If you prefer not to use sudo, you can grant network permissions:

1. Open **System Settings** → **Privacy & Security** → **Network**
2. Click the lock icon and enter your password
3. Check the box next to **Terminal** (or your terminal app)
4. Run normally: `streamlit run streamlit_app.py`
5. Use **Wireshark (tshark)** method in the app (no sudo needed)

## Comparison: Scapy vs Wireshark

| Feature | Scapy | Wireshark (tshark) |
|---------|-------|-------------------|
| Requires sudo | ✅ Yes | ❌ No |
| Python-based | ✅ Yes | ❌ No |
| Installation | `pip install scapy` | `brew install wireshark` |
| Performance | Good | Excellent |
| Filtering | Python code | BPF filters |
| Recommended for | Development | Production |

## Example BPF Filters

When using Scapy, you can use BPF (Berkeley Packet Filter) syntax:

```bash
# Capture only TCP traffic
tcp

# Capture traffic on port 80
port 80

# Capture traffic to/from specific host
host 192.168.1.1

# Capture HTTP traffic
tcp port 80

# Capture DNS traffic
port 53

# Combine filters
tcp and port 443
```

## Security Note

⚠️ **Important:** Only capture traffic on networks you own or have permission to monitor. Unauthorized network monitoring may be illegal.

## Quick Reference

```bash
# Run with Scapy
./run_scapy.sh

# Or manually
source venv/bin/activate
sudo env PATH="$PATH" streamlit run streamlit_app.py

# Check Scapy installation
python3 -c "import scapy; print(scapy.__version__)"

# List network interfaces
ifconfig | grep "^[a-z]"
```

## Next Steps

1. ✅ Run `./run_scapy.sh`
2. ✅ Open Live Capture tab
3. ✅ Select "Scapy (Python)" method
4. ✅ Choose interface and start capturing
5. ✅ View real-time attack detection

---

For more information, see:
- `STEPS_TO_RUN_MAC.md` - Complete setup guide
- `macOS_SETUP.md` - macOS-specific instructions
- `README.md` - Full documentation
