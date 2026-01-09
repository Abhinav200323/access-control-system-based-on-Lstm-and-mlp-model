# Sudo Command for Scapy on macOS

## Quick Command

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
source venv/bin/activate
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

## Step-by-Step

### 1. Navigate to Project Directory
```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Run with Sudo
```bash
sudo env PATH="$PATH" streamlit run streamlit_app.py
```

**You'll be prompted for your admin password.**

## One-Line Command (All Steps Combined)

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major" && source venv/bin/activate && sudo env PATH="$PATH" streamlit run streamlit_app.py
```

## Why `env PATH="$PATH"`?

- Preserves your PATH environment variable
- Ensures Python and packages from venv are found
- Required for sudo to access virtual environment packages

## Alternative: Use the Script

Instead of typing the command manually, you can use:

```bash
cd "/Users/abhin/Desktop/ai cyber/Major/Major"
./run_scapy.sh
```

## After Running

1. The app opens at `http://localhost:8501`
2. Go to **"🛰️ Live Capture"** tab
3. Select **"Scapy (Python)"** as capture method
4. Choose interface (e.g., `en0`)
5. Click **"▶️ Start Sniffing"**

## Troubleshooting

### "Command not found: streamlit"
Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### "Permission denied"
You must use `sudo` for Scapy. The command above includes sudo.

### "Module not found: scapy"
Install Scapy:
```bash
source venv/bin/activate
pip install scapy
```

## Quick Reference

```bash
# Full command
cd "/Users/abhin/Desktop/ai cyber/Major/Major" && source venv/bin/activate && sudo env PATH="$PATH" streamlit run streamlit_app.py

# Or use script
./run_scapy.sh
```
