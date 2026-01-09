"""
Complete workflow: Wireshark PCAP -> KDD CSV -> Sequences -> Train Behavioral LSTM -> Get Threshold

This script automates the entire process of training a behavioral LSTM model from Wireshark captures.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from typing import Optional

def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow: PCAP -> CSV -> Sequences -> Train Behavioral LSTM"
    )
    parser.add_argument(
        "--pcap",
        type=str,
        required=True,
        help="Path to Wireshark PCAP file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for all generated files (default: current directory)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Sequence length for behavioral LSTM (default: 10)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--skip_pcap",
        action="store_true",
        help="Skip PCAP conversion (assume CSV already exists)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Use existing CSV file instead of converting PCAP (requires --skip_pcap)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Wireshark to Behavioral LSTM Training Workflow")
    print("=" * 60)
    print()
    
    # Step 1: Convert PCAP to CSV
    if not args.skip_pcap:
        if not os.path.exists(args.pcap):
            print(f"❌ Error: PCAP file not found: {args.pcap}")
            return 1
        
        print(f"Step 1: Converting PCAP to KDD CSV...")
        print(f"  Input: {args.pcap}")
        
        csv_path = os.path.join(args.output_dir, f"wireshark_capture_{os.path.basename(args.pcap).replace('.pcap', '').replace('.pcapng', '')}.csv")
        
        try:
            from wireshark_to_kdd import pcap_to_kdd_csv
            csv_path = pcap_to_kdd_csv(
                pcap_path=args.pcap,
                output_csv=csv_path,
                time_window=2.0,
                min_packets_per_flow=1
            )
            print(f"  ✅ Output: {csv_path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return 1
    else:
        if args.csv:
            csv_path = args.csv
        else:
            print("❌ Error: --skip_pcap requires --csv to specify existing CSV file")
            return 1
        
        if not os.path.exists(csv_path):
            print(f"❌ Error: CSV file not found: {csv_path}")
            return 1
        
        print(f"Step 1: Using existing CSV (skipped)")
        print(f"  CSV: {csv_path}")
    
    # Step 2: Build sequences
    print()
    print(f"Step 2: Building sequences from CSV...")
    print(f"  Input: {csv_path}")
    print(f"  Sequence length: {args.seq_len}")
    
    x_seq_path = os.path.join(args.output_dir, "X_seq.npy")
    y_seq_path = os.path.join(args.output_dir, "y_seq.npy")
    
    try:
        cmd = [
            sys.executable, "build_beh_sequences.py",
            "--csv", csv_path,
            "--out_x", x_seq_path,
            "--out_y", y_seq_path,
            "--seq_len", str(args.seq_len)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"  ❌ Error: {result.stderr}")
            return 1
        
        print(f"  ✅ Output: {x_seq_path}, {y_seq_path}")
        print(f"  {result.stdout}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    # Step 3: Train behavioral LSTM
    print()
    print(f"Step 3: Training behavioral LSTM...")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    model_path = os.path.join(args.output_dir, "beh_lstm.h5")
    
    try:
        cmd = [
            sys.executable, "train_beh_lstm.py",
            "--x", x_seq_path,
            "--y", y_seq_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--out", model_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"  ❌ Error: {result.stderr}")
            return 1
        
        print(f"  ✅ Model saved: {model_path}")
        print(f"  {result.stdout}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    # Step 4: Check for threshold file
    print()
    print(f"Step 4: Checking threshold file...")
    th_path = os.path.join(args.output_dir, "beh_lstm.threshold.json")
    
    if os.path.exists(th_path):
        import json
        with open(th_path, "r") as f:
            th_data = json.load(f)
        print(f"  ✅ Threshold file: {th_path}")
        print(f"  Optimal threshold: {th_data.get('threshold', 'N/A')}")
        print(f"  F1-Score: {th_data.get('f1', 'N/A'):.4f}")
        print(f"  Precision: {th_data.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {th_data.get('recall', 'N/A'):.4f}")
    else:
        print(f"  ⚠️  Threshold file not found: {th_path}")
    
    print()
    print("=" * 60)
    print("✅ Workflow completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Model is ready: {model_path}")
    print(f"  2. Use in Streamlit: Click 'Load / Reload Detector' in sidebar")
    print(f"  3. The behavioral LSTM will be automatically loaded for live detection")
    
    return 0


if __name__ == "__main__":
    exit(main())

