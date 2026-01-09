"""
Build per-user behavioural sequences from a KDD-style CSV for training beh_lstm.

This script is a helper to create:
  - X_seq.npy: (N_seq, T, D) sequence features
  - y_seq.npy: (N_seq,) 0/1 labels (0 = normal, 1 = attack)

Assumptions (you can edit these):
  - Input CSV has the 41 KDD features with the usual column names.
  - Input CSV has a 'label' column with values like 'normal', 'neptune', etc.
  - We simulate a "user_id" by grouping rows into fixed-size chunks, because
    NSL-KDD does not contain real per-user identifiers. For live Scapy data,
    the behavioural LSTM will instead use real user_id = src_ip.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


KDD_FEATURE_COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Categorical columns that must be encoded before converting to float
CATEGORICAL_COLUMNS: List[str] = ["protocol_type", "service", "flag"]


def build_sequences(
    csv_path: str,
    out_x: str = "X_seq.npy",
    out_y: str = "y_seq.npy",
    seq_len: int = 20,
    chunk_size: int = 1000,
) -> None:
    if not os.path.exists(csv_path):
        raise SystemExit(f"Input CSV {csv_path} not found")

    df = pd.read_csv(csv_path)

    missing = [c for c in KDD_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required KDD columns: {missing}")
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column.")

    # Simple binary label per row: 0 normal, 1 attack
    is_attack = (df["label"].astype(str).str.lower() != "normal").to_numpy(dtype=int)

    # Encode categorical KDD columns as integers so we can form numeric sequences.
    df_enc = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df_enc.columns:
            codes, _ = pd.factorize(df_enc[col].astype(str))
            df_enc[col] = codes.astype(float)

    feats = df_enc[KDD_FEATURE_COLUMNS].to_numpy(dtype=float)

    X_seq = []
    y_seq = []

    # Simulate user sequences by slicing the dataset into chunks of rows
    # In real deployment, live mode will use true per-user (e.g., src_ip) sequences.
    n = feats.shape[0]
    for start_idx in range(0, n, chunk_size):
        end_idx = min(n, start_idx + chunk_size)
        chunk_feats = feats[start_idx:end_idx]
        chunk_labels = is_attack[start_idx:end_idx]
        if chunk_feats.shape[0] < seq_len:
            continue

        # Sequence label: 1 if any row in the sequence is attack, else 0
        for i in range(0, chunk_feats.shape[0] - seq_len + 1, seq_len):
            window = chunk_feats[i : i + seq_len]
            window_labels = chunk_labels[i : i + seq_len]
            if window.shape[0] != seq_len:
                continue
            seq_label = int(np.any(window_labels == 1))
            X_seq.append(window)
            y_seq.append(seq_label)

    if not X_seq:
        raise SystemExit("No sequences were built; try smaller seq_len or chunk_size.")

    X_arr = np.stack(X_seq, axis=0)
    y_arr = np.asarray(y_seq, dtype=int)

    np.save(out_x, X_arr)
    np.save(out_y, y_arr)
    print(f"Saved X_seq to {out_x} with shape {X_arr.shape}")
    print(f"Saved y_seq to {out_y} with shape {y_arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input KDD-style CSV file")
    parser.add_argument("--out_x", type=str, default="X_seq.npy")
    parser.add_argument("--out_y", type=str, default="y_seq.npy")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    build_sequences(
        csv_path=args.csv,
        out_x=args.out_x,
        out_y=args.out_y,
        seq_len=args.seq_len,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()


