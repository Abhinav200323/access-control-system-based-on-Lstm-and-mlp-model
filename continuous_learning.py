"""
Continuous learning module for behavioral LSTM.
Collects live data, builds sequences, trains model, and cycles.
"""

from __future__ import annotations

import os
import time
import subprocess
import sys
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

import numpy as np
import pandas as pd


def build_sequences_from_live_data(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    feature_cols: Optional[List[str]] = None,
    label_col: str = "prediction",
    seq_len: int = 10,
    min_flows_per_user: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences from live captured data.
    
    Args:
        df: DataFrame with live captured flows
        user_id_col: Column name for user identifier (e.g., 'user_id' or 'src_ip')
        feature_cols: List of feature column names (41 KDD features)
        label_col: Column name for labels ('prediction' or 'revoke')
        seq_len: Length of sequences (timesteps)
        min_flows_per_user: Minimum flows required per user to create sequences
    
    Returns:
        X_seq: (N_seq, T, D) array of sequences
        y_seq: (N_seq,) array of binary labels (0=normal, 1=attack)
    """
    if feature_cols is None:
        # Default KDD feature columns
        feature_cols = [
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
    
    # Check required columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if user_id_col not in df.columns:
        # Try to use src_ip as user_id
        if "src_ip" in df.columns:
            df = df.copy()
            df[user_id_col] = df["src_ip"].astype(str)
        else:
            raise ValueError(f"User ID column '{user_id_col}' not found and 'src_ip' not available")
    
    if label_col not in df.columns:
        # Create binary labels from prediction or revoke
        if "revoke" in df.columns:
            df = df.copy()
            df[label_col] = df["revoke"].astype(int)
        elif "prediction" in df.columns:
            df = df.copy()
            # Mark as attack if prediction is not 'normal'
            df[label_col] = (df["prediction"].astype(str).str.lower() != "normal").astype(int)
        else:
            raise ValueError(f"Label column '{label_col}' not found and no fallback available")
    
    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    df_enc = df.copy()
    for col in categorical_cols:
        if col in df_enc.columns:
            codes, _ = pd.factorize(df_enc[col].astype(str), na_sentinel=-1)
            df_enc[col] = codes.astype(float)
    
    # Extract features
    available_feats = [c for c in feature_cols if c in df_enc.columns]
    if len(available_feats) < len(feature_cols):
        # Pad missing features with zeros
        for col in feature_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0.0
        available_feats = feature_cols
    
    # Build sequences per user
    X_seq = []
    y_seq = []
    
    for user_id, user_df in df_enc.groupby(user_id_col):
        if len(user_df) < min_flows_per_user:
            continue
        
        user_feats = user_df[available_feats].to_numpy(dtype=float)
        user_labels = user_df[label_col].to_numpy(dtype=int)
        
        # Create sliding windows
        for i in range(0, len(user_feats) - seq_len + 1, seq_len // 2):  # 50% overlap
            window_feats = user_feats[i:i + seq_len]
            window_labels = user_labels[i:i + seq_len]
            
            if len(window_feats) == seq_len:
                # Sequence label: 1 if any flow in sequence is attack, else 0
                seq_label = int(np.any(window_labels == 1))
                X_seq.append(window_feats)
                y_seq.append(seq_label)
    
    if not X_seq:
        raise ValueError(f"No sequences built. Need at least {min_flows_per_user} flows per user.")
    
    X_arr = np.stack(X_seq, axis=0)
    y_arr = np.asarray(y_seq, dtype=int)
    
    return X_arr, y_arr


def train_behavioral_lstm_from_data(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    model_path: str = "beh_lstm.h5",
    epochs: int = 10,
    batch_size: int = 64,
    timesteps: Optional[int] = None,
) -> str:
    """
    Train behavioral LSTM model from sequences.
    
    Args:
        X_seq: (N_seq, T, D) sequence features
        y_seq: (N_seq,) binary labels
        model_path: Path to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        timesteps: Override timesteps (default: use from X_seq shape)
    
    Returns:
        Path to saved model
    """
    if X_seq.ndim != 3:
        raise ValueError(f"X_seq must be 3D (N_seq, T, D), got shape {X_seq.shape}")
    if y_seq.ndim != 1 or y_seq.shape[0] != X_seq.shape[0]:
        raise ValueError(f"y_seq must be 1D with same first dimension as X_seq")
    
    timesteps = timesteps or X_seq.shape[1]
    features = X_seq.shape[2]
    
    # Save sequences temporarily
    temp_x = f"temp_X_seq_{int(time.time())}.npy"
    temp_y = f"temp_y_seq_{int(time.time())}.npy"
    
    try:
        np.save(temp_x, X_seq)
        np.save(temp_y, y_seq)
        
        # Call training script
        cmd = [
            sys.executable, "train_beh_lstm.py",
            "--x", temp_x,
            "--y", temp_y,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--out", model_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        return model_path
    finally:
        # Clean up temporary files
        for f in [temp_x, temp_y]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def collect_and_train_cycle(
    collected_data: pd.DataFrame,
    artifact_dir: str = ".",
    collection_duration_min: int = 10,
    seq_len: int = 10,
    epochs: int = 10,
    batch_size: int = 64,
    user_id_col: str = "user_id",
) -> Dict[str, Any]:
    """
    Complete cycle: build sequences from collected data and train model.
    
    Args:
        collected_data: DataFrame with collected live data
        artifact_dir: Directory to save the model
        collection_duration_min: Duration of collection (for logging)
        seq_len: Sequence length
        epochs: Training epochs
        batch_size: Training batch size
        user_id_col: User identifier column
    
    Returns:
        Dictionary with results and statistics
    """
    results = {
        "success": False,
        "error": None,
        "sequences_built": 0,
        "model_path": None,
        "training_time": 0.0,
    }
    
    try:
        # Build sequences
        t0 = time.time()
        X_seq, y_seq = build_sequences_from_live_data(
            df=collected_data,
            user_id_col=user_id_col,
            seq_len=seq_len,
        )
        results["sequences_built"] = len(X_seq)
        
        # Train model
        model_path = os.path.join(artifact_dir, "beh_lstm.h5")
        train_behavioral_lstm_from_data(
            X_seq=X_seq,
            y_seq=y_seq,
            model_path=model_path,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        results["model_path"] = model_path
        results["training_time"] = time.time() - t0
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

