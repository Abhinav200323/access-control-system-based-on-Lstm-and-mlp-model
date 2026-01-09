"""
Training script for a behavioural LSTM model on per-user sequences.

This is a scaffold: it expects you to prepare NumPy arrays:
  - X_seq.npy: shape (N_seq, T, D) sequences of KDD-style features
  - y_seq.npy: shape (N_seq,) binary labels (0 = normal, 1 = attack)

You can adapt the data-loading section to your own log format.
The resulting model is saved as `beh_lstm.h5` and can be used by
`behavioral_lstm.BehavioralLSTM` in live mode.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(timesteps: int, features: int) -> keras.Model:
    inputs = keras.Input(shape=(timesteps, features))
    x = layers.Masking()(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=str, default="X_seq.npy", help="Path to sequence features .npy")
    parser.add_argument("--y", type=str, default="y_seq.npy", help="Path to sequence labels .npy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", type=str, default="beh_lstm.h5")
    args = parser.parse_args()

    if not os.path.exists(args.x) or not os.path.exists(args.y):
        raise SystemExit(
            f"Expected training data files {args.x} and {args.y} to exist. "
            "Prepare per-user sequences and labels and save them as NumPy arrays."
        )

    X_seq = np.load(args.x)
    y_seq = np.load(args.y)

    if X_seq.ndim != 3:
        raise SystemExit(f"X_seq must be 3D (N_seq, T, D), got shape {X_seq.shape}")
    if y_seq.ndim != 1 or y_seq.shape[0] != X_seq.shape[0]:
        raise SystemExit("y_seq must be 1D with same first dimension as X_seq")

    timesteps = X_seq.shape[1]
    features = X_seq.shape[2]

    model = build_model(timesteps, features)
    model.summary()

    # Split data for validation
    split_idx = int(len(X_seq) * 0.8)
    X_train = X_seq[:split_idx]
    y_train = y_seq[:split_idx]
    X_val = X_seq[split_idx:]
    y_val = y_seq[split_idx:]
    
    model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        shuffle=True,
    )

    model.save(args.out)
    print(f"Saved behavioural LSTM to {args.out}")
    
    # Compute optimal threshold from validation set
    try:
        import json
        val_pred = model.predict(X_val, verbose=0)
        val_pred = np.asarray(val_pred).flatten()
        
        # Sweep thresholds to maximize F1
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.3, 0.9, 100):
            y_hat = (val_pred >= t).astype(int)
            tp = int(np.sum((y_hat == 1) & (y_val == 1)))
            fp = int(np.sum((y_hat == 1) & (y_val == 0)))
            fn = int(np.sum((y_hat == 0) & (y_val == 1)))
            
            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0
            else:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        
        # Save threshold metadata
        th_path = os.path.splitext(args.out)[0] + ".threshold.json"
        meta = {
            "model": os.path.basename(args.out),
            "threshold": best_t,
            "metric": "F1",
            "f1": best_f1,
            "precision": float(tp / max(tp + fp, 1)) if tp + fp > 0 else 0.0,
            "recall": float(tp / max(tp + fn, 1)) if tp + fn > 0 else 0.0
        }
        with open(th_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved learned threshold to: {th_path} (F1={best_f1:.4f}, t={best_t:.3f})")
    except Exception as e:
        print(f"Threshold computation skipped: {e}")


if __name__ == "__main__":
    main()


