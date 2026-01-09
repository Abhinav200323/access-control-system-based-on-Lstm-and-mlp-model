from __future__ import annotations
"""
Train an LSTM model on NSL-KDD data and save it as att_det_lstm.h5.

Usage (PowerShell):
  python train_lstm.py --train .\archive\KDDTrain+_20Percent.txt --val .\archive\KDDTest+.txt \
      --artifact_dir . --epochs 8 --batch_size 256

Notes:
- Reuses AttackDetector preprocessing (one-hot + z-score). Converts to dense for training.
- Uses existing label_binarizer if available; otherwise creates a simple index mapping.
"""

import argparse
from typing import Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
from scipy import sparse
import json

from attack_detection_pipeline import AttackDetector, KDD99_COLUMNS
from lstm_model import train_and_save_lstm


def _load_dataframe(path: str) -> pd.DataFrame:
    cols = KDD99_COLUMNS  # includes label and difficulty at the end
    df = pd.read_csv(path, header=None, names=cols)
    return df


def _extract_X_y(det: AttackDetector, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Keep only required 41 features + optional label for y
    X_df = df[det.required_input_columns()].copy()
    X_enc = det._preprocess(X_df)  # sparse CSR
    if sparse.issparse(X_enc):
        X = X_enc.toarray()
    else:
        X = np.asarray(X_enc)

    # Labels (if present)
    y: np.ndarray
    if "label" in df.columns:
        labels = df["label"].astype(str).values
        if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
            # Map labels to indices using existing binarizer
            classes = list(det.label_binarizer.classes_)
            class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_idx.get(lbl, 0) for lbl in labels], dtype=int)
        else:
            # Create a simple mapping locally
            uniq = sorted(pd.unique(labels))
            class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(uniq)}
            y = np.array([class_to_idx[lbl] for lbl in labels], dtype=int)
    else:
        # If no labels, create dummy zeros (not ideal for real training)
        y = np.zeros((X.shape[0],), dtype=int)

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.path.join("archive", "KDDTrain+_20Percent.txt"))
    parser.add_argument("--val", type=str, default=os.path.join("archive", "KDDTest+.txt"))
    parser.add_argument("--artifact_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--output", type=str, default="att_det_lstm.h5")
    args = parser.parse_args()

    det = AttackDetector(artifact_dir=args.artifact_dir)

    print(f"Loading train data from: {args.train}")
    df_tr = _load_dataframe(args.train)
    X_tr, y_tr = _extract_X_y(det, df_tr)
    print(f"Train shapes: X={X_tr.shape}, y={y_tr.shape}")

    print(f"Loading val data from: {args.val}")
    df_va = _load_dataframe(args.val)
    X_va, y_va = _extract_X_y(det, df_va)
    print(f"Val shapes:   X={X_va.shape}, y={y_va.shape}")

    saved = train_and_save_lstm(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.output,
        lstm_units=args.lstm_units,
    )

    print(f"Saved LSTM model to: {saved}")

    # Compute an operating threshold from validation data (risk vs normal)
    try:
        from tensorflow import keras  # local import to avoid TF cost when not needed
        model = keras.models.load_model(saved, compile=False)
        # Predict probabilities on validation set
        Xva_seq = X_va.reshape((-1, 1, X_va.shape[1]))
        proba = model.predict(Xva_seq, verbose=0)
        # Determine risky set
        risky_mask = None
        if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
            classes = list(det.label_binarizer.classes_)
            risky_idx = [i for i, c in enumerate(classes) if str(c).lower() != "normal"]
            if not risky_idx:
                risky_idx = list(range(1, proba.shape[1]))
            risky_mask = np.zeros(proba.shape[1], dtype=bool)
            risky_mask[risky_idx] = True
            y_true_bin = np.array([0 if str(classes[y]).lower()=="normal" else 1 for y in y_va], dtype=int)
        else:
            # Assume class 0 normal
            risky_mask = np.ones(proba.shape[1], dtype=bool)
            risky_mask[0] = False
            y_true_bin = (y_va != 0).astype(int)

        # Score = max probability over risky classes
        risky_scores = proba[:, risky_mask].max(axis=1)

        # Sweep thresholds to maximize F1
        best_t = 0.9
        best_f1 = -1.0
        for t in np.linspace(0.5, 0.99, 100):
            y_hat = (risky_scores >= t).astype(int)
            tp = int(np.sum((y_hat == 1) & (y_true_bin == 1)))
            fp = int(np.sum((y_hat == 1) & (y_true_bin == 0)))
            fn = int(np.sum((y_hat == 0) & (y_true_bin == 1)))
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

        meta = {"model": os.path.basename(saved), "threshold": best_t, "metric": "F1", "f1": best_f1}
        th_path = os.path.splitext(saved)[0] + ".threshold.json"
        with open(th_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved learned threshold to: {th_path} (F1={best_f1:.4f}, t={best_t:.3f})")
    except Exception as e:
        print(f"Threshold computation skipped due to error: {e}")


if __name__ == "__main__":
    main()


