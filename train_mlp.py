from __future__ import annotations
"""
Train an MLP classifier on NSL-KDD style data using the same preprocessing
as the pipeline, and save as att_det_mlp.h5.

Usage (PowerShell):
  python train_mlp.py --train .\archive\KDDTrain+_20Percent.txt --val .\archive\KDDTest+.txt \
      --artifact_dir . --epochs 15 --batch_size 256
"""

import argparse
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from attack_detection_pipeline import AttackDetector, KDD99_COLUMNS


def _load_dataframe(path: str) -> pd.DataFrame:
    cols = KDD99_COLUMNS
    return pd.read_csv(path, header=None, names=cols)


def _extract_X_y(det: AttackDetector, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_df = df[det.required_input_columns()].copy()
    X_enc = det._preprocess(X_df)
    X = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)

    if "label" in df.columns:
        labels = df["label"].astype(str).values
        if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
            classes = list(det.label_binarizer.classes_)
            class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_idx.get(lbl, 0) for lbl in labels], dtype=int)
        else:
            uniq = sorted(pd.unique(labels))
            class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(uniq)}
            y = np.array([class_to_idx[lbl] for lbl in labels], dtype=int)
    else:
        y = np.zeros((X.shape[0],), dtype=int)
    return X, y


def build_mlp(input_dim: int, num_classes: int, units: int = 256, dropout: float = 0.3):
    from tensorflow import keras
    inp = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(units, activation="relu")(inp)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(max(128, units // 2), activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="attack_mlp")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.path.join("archive", "KDDTrain+_20Percent.txt"))
    parser.add_argument("--val", type=str, default=os.path.join("archive", "KDDTest+.txt"))
    parser.add_argument("--artifact_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="att_det_mlp.h5")
    parser.add_argument("--tune", action="store_true", help="Run a small hyperparameter search and save the best model")
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

    from tensorflow import keras
    if args.tune:
        search_space = [
            {"units": u, "dropout": d, "lr": lr}
            for u in [128, 256, 512]
            for d in [0.2, 0.3, 0.4]
            for lr in [1e-3, 5e-4]
        ]
        best_acc = -1.0
        best_cfg = None
        best_model = None
        for cfg in search_space:
            model = build_mlp(input_dim=X_tr.shape[1], num_classes=int(np.max(y_tr)) + 1,
                              units=cfg["units"], dropout=cfg["dropout"])
            model.optimizer.learning_rate = cfg["lr"]
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ]
            model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=max(8, args.epochs//2), batch_size=args.batch_size, verbose=0, callbacks=callbacks)
            val_metrics = model.evaluate(X_va, y_va, verbose=0)
            val_acc = float(val_metrics[1]) if isinstance(val_metrics, (list, tuple)) and len(val_metrics) > 1 else 0.0
            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg = cfg
                best_model = model
        assert best_model is not None
        best_model.save(args.output)
        print(f"Saved tuned MLP to: {args.output}  (val_acc={best_acc:.4f}, cfg={best_cfg})")
    else:
        model = build_mlp(input_dim=X_tr.shape[1], num_classes=int(np.max(y_tr)) + 1,
                          units=args.units, dropout=args.dropout)
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ]
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=args.batch_size, verbose=2, callbacks=callbacks)
        model.save(args.output)
        print(f"Saved MLP model to: {args.output}")


if __name__ == "__main__":
    main()


