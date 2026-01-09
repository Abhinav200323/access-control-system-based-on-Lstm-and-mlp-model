"""
Generate training/validation curves by retraining models with history tracking.

This script retrains models and saves training history to generate accuracy/loss curves.

Usage:
    python generate_training_curves.py --train_data KDDTrain+_20Percent.txt --val_data KDDTest+.txt --artifact_dir .
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from attack_detection_pipeline import AttackDetector, KDD99_COLUMNS
from train_mlp import build_mlp, _load_dataframe, _extract_X_y
from lstm_model import build_lstm_model
from train_beh_lstm import build_model

from tensorflow import keras

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def train_mlp_with_history(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 256,
    units: int = 256,
    dropout: float = 0.3
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train MLP and return model with training history."""
    model = build_mlp(
        input_dim=X_train.shape[1],
        num_classes=int(np.max(y_train)) + 1,
        units=units,
        dropout=dropout
    )
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks
    )
    
    return model, history


def train_lstm_with_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 8,
    batch_size: int = 256,
    lstm_units: int = 64
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train LSTM and return model with training history."""
    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]
    
    model = build_lstm_model(input_dim=input_dim, num_classes=num_classes, lstm_units=lstm_units)
    
    Xtr_seq = X_train.reshape((-1, 1, input_dim))
    Xva_seq = X_val.reshape((-1, 1, input_dim))
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]
    
    history = model.fit(
        Xtr_seq, y_train,
        validation_data=(Xva_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks
    )
    
    return model, history


def plot_training_curves(history: keras.callbacks.History, model_name: str, save_path: str):
    """Plot training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    hist_dict = history.history
    epochs = range(1, len(hist_dict['loss']) + 1)
    
    # Accuracy
    ax1.plot(epochs, hist_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    if 'val_accuracy' in hist_dict:
        ax1.plot(epochs, hist_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    ax1.set_title(f'Model Accuracy - {model_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Loss
    ax2.plot(epochs, hist_dict['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in hist_dict:
        ax2.plot(epochs, hist_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax2.set_title(f'Model Loss - {model_name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves: {save_path}")


def save_history_json(history: keras.callbacks.History, save_path: str):
    """Save training history to JSON file."""
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(save_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Saved training history to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate training curves')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (CSV or NSL-KDD format)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data (CSV or NSL-KDD format)')
    parser.add_argument('--artifact_dir', type=str, default='.',
                        help='Directory containing preprocessing artifacts')
    parser.add_argument('--output_dir', type=str, default='training_curves',
                        help='Directory to save training curves')
    parser.add_argument('--epochs_mlp', type=int, default=15,
                        help='Number of epochs for MLP training')
    parser.add_argument('--epochs_lstm', type=int, default=8,
                        help='Number of epochs for LSTM training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--models', type=str, nargs='+', default=['mlp', 'lstm'],
                        choices=['mlp', 'lstm'],
                        help='Models to train')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Generating Training Curves")
    print("="*60)
    
    # Load data
    print(f"\nLoading training data from: {args.train_data}")
    det = AttackDetector(artifact_dir=args.artifact_dir)
    
    if args.train_data.endswith('.csv'):
        df_train = pd.read_csv(args.train_data)
    else:
        df_train = _load_dataframe(args.train_data)
    
    X_train, y_train = _extract_X_y(det, df_train)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    print(f"\nLoading validation data from: {args.val_data}")
    if args.val_data.endswith('.csv'):
        df_val = pd.read_csv(args.val_data)
    else:
        df_val = _load_dataframe(args.val_data)
    
    X_val, y_val = _extract_X_y(det, df_val)
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    
    # Train models and generate curves
    if 'mlp' in args.models:
        print("\n" + "="*60)
        print("Training MLP model...")
        print("="*60)
        model_mlp, history_mlp = train_mlp_with_history(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs_mlp,
            batch_size=args.batch_size
        )
        
        # Plot curves
        plot_training_curves(history_mlp, 'MLP', 
                           os.path.join(args.output_dir, 'training_curves_mlp.png'))
        save_history_json(history_mlp, 
                         os.path.join(args.output_dir, 'training_history_mlp.json'))
        
        # Save model
        mlp_path = os.path.join(args.output_dir, 'att_det_mlp_with_history.h5')
        model_mlp.save(mlp_path)
        print(f"Saved MLP model to: {mlp_path}")
    
    if 'lstm' in args.models:
        print("\n" + "="*60)
        print("Training LSTM model...")
        print("="*60)
        model_lstm, history_lstm = train_lstm_with_history(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs_lstm,
            batch_size=args.batch_size
        )
        
        # Plot curves
        plot_training_curves(history_lstm, 'LSTM',
                           os.path.join(args.output_dir, 'training_curves_lstm.png'))
        save_history_json(history_lstm,
                         os.path.join(args.output_dir, 'training_history_lstm.json'))
        
        # Save model
        lstm_path = os.path.join(args.output_dir, 'att_det_lstm_with_history.h5')
        model_lstm.save(lstm_path)
        print(f"Saved LSTM model to: {lstm_path}")
    
    print("\n" + "="*60)
    print("Training curves generation complete!")
    print(f"Results saved in: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

