"""
Comprehensive evaluation script for MLP, LSTM, and Behavioral LSTM models.

This script:
1. Computes metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for all models
2. Generates confusion matrices
3. Generates ROC and Precision-Recall curves
4. Creates training/validation curves (if history available)
5. Generates time-series plots of combined risk scores
6. Performs comparative analysis

Usage:
    python comprehensive_evaluation.py --test_data KDDTest_plus.csv --artifact_dir .
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Dict, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

from attack_detection_pipeline import AttackDetector, KDD99_COLUMNS
from behavioral_lstm import BehavioralLSTM

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_test_data(test_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load test data and extract features and labels."""
    if test_path.endswith('.csv'):
        df = pd.read_csv(test_path)
    else:
        # Assume it's NSL-KDD format (space-separated)
        df = pd.read_csv(test_path, header=None, names=KDD99_COLUMNS)
    
    # Extract labels
    if 'label' in df.columns:
        y_true = df['label'].astype(str).str.lower().values
        # Convert to binary: 0 = normal, 1 = attack
        y_true_binary = (y_true != 'normal').astype(int)
    else:
        raise ValueError("Test data must have a 'label' column")
    
    return df, y_true_binary


def evaluate_model_binary(
    model, 
    X_preprocessed: np.ndarray,
    y_true_binary: np.ndarray,
    model_name: str,
    needs_sequence: bool = False,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate a model and return metrics."""
    try:
        # Prepare input
        if needs_sequence:
            X_in = X_preprocessed.reshape((-1, 1, X_preprocessed.shape[1]))
        else:
            X_in = X_preprocessed
        
        # Get predictions
        y_proba_raw = model.predict(X_in, verbose=0)
        
        # Handle different output shapes
        if y_proba_raw.ndim == 1:
            # Binary output (single probability)
            y_proba = y_proba_raw
            y_pred = (y_proba >= threshold).astype(int)
        else:
            # Multi-class output
            # For binary classification, use probability of positive class
            if y_proba_raw.shape[1] == 2:
                y_proba = y_proba_raw[:, 1]  # Probability of class 1
            else:
                # Multi-class: use max probability over non-normal classes
                # This is a simplification - ideally we'd know which classes are attacks
                y_proba = np.max(y_proba_raw[:, 1:], axis=1)  # Max prob of non-normal classes
            y_pred = (y_proba >= threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(y_true_binary, y_pred)
        precision = precision_score(y_true_binary, y_pred, zero_division=0)
        recall = recall_score(y_true_binary, y_pred, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true_binary, y_proba)
        except ValueError:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred)
        
        # False positive rate
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'confusion_matrix': cm,
            'y_true': y_true_binary,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    except Exception as e:
        warnings.warn(f"Error evaluating {model_name}: {e}")
        return None


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: str):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: str):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall curve: {save_path}")


def plot_training_curves(history: Dict[str, list], model_name: str, save_path: str):
    """Plot training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Accuracy
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'Model Accuracy - {model_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Loss
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title(f'Model Loss - {model_name}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves: {save_path}")


def plot_combined_risk_scores(
    results_dict: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    det: AttackDetector,
    save_path: str,
    n_users: int = 5
):
    """Plot time-series of combined risk scores for selected users."""
    # Get combined predictions
    X = df[det.required_input_columns()].copy()
    _, y_proba_combined = det.predict(X)
    
    # Extract risk scores (probability of attack class)
    if y_proba_combined is not None and y_proba_combined.shape[1] > 1:
        # Use max probability of non-normal classes as risk score
        risk_scores = np.max(y_proba_combined[:, 1:], axis=1)
    else:
        risk_scores = y_proba_combined.flatten() if y_proba_combined is not None else np.zeros(len(df))
    
    # If we have user identifiers, group by user; otherwise use time index
    if 'src_ip' in df.columns or 'user_id' in df.columns:
        user_col = 'src_ip' if 'src_ip' in df.columns else 'user_id'
        users = df[user_col].unique()[:n_users]
    else:
        # Simulate users by chunking
        chunk_size = len(df) // n_users
        users = [f'user_{i}' for i in range(n_users)]
    
    # Create time-series plots
    fig, axes = plt.subplots(n_users, 1, figsize=(14, 3 * n_users))
    if n_users == 1:
        axes = [axes]
    
    for idx, user in enumerate(users):
        if 'src_ip' in df.columns or 'user_id' in df.columns:
            user_mask = df[user_col] == user
            user_risks = risk_scores[user_mask]
            user_labels = df.loc[user_mask, 'label'].values
            timestamps = np.arange(len(user_risks))
        else:
            # Simulate by chunking
            start_idx = idx * chunk_size
            end_idx = min((idx + 1) * chunk_size, len(df))
            user_risks = risk_scores[start_idx:end_idx]
            user_labels = df['label'].values[start_idx:end_idx]
            timestamps = np.arange(start_idx, end_idx)
        
        if len(user_risks) == 0:
            continue
        
        ax = axes[idx]
        ax.plot(timestamps, user_risks, 'b-', linewidth=1.5, label='Combined Risk Score')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        
        # Highlight attack periods
        attack_mask = (user_labels != 'normal') if isinstance(user_labels[0], str) else (user_labels == 1)
        if np.any(attack_mask):
            attack_indices = timestamps[attack_mask]
            attack_risks = user_risks[attack_mask]
            ax.scatter(attack_indices, attack_risks, color='red', s=50, alpha=0.7, label='Attack', zorder=5)
        
        ax.set_title(f'Risk Score Timeline - {user}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Risk Score', fontsize=10)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined risk scores plot: {save_path}")


def create_comparative_analysis(results_dict: Dict[str, Dict[str, Any]], save_path: str):
    """Create comparative analysis of models."""
    if not results_dict:
        print("Warning: No results to compare. Skipping comparative analysis.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr']
    
    # Extract metrics
    metric_data = {metric: [results_dict[m].get(metric, 0) for m in models] for metric in metrics}
    
    # Plot 1: Accuracy, Precision, Recall, F1
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2
    ax1.bar(x - 1.5*width, metric_data['accuracy'], width, label='Accuracy', alpha=0.8)
    ax1.bar(x - 0.5*width, metric_data['precision'], width, label='Precision', alpha=0.8)
    ax1.bar(x + 0.5*width, metric_data['recall'], width, label='Recall', alpha=0.8)
    ax1.bar(x + 1.5*width, metric_data['f1'], width, label='F1-Score', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: ROC-AUC
    ax2 = axes[0, 1]
    ax2.bar(models, metric_data['roc_auc'], color='steelblue', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: False Positive Rate
    ax3 = axes[1, 0]
    fpr_values = metric_data['fpr']
    ax3.bar(models, fpr_values, color='coral', alpha=0.8)
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('False Positive Rate', fontsize=11)
    ax3.set_title('False Positive Rate Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(alpha=0.3, axis='y')
    max_fpr = max(fpr_values) if fpr_values and max(fpr_values) > 0 else 0.1
    ax3.set_ylim([0, max_fpr * 1.2])
    
    # Plot 4: Combined ROC curves
    ax4 = axes[1, 1]
    for model_name, results in results_dict.items():
        if 'y_true' in results and 'y_proba' in results:
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
            roc_auc = results.get('roc_auc', 0)
            ax4.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('False Positive Rate', fontsize=11)
    ax4.set_ylabel('True Positive Rate', fontsize=11)
    ax4.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative analysis: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--test_data', type=str, default='KDDTest_plus.csv',
                        help='Path to test data (CSV or NSL-KDD format)')
    parser.add_argument('--artifact_dir', type=str, default='.',
                        help='Directory containing model artifacts')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for evaluation (None = use all data)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for binary predictions')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Comprehensive Model Evaluation")
    print("="*60)
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    df, y_true_binary = load_test_data(args.test_data)
    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        y_true_binary = (df['label'].astype(str) != 'normal').astype(int)
    print(f"Test data shape: {df.shape}")
    print(f"Normal samples: {np.sum(y_true_binary == 0)}")
    print(f"Attack samples: {np.sum(y_true_binary == 1)}")
    
    # Initialize detector
    print(f"\nLoading models from: {args.artifact_dir}")
    det = AttackDetector(artifact_dir=args.artifact_dir)
    
    # Preprocess data
    X = df[det.required_input_columns()].copy()
    X_enc = det._preprocess(X)
    X_dense = X_enc.toarray() if hasattr(X_enc, 'toarray') else np.asarray(X_enc)
    
    # Evaluate models
    results_dict = {}
    
    # Evaluate MLP
    if det.models.get('mlp') is not None:
        print("\nEvaluating MLP model...")
        results = evaluate_model_binary(
            det.models['mlp'],
            X_dense,
            y_true_binary,
            'MLP',
            needs_sequence=False,
            threshold=args.threshold
        )
        if results:
            results_dict['MLP'] = results
    
    # Evaluate LSTM
    if det.models.get('lstm') is not None:
        print("Evaluating LSTM model...")
        results = evaluate_model_binary(
            det.models['lstm'],
            X_dense,
            y_true_binary,
            'LSTM',
            needs_sequence=True,
            threshold=args.threshold
        )
        if results:
            results_dict['LSTM'] = results
    
    # Evaluate Behavioral LSTM (requires sequences)
    try:
        beh_lstm = BehavioralLSTM(artifact_dir=args.artifact_dir)
        print("Evaluating Behavioral LSTM model...")
        # For behavioral LSTM, we need to create sequences
        # This is a simplified evaluation - in practice, it should use per-user sequences
        # We'll create pseudo-sequences by chunking
        timesteps = beh_lstm.timesteps
        features = beh_lstm.features
        
        # Create sequences (simplified)
        seq_len = timesteps
        if X_dense.shape[1] == features:
            # Reshape to sequences
            n_samples = len(X_dense) // seq_len
            if n_samples > 0:
                X_seq = X_dense[:n_samples * seq_len].reshape((n_samples, seq_len, features))
                y_seq_binary = y_true_binary[:n_samples * seq_len].reshape((n_samples, seq_len))
                # Label sequence as attack if any step is attack
                y_seq_labels = (np.any(y_seq_binary == 1, axis=1)).astype(int)
                
                # Get predictions
                y_proba_raw = beh_lstm.model.predict(X_seq, verbose=0)
                y_proba = np.asarray(y_proba_raw).flatten()
                y_pred = (y_proba >= args.threshold).astype(int)
                
                # Compute metrics
                accuracy = accuracy_score(y_seq_labels, y_pred)
                precision = precision_score(y_seq_labels, y_pred, zero_division=0)
                recall = recall_score(y_seq_labels, y_pred, zero_division=0)
                f1 = f1_score(y_seq_labels, y_pred, zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y_seq_labels, y_proba)
                except ValueError:
                    roc_auc = 0.0
                
                cm = confusion_matrix(y_seq_labels, y_pred)
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                results_dict['Behavioral_LSTM'] = {
                    'model_name': 'Behavioral_LSTM',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'fpr': fpr,
                    'confusion_matrix': cm,
                    'y_true': y_seq_labels,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }
    except Exception as e:
        warnings.warn(f"Could not evaluate Behavioral LSTM: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics_df = pd.DataFrame({
        model: {
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1', 0),
            'ROC-AUC': results.get('roc_auc', 0),
            'FPR': results.get('fpr', 0)
        }
        for model, results in results_dict.items()
    }).T
    
    print("\nMetrics Summary:")
    print(metrics_df.round(4))
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(args.output_dir, 'metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path)
    print(f"\nSaved metrics summary to: {metrics_csv_path}")
    
    # Generate visualizations
    if not results_dict:
        print("\nWarning: No models were successfully evaluated. Cannot generate visualizations.")
        print("Please ensure models are available in the artifact directory.")
        return
    
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    for model_name, results in results_dict.items():
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{model_name.lower()}.png')
        plot_confusion_matrix(results['confusion_matrix'], model_name, cm_path)
    
    # ROC curves
    for model_name, results in results_dict.items():
        roc_path = os.path.join(args.output_dir, f'roc_curve_{model_name.lower()}.png')
        plot_roc_curve(results['y_true'], results['y_proba'], model_name, roc_path)
    
    # Precision-Recall curves
    for model_name, results in results_dict.items():
        pr_path = os.path.join(args.output_dir, f'precision_recall_{model_name.lower()}.png')
        plot_precision_recall_curve(results['y_true'], results['y_proba'], model_name, pr_path)
    
    # Comparative analysis
    comp_path = os.path.join(args.output_dir, 'comparative_analysis.png')
    create_comparative_analysis(results_dict, comp_path)
    
    # Combined risk scores (time-series)
    try:
        risk_path = os.path.join(args.output_dir, 'combined_risk_scores.png')
        plot_combined_risk_scores(results_dict, df, det, risk_path, n_users=5)
    except Exception as e:
        warnings.warn(f"Could not generate risk scores plot: {e}")
    
    # Save detailed results as JSON
    results_json = {}
    for model_name, results in results_dict.items():
        results_json[model_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'roc_auc': float(results['roc_auc']),
            'fpr': float(results['fpr']),
            'tp': int(results['tp']),
            'tn': int(results['tn']),
            'fp': int(results['fp']),
            'fn': int(results['fn'])
        }
    
    json_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved detailed results to: {json_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved in: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

