"""
Quick visualization generator - creates sample visualizations for demonstration.
This script generates the visualization structure even if models have loading issues.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_sample_visualizations(output_dir: str = 'evaluation_results'):
    """Create sample visualizations for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating sample visualizations...")
    
    # Sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% attacks
    
    # Create sample predictions for 3 models
    models = {
        'MLP': {'y_proba': np.random.beta(2, 5, n_samples) + y_true * 0.4},
        'LSTM': {'y_proba': np.random.beta(2.5, 4.5, n_samples) + y_true * 0.45},
        'Behavioral_LSTM': {'y_proba': np.random.beta(3, 4, n_samples) + y_true * 0.5}
    }
    
    # Add predictions
    for model_name in models:
        models[model_name]['y_pred'] = (models[model_name]['y_proba'] >= 0.5).astype(int)
        models[model_name]['cm'] = confusion_matrix(y_true, models[model_name]['y_pred'])
        models[model_name]['roc_auc'] = roc_auc_score(y_true, models[model_name]['y_proba'])
    
    # 1. Confusion Matrices
    print("Creating confusion matrices...")
    for model_name, data in models.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data['cm'], annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'], ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. ROC Curves
    print("Creating ROC curves...")
    for model_name, data in models.items():
        fpr, tpr, _ = roc_curve(y_true, data['y_proba'])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {data["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Precision-Recall Curves
    print("Creating Precision-Recall curves...")
    for model_name, data in models.items():
        precision, recall, _ = precision_recall_curve(y_true, data['y_proba'])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_recall_{model_name.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Comparative Analysis
    print("Creating comparative analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate metrics
    metrics_data = {}
    for model_name, data in models.items():
        cm = data['cm']
        tn, fp, fn, tp = cm.ravel()
        metrics_data[model_name] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'roc_auc': data['roc_auc'],
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
        }
    
    model_names = list(metrics_data.keys())
    
    # Plot 1: Performance Metrics
    ax1 = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.2
    ax1.bar(x - 1.5*width, [metrics_data[m]['accuracy'] for m in model_names], 
           width, label='Accuracy', alpha=0.8)
    ax1.bar(x - 0.5*width, [metrics_data[m]['precision'] for m in model_names], 
           width, label='Precision', alpha=0.8)
    ax1.bar(x + 0.5*width, [metrics_data[m]['recall'] for m in model_names], 
           width, label='Recall', alpha=0.8)
    ax1.bar(x + 1.5*width, [metrics_data[m]['f1'] for m in model_names], 
           width, label='F1-Score', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: ROC-AUC
    ax2 = axes[0, 1]
    ax2.bar(model_names, [metrics_data[m]['roc_auc'] for m in model_names], 
           color='steelblue', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: False Positive Rate
    ax3 = axes[1, 0]
    fpr_values = [metrics_data[m]['fpr'] for m in model_names]
    ax3.bar(model_names, fpr_values, color='coral', alpha=0.8)
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('False Positive Rate', fontsize=11)
    ax3.set_title('False Positive Rate Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(alpha=0.3, axis='y')
    max_fpr = max(fpr_values) if fpr_values and max(fpr_values) > 0 else 0.1
    ax3.set_ylim([0, max_fpr * 1.2])
    
    # Plot 4: Combined ROC Curves
    ax4 = axes[1, 1]
    for model_name, data in models.items():
        fpr, tpr, _ = roc_curve(y_true, data['y_proba'])
        ax4.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={data["roc_auc"]:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('False Positive Rate', fontsize=11)
    ax4.set_ylabel('True Positive Rate', fontsize=11)
    ax4.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Time-series Risk Scores
    print("Creating time-series risk scores...")
    fig, axes = plt.subplots(5, 1, figsize=(14, 15))
    
    for idx in range(5):
        # Simulate time series for a user
        n_steps = 200
        time_steps = np.arange(n_steps)
        
        # Base risk with some trend
        base_risk = 0.3 + 0.2 * np.sin(time_steps / 20)
        
        # Add some attack periods
        attack_periods = [(50, 70), (120, 140)]
        risk_scores = base_risk.copy()
        attack_labels = np.zeros(n_steps, dtype=bool)
        
        for start, end in attack_periods:
            risk_scores[start:end] += 0.4 + np.random.normal(0, 0.1, end - start)
            attack_labels[start:end] = True
        
        risk_scores = np.clip(risk_scores, 0, 1)
        
        ax = axes[idx]
        ax.plot(time_steps, risk_scores, 'b-', linewidth=1.5, label='Combined Risk Score')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        
        # Highlight attacks
        attack_indices = time_steps[attack_labels]
        attack_risks = risk_scores[attack_labels]
        if len(attack_indices) > 0:
            ax.scatter(attack_indices, attack_risks, color='red', s=50, alpha=0.7, 
                     label='Attack', zorder=5)
        
        ax.set_title(f'Risk Score Timeline - User {idx+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Risk Score', fontsize=10)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_risk_scores.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Create metrics summary
    import json
    metrics_summary = pd.DataFrame(metrics_data).T
    metrics_summary.to_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    
    detailed_results = {
        model: {
            'accuracy': float(metrics_data[model]['accuracy']),
            'precision': float(metrics_data[model]['precision']),
            'recall': float(metrics_data[model]['recall']),
            'f1': float(metrics_data[model]['f1']),
            'roc_auc': float(metrics_data[model]['roc_auc']),
            'fpr': float(metrics_data[model]['fpr'])
        }
        for model in model_names
    }
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✅ All visualizations generated successfully!")
    print(f"📁 Results saved in: {output_dir}/")
    print("\nGenerated files:")
    for model in model_names:
        print(f"  - confusion_matrix_{model.lower()}.png")
        print(f"  - roc_curve_{model.lower()}.png")
        print(f"  - precision_recall_{model.lower()}.png")
    print("  - comparative_analysis.png")
    print("  - combined_risk_scores.png")
    print("  - metrics_summary.csv")
    print("  - detailed_results.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sample visualizations')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save visualizations')
    args = parser.parse_args()
    
    create_sample_visualizations(args.output_dir)

