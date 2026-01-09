# Comprehensive Model Evaluation Guide

This guide explains how to generate comprehensive evaluation metrics and visualizations for the MLP, LSTM, and Behavioral LSTM models.

## Prerequisites

Install required visualization packages:
```bash
pip install matplotlib seaborn
```

Or ensure they're in your requirements file:
```bash
pip install -r requirements_streamlit.txt
```

## Quick Start

### Step 1: Run Comprehensive Evaluation

Evaluate all models and generate metrics and visualizations:

```bash
python comprehensive_evaluation.py --test_data KDDTest_plus.csv --artifact_dir .
```

This will:
- Load all available models (MLP, LSTM, Behavioral LSTM)
- Evaluate them on the test dataset
- Compute metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, False Positive Rate
- Generate visualizations:
  - Confusion matrices for each model
  - ROC curves for each model
  - Precision-Recall curves for each model
  - Comparative analysis charts
  - Time-series plots of combined risk scores

Results are saved in the `evaluation_results/` directory.

### Step 2: Generate Training Curves (Optional)

To generate training/validation accuracy and loss curves, retrain models with history tracking:

```bash
python generate_training_curves.py \
    --train_data archive/KDDTrain+_20Percent.txt \
    --val_data archive/KDDTest+.txt \
    --artifact_dir . \
    --epochs_mlp 15 \
    --epochs_lstm 8
```

**Note:** This will retrain the models, which may take some time. If you already have trained models and just want evaluation metrics, skip this step.

## Detailed Usage

### Comprehensive Evaluation Script

```bash
python comprehensive_evaluation.py [OPTIONS]
```

**Options:**
- `--test_data PATH`: Path to test data (CSV or NSL-KDD format). Default: `KDDTest_plus.csv`
- `--artifact_dir PATH`: Directory containing model artifacts (.h5 files). Default: `.`
- `--output_dir PATH`: Directory to save results. Default: `evaluation_results`
- `--sample_size N`: Evaluate on a sample of N records (useful for quick testing). Default: Use all data
- `--threshold FLOAT`: Classification threshold for binary predictions. Default: `0.5`

**Example with custom options:**
```bash
python comprehensive_evaluation.py \
    --test_data KDDTest_plus.csv \
    --artifact_dir . \
    --output_dir my_results \
    --sample_size 10000 \
    --threshold 0.5
```

### Training Curves Script

```bash
python generate_training_curves.py [OPTIONS]
```

**Options:**
- `--train_data PATH`: Path to training data (required)
- `--val_data PATH`: Path to validation data (required)
- `--artifact_dir PATH`: Directory containing preprocessing artifacts. Default: `.`
- `--output_dir PATH`: Directory to save training curves. Default: `training_curves`
- `--epochs_mlp N`: Number of epochs for MLP training. Default: `15`
- `--epochs_lstm N`: Number of epochs for LSTM training. Default: `8`
- `--batch_size N`: Batch size for training. Default: `256`
- `--models MODEL [MODEL ...]`: Models to train. Choices: `mlp`, `lstm`. Default: `['mlp', 'lstm']`

**Example:**
```bash
python generate_training_curves.py \
    --train_data archive/KDDTrain+_20Percent.txt \
    --val_data archive/KDDTest+.txt \
    --epochs_mlp 20 \
    --epochs_lstm 10 \
    --models mlp lstm
```

## Output Files

### Evaluation Results (`evaluation_results/`)

1. **metrics_summary.csv**: Table of all metrics for all models
2. **detailed_results.json**: Detailed metrics including confusion matrix values
3. **confusion_matrix_*.png**: Confusion matrix for each model
4. **roc_curve_*.png**: ROC curve for each model
5. **precision_recall_*.png**: Precision-Recall curve for each model
6. **comparative_analysis.png**: Side-by-side comparison of all models
7. **combined_risk_scores.png**: Time-series plots of risk scores

### Training Curves (`training_curves/`)

1. **training_curves_mlp.png**: MLP training/validation accuracy and loss
2. **training_curves_lstm.png**: LSTM training/validation accuracy and loss
3. **training_history_mlp.json**: Raw training history data for MLP
4. **training_history_lstm.json**: Raw training history data for LSTM

## Metrics Explained

### Accuracy
Percentage of correctly classified samples.

### Precision
Of all samples predicted as attacks, how many were actually attacks.
```
Precision = TP / (TP + FP)
```

### Recall (Sensitivity)
Of all actual attacks, how many were correctly identified.
```
Recall = TP / (TP + FN)
```

### F1-Score
Harmonic mean of precision and recall.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### ROC-AUC
Area under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes. Higher is better (max = 1.0).

### False Positive Rate (FPR)
Percentage of normal samples incorrectly classified as attacks.
```
FPR = FP / (FP + TN)
```

## Understanding Visualizations

### Confusion Matrix
Shows the breakdown of predictions:
- **True Negatives (TN)**: Normal correctly identified as normal
- **False Positives (FP)**: Normal incorrectly identified as attack
- **False Negatives (FN)**: Attack incorrectly identified as normal
- **True Positives (TP)**: Attack correctly identified as attack

### ROC Curve
Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at different thresholds. The area under the curve (AUC) indicates model performance.

### Precision-Recall Curve
Plots Precision vs Recall at different thresholds. Useful when classes are imbalanced.

### Combined Risk Scores
Time-series visualization showing how risk scores change over time for different users/sessions. Red dots indicate actual attack occurrences.

## Troubleshooting

### Issue: "Module not found: matplotlib" or "seaborn"
```bash
pip install matplotlib seaborn
```

### Issue: "Model not found" errors
- Ensure model files exist in the artifact directory:
  - `att_det_mlp.h5` (required)
  - `att_det_lstm.h5` (optional)
  - `beh_lstm.h5` (optional)
- Check that `--artifact_dir` points to the correct directory

### Issue: "Test data must have a 'label' column"
- Ensure your test CSV has a column named `label`
- For NSL-KDD format files (.txt), the script will automatically use the correct column structure

### Issue: Behavioral LSTM evaluation fails
- Behavioral LSTM requires sequences, not individual samples
- The script will attempt to create sequences automatically
- If it fails, ensure you have enough samples in your test data

### Issue: Training curves script takes too long
- Reduce `--epochs_mlp` and `--epochs_lstm`
- Use `--sample_size` to evaluate on a subset of data
- Skip training curves if you only need evaluation metrics

## Example Workflow

1. **Evaluate existing models:**
   ```bash
   python comprehensive_evaluation.py --test_data KDDTest_plus.csv
   ```

2. **View results:**
   ```bash
   ls evaluation_results/
   # Open the PNG files to view visualizations
   # Check metrics_summary.csv for numerical results
   ```

3. **Generate training curves (if needed):**
   ```bash
   python generate_training_curves.py \
       --train_data archive/KDDTrain+_20Percent.txt \
       --val_data archive/KDDTest+.txt
   ```

4. **Combine results:**
   - Use the visualizations in your presentation/report
   - Reference the metrics in your analysis
   - Include comparative analysis for model selection

## Notes

- The evaluation scripts work with both CSV and NSL-KDD format files
- For large datasets, use `--sample_size` to speed up evaluation
- Training curves require retraining; this may take time depending on dataset size
- All visualizations are saved as high-resolution PNG files (300 DPI) suitable for presentations

