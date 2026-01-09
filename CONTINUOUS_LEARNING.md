# Continuous Learning Feature

## Overview

The Continuous Learning feature automatically collects live network data, trains behavioral LSTM models, and cycles continuously. This is especially useful when:

- The behavioral LSTM model has compatibility issues (e.g., "Unknown layer: 'NotEqual'")
- You want to adapt the model to your specific network environment
- You need to continuously improve the model based on real-time traffic patterns

## How It Works

1. **Data Collection**: Collects live captured network flows for a specified duration (default: 10 minutes)
2. **Sequence Building**: Automatically builds per-user sequences from collected data
3. **Model Training**: Trains a behavioral LSTM model on the collected sequences
4. **Data Cleanup**: Deletes collected data after training
5. **Auto-Cycle**: Automatically starts the next collection cycle

## Usage

### Step 1: Enable Continuous Learning

1. Go to the **Live Capture** tab in Streamlit
2. Scroll to the **"🔄 Continuous Learning"** section
3. Check **"Enable Continuous Learning"**

### Step 2: Configure Settings

- **Collection Duration**: How long to collect data before training (1-60 minutes)
- **Sequence Length**: Number of timesteps per sequence (5-50, default: 10)
- **Training Epochs**: Number of training epochs (5-50, default: 10)

### Step 3: Start Data Collection

1. Click **"🚀 Start Data Collection"**
2. The system will collect live network flows
3. Monitor progress:
   - Elapsed time
   - Remaining time
   - Number of flows collected

### Step 4: Automatic Training

When the collection period completes:
1. Click **"🎓 Train Model Now"** (or it will auto-train if configured)
2. The system will:
   - Build sequences from collected data
   - Train the behavioral LSTM model
   - Save the model as `beh_lstm.h5`
   - Reload the detector with the new model
   - Start the next collection cycle automatically

## Requirements

- **Live capture must be running**: Start sniffing before enabling continuous learning
- **Sufficient data**: Need at least 5 flows per user to build sequences
- **User identification**: Data must have `user_id` or `src_ip` column for per-user sequences

## Data Flow

```
Live Capture → Data Collection → Sequence Building → Model Training → Model Deployment → (Repeat)
```

## Technical Details

### Sequence Building

- Sequences are built per-user (using `user_id` or `src_ip`)
- Each sequence contains `seq_len` consecutive flows from the same user
- Sequence label: 1 if any flow in the sequence is an attack, else 0
- Sequences use 50% overlap for better coverage

### Model Training

- Uses the same architecture as `train_beh_lstm.py`
- Trains on collected sequences
- Saves model to artifact directory
- Automatically reloads detector after training

### Data Management

- Collected data is stored in Streamlit session state during collection
- Data is automatically deleted after training
- No persistent storage of collected data (privacy-friendly)

## Troubleshooting

### "No sequences built"

**Cause**: Not enough flows per user or collection period too short

**Solution**:
- Increase collection duration
- Ensure live capture is actively collecting traffic
- Check that `user_id` or `src_ip` column exists in captured data

### Training fails

**Cause**: Insufficient data or model compatibility issues

**Solution**:
- Collect more data (increase duration)
- Check that collected data has required KDD features
- Verify TensorFlow version compatibility

### Model not reloading

**Cause**: Detector reload failed after training

**Solution**:
- Manually click "Load / Reload Detector" in the sidebar
- Check that `beh_lstm.h5` was created in artifact directory
- Verify model file is not corrupted

## Best Practices

1. **Collection Duration**: 
   - Start with 10-15 minutes for initial training
   - Adjust based on network traffic volume
   - Longer durations = more data but slower adaptation

2. **Sequence Length**:
   - Default (10) works well for most cases
   - Shorter (5-7) for faster adaptation
   - Longer (15-20) for more context

3. **Training Epochs**:
   - Start with 10 epochs
   - Increase if model performance is poor
   - Decrease if training takes too long

4. **Monitoring**:
   - Watch the number of flows collected
   - Ensure sufficient data before training
   - Check model performance after each cycle

## Privacy & Security

- Collected data is only stored temporarily in memory
- Data is automatically deleted after training
- No persistent storage of network flows
- Model only learns behavioral patterns, not raw packet data

## Integration with Existing Models

The continuous learning feature:
- Works alongside existing MLP and LSTM models
- Only trains the behavioral LSTM component
- Does not affect flow-based detection models
- Can be enabled/disabled independently

## Example Workflow

1. **Initial Setup**:
   ```
   Start Live Capture → Enable Continuous Learning → Set 10 min collection
   ```

2. **First Cycle**:
   ```
   Collect 10 min of data → Train model → Deploy model → Start next cycle
   ```

3. **Ongoing**:
   ```
   Continuous cycles every 10 minutes
   Model adapts to current network patterns
   ```

## Advanced Configuration

### Custom Sequence Building

You can modify `continuous_learning.py` to customize:
- Sequence building logic
- Feature selection
- User identification method
- Label assignment strategy

### Custom Training

Modify training parameters in the Streamlit UI or edit `continuous_learning.py` to:
- Change model architecture
- Adjust training hyperparameters
- Add custom callbacks
- Implement transfer learning

## Limitations

- Requires active network traffic for meaningful training
- Needs sufficient flows per user (minimum 5)
- Training time depends on data volume and epochs
- Model compatibility still depends on TensorFlow version

## Future Enhancements

Potential improvements:
- Automatic cycle management (no manual intervention)
- Incremental learning (update existing model)
- Model versioning and rollback
- Performance metrics tracking
- Alert system for model degradation

