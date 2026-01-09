from __future__ import annotations
"""
Behavioral LSTM model for per-user / per-identity sequences.

This component is optional. If a file named `beh_lstm.h5` is present in the
artifact directory and TensorFlow/Keras is available, it will be used in the
Streamlit live mode to provide a per-user behavioural risk score.
"""

import os
from typing import Dict

import numpy as np

try:
    from tensorflow import keras
    _KERAS_ERR = None
except Exception as e:  # pragma: no cover - environment dependent
    keras = None  # type: ignore
    _KERAS_ERR = e


class BehavioralLSTM:
    """
    Thin wrapper around a Keras LSTM model saved as `beh_lstm.h5`.

    Expected model input shape: (batch, timesteps, features)
    Expected output: scalar risk / probability per sequence in [0, 1].
    """

    def __init__(self, artifact_dir: str = ".") -> None:
        if _KERAS_ERR is not None:
            raise ImportError(f"TensorFlow/Keras is not available: {_KERAS_ERR}")

        # Resolve to absolute path for consistency
        abs_artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
        path = os.path.join(abs_artifact_dir, "beh_lstm.h5")
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Behavioural LSTM artifact 'beh_lstm.h5' not found at {path}"
            )

        # Try loading with multiple fallback strategies for version compatibility
        load_errors = []
        last_error = None
        
        # Strategy 1: Normal load
        try:
            self.model = keras.models.load_model(path, compile=False)
        except Exception as e1:
            last_error = e1
            error_str = str(e1)
            load_errors.append(f"Normal load: {error_str}")
            
            # Check if it's the "NotEqual" or similar layer error
            is_layer_error = "Unknown layer" in error_str or "NotEqual" in error_str
            
            # Strategy 2: Load with safe_mode=False (TF 2.13+)
            if is_layer_error:
                try:
                    self.model = keras.models.load_model(
                        path, 
                        compile=False,
                        safe_mode=False
                    )
                except (TypeError, AttributeError):
                    # safe_mode parameter doesn't exist in older TF versions, continue to next strategy
                    pass
                except Exception as e2:
                    load_errors.append(f"Safe mode: {e2}")
                    last_error = e2
                else:
                    # Success with safe_mode=False
                    last_error = None
            
            # Strategy 3: Load with custom_objects (if still failing)
            if last_error is not None and is_layer_error:
                try:
                    import tensorflow as tf
                    
                    # Register TensorFlow operations that might be used in Lambda layers
                    # These are often the cause of "Unknown layer" errors
                    custom_objects = {}
                    
                    # Add common TensorFlow operations
                    if hasattr(tf, 'not_equal'):
                        custom_objects['NotEqual'] = tf.not_equal
                        custom_objects['not_equal'] = tf.not_equal
                    if hasattr(tf, 'equal'):
                        custom_objects['Equal'] = tf.equal
                        custom_objects['equal'] = tf.equal
                    if hasattr(tf, 'greater'):
                        custom_objects['Greater'] = tf.greater
                    if hasattr(tf, 'less'):
                        custom_objects['Less'] = tf.less
                    
                    # Try loading with custom_objects
                    self.model = keras.models.load_model(
                        path, 
                        compile=False,
                        custom_objects=custom_objects
                    )
                    last_error = None
                except Exception as e3:
                    load_errors.append(f"Custom objects: {e3}")
                    last_error = e3
            
            # If all strategies failed, raise a helpful error
            if last_error is not None:
                error_msg = (
                    f"Failed to load behavioral LSTM model from {path}.\n\n"
                    f"Error: {str(last_error)}\n\n"
                )
                
                if "NotEqual" in str(last_error) or "Unknown layer" in str(last_error):
                    error_msg += (
                        "This is a TensorFlow version compatibility issue.\n"
                        "The model was likely saved with a different TensorFlow version.\n\n"
                        "Solutions:\n"
                        "  1. Retrain the model with your current TensorFlow version:\n"
                        "     python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --out beh_lstm.h5\n\n"
                        "  2. Update TensorFlow to the latest version:\n"
                        "     pip install --upgrade tensorflow\n\n"
                        "  3. Check your TensorFlow version:\n"
                        "     python -c 'import tensorflow as tf; print(tf.__version__)'\n"
                    )
                else:
                    error_msg += (
                        "Attempted loading strategies:\n" + 
                        "\n".join(f"  - {err}" for err in load_errors) + "\n\n"
                        "This might be a corrupted model file or version incompatibility.\n"
                        "Try retraining the model."
                    )
                
                raise RuntimeError(error_msg) from last_error
        shp = getattr(self.model, "input_shape", None)
        if shp is None or len(shp) != 3:
            raise ValueError(
                f"Behavioral model must take 3D input (batch, timesteps, features), got {shp}"
            )
        self.timesteps = int(shp[1])
        self.features = int(shp[2])

    def score_users(self, user_sequences: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Given a mapping user_id -> sequence array (timesteps, features),
        return user_id -> risk score in [0, 1].
        """
        xs = []
        keys = []
        for uid, seq in user_sequences.items():
            arr = np.asarray(seq, dtype=float)
            # Expect 2D sequence with matching feature dimension
            if arr.ndim != 2 or arr.shape[1] != self.features:
                continue
            if arr.shape[0] < self.timesteps:
                # Not enough history for this user yet
                continue
            if arr.shape[0] > self.timesteps:
                # Use the most recent `timesteps` steps
                arr = arr[-self.timesteps :, :]
            xs.append(arr)
            keys.append(uid)

        if not xs:
            return {}

        X = np.stack(xs, axis=0)
        out = self.model.predict(X, verbose=0)
        out = np.asarray(out, dtype=float)
        if out.ndim > 1:
            scores = out.reshape((out.shape[0], -1))[:, -1]
        else:
            scores = out

        scores = np.clip(scores, 0.0, 1.0)
        return {uid: float(s) for uid, s in zip(keys, scores)}


def get_behavioral_lstm(artifact_dir: str = ".") -> BehavioralLSTM:
    """
    Convenience helper used by Streamlit to load the behavioural model.
    """
    return BehavioralLSTM(artifact_dir=artifact_dir)


