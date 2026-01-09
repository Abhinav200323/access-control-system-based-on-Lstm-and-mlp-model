from __future__ import annotations
"""
Utility for building and training an LSTM model compatible with the attack detection
pipeline. Saves the trained model as att_det_lstm.h5 in the project directory.

Note: The current pipeline encodes each sample as a single feature vector. To use
an LSTM (which expects sequences), we reshape each vector into a pseudo-sequence
of length 1 with feature_dim steps. This provides a simple way to integrate an
LSTM without changing upstream featurization. For better temporal modeling,
provide true sequences during training and adapt the pipeline accordingly.
"""

from typing import Optional
import numpy as np

try:
    from tensorflow import keras
except Exception as e:  # pragma: no cover
    raise ImportError(f"TensorFlow/Keras is required for LSTM training: {e}")


def build_lstm_model(input_dim: int, num_classes: int, lstm_units: int = 64, dropout: float = 0.2) -> keras.Model:
    """
    Build a simple LSTM classifier that operates on a pseudo-sequence of length 1.
    Input shape: (None, 1, input_dim)
    Output: num_classes (softmax)
    """
    inputs = keras.Input(shape=(1, input_dim), name="features_seq")
    x = keras.layers.LSTM(lstm_units, name="lstm")(inputs)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(max(32, num_classes * 2), activation="relu", name="dense1")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="attack_lstm")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_lstm(X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None, epochs: int = 10, batch_size: int = 256,
                        model_path: str = "att_det_lstm.h5", lstm_units: int = 64) -> str:
    """
    Train the LSTM on vector inputs by reshaping to (N, 1, D) and save to model_path.
    Returns saved model path.
    """
    if X_train.ndim != 2:
        raise ValueError(f"Expected X_train shape (N, D), got {X_train.shape}")
    if X_val is not None and X_val.ndim != 2:
        raise ValueError(f"Expected X_val shape (N, D), got {None if X_val is None else X_val.shape}")

    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]

    model = build_lstm_model(input_dim=input_dim, num_classes=num_classes, lstm_units=lstm_units)

    Xtr = X_train.reshape((-1, 1, input_dim))
    Xva = None if X_val is None else X_val.reshape((-1, 1, input_dim))

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        Xtr,
        y_train,
        validation_data=None if X_val is None else (Xva, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    model.save(model_path)
    return model_path


