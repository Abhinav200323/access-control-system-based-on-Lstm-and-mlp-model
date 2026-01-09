
from __future__ import annotations
import os, sys, warnings
from typing import List, Tuple, Optional, Dict, Any
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import zscore
import json

_SKLEARN = None
_KERAS = None
_JOBLIB = None

def _get_compatible_input_layer():
    """Create a compatible InputLayer class that handles batch_shape parameter."""
    if isinstance(_KERAS, Exception):
        return None
    
    class CompatibleInputLayer(_KERAS.layers.InputLayer):
        """InputLayer that handles batch_shape parameter for TensorFlow version compatibility."""
        def __init__(self, *args, **kwargs):
            # Convert batch_shape to input_shape if needed (for TF 2.10+ compatibility)
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1:
                    # Convert batch_shape [None, ...] to input_shape [...]
                    kwargs['input_shape'] = batch_shape[1:]
            super().__init__(*args, **kwargs)
    
    return CompatibleInputLayer

def _lazy_imports():
    global _SKLEARN, _KERAS, _JOBLIB
    if _SKLEARN is None:
        import importlib
        _SKLEARN = importlib.import_module("sklearn")
    if _KERAS is None:
        try:
            from tensorflow import keras
            _KERAS = keras
        except Exception as e:
            _KERAS = e
    if _JOBLIB is None:
        import importlib
        try:
            _JOBLIB = importlib.import_module("joblib")
        except Exception:
            _JOBLIB = None

DEFAULT_ARTIFACTS = {
    "mlp": "att_det_mlp.h5",
    "lstm": "att_det_lstm.h5",  # optional
    "onehot": "onehot_encoder.pkl",
    "label_binarizer": "label_binarizer.pkl"
}

KDD99_COLUMNS: List[str] = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

CATEGORICAL: List[str] = ["protocol_type","service","flag"]
NUMERIC: List[str] = [c for c in KDD99_COLUMNS[:41] if c not in CATEGORICAL]

def _safe_unpickle(path: str) -> Any:
    import pickle, numpy as _np
    sys.modules.setdefault("numpy._core", _np.core)
    if path.endswith(".pkl"):
        try:
            if _JOBLIB is not None:
                # Suppress sklearn version warnings when loading pickled models
                with warnings.catch_warnings():
                    # Filter out InconsistentVersionWarning from sklearn
                    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
                    warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
                    return _JOBLIB.load(path)
        except Exception:
            pass
    # Suppress sklearn version warnings for pickle.load as well
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
        warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
        with open(path, "rb") as f:
            return pickle.load(f)

class AttackDetector:
    def __init__(self, artifact_dir: str=".", force_cpu: bool=False) -> None:
        _lazy_imports()
        # Resolve artifact_dir to absolute path for consistency
        self.artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Ensure Keras is loaded before trying to create compatible layers
        if isinstance(_KERAS, Exception):
            warnings.warn(f"Keras unavailable: {_KERAS}")

        # Helper function to get absolute paths to artifacts
        def p(k: str) -> str:
            path = os.path.join(self.artifact_dir, DEFAULT_ARTIFACTS[k])
            return os.path.abspath(path)

        self.onehot = None
        self.label_binarizer = None
        if os.path.exists(p("onehot")):
            try: self.onehot = _safe_unpickle(p("onehot"))
            except Exception as e: warnings.warn(f"Failed to load onehot encoder: {e}")
        else:
            warnings.warn("onehot_encoder.pkl not found; runtime fallback encoder will be used.")
        if os.path.exists(p("label_binarizer")):
            try: self.label_binarizer = _safe_unpickle(p("label_binarizer"))
            except Exception as e: warnings.warn(f"Failed to load label_binarizer: {e}")
        else:
            warnings.warn("label_binarizer.pkl not found; numeric class ids will be returned.")

        # Load only Keras models: MLP (required) and LSTM (optional)
        self.models: Dict[str, Any] = {"mlp": None, "lstm": None}
        if isinstance(_KERAS, Exception):
            warnings.warn(f"Keras unavailable: {_KERAS}")
        else:
            mlp_path = p("mlp")
            if os.path.exists(mlp_path):
                try:
                    import tensorflow as tf
                    
                    # Get compatible InputLayer class
                    CompatibleInputLayer = _get_compatible_input_layer()
                    
                    # Build custom_objects dictionary
                    custom_objects = {}
                    if CompatibleInputLayer:
                        custom_objects['InputLayer'] = CompatibleInputLayer
                        custom_objects['CompatibleInputLayer'] = CompatibleInputLayer
                    
                    # Add common TensorFlow operations that might be in Lambda layers
                    if hasattr(tf, 'not_equal'):
                        custom_objects['NotEqual'] = tf.not_equal
                    if hasattr(tf, 'equal'):
                        custom_objects['Equal'] = tf.equal
                    if hasattr(tf, 'greater'):
                        custom_objects['Greater'] = tf.greater
                    if hasattr(tf, 'less'):
                        custom_objects['Less'] = tf.less
                    
                    # Try multiple loading strategies
                    load_success = False
                    last_error = None
                    
                    # Strategy 1: Load with custom_objects
                    if custom_objects:
                        try:
                            self.models["mlp"] = _KERAS.models.load_model(
                                mlp_path, compile=False, custom_objects=custom_objects
                            )
                            load_success = True
                        except Exception as e1:
                            last_error = e1
                    
                    # Strategy 2: Load with safe_mode=False (TF 2.13+)
                    if not load_success:
                        try:
                            self.models["mlp"] = _KERAS.models.load_model(
                                mlp_path, compile=False, safe_mode=False, 
                                custom_objects=custom_objects if custom_objects else None
                            )
                            load_success = True
                        except (TypeError, AttributeError, Exception) as e2:
                            if last_error is None:
                                last_error = e2
                    
                    # Strategy 3: Normal load (fallback)
                    if not load_success:
                        try:
                            self.models["mlp"] = _KERAS.models.load_model(mlp_path, compile=False)
                            load_success = True
                        except Exception as e3:
                            if last_error is None:
                                last_error = e3
                    
                    if not load_success and last_error:
                        raise last_error
                    
                    # Verify model was actually loaded
                    if self.models["mlp"] is not None:
                        # Test that model can be called (basic sanity check)
                        try:
                            test_input = np.zeros((1, 1), dtype=float)
                            _ = self.models["mlp"].predict(test_input, verbose=0)
                        except Exception as test_err:
                            warnings.warn(f"MLP model loaded but prediction test failed: {test_err}")
                            self.models["mlp"] = None
                        
                except Exception as e:
                    warnings.warn(f"Failed to load Keras MLP from {mlp_path}: {e}")
                    self.models["mlp"] = None
            else:
                warnings.warn(f"att_det_mlp.h5 not found at {mlp_path}; MLP predictions will be unavailable.")

            lstm_path = p("lstm")
            if os.path.exists(lstm_path):
                try:
                    import tensorflow as tf
                    
                    # Get compatible InputLayer class
                    CompatibleInputLayer = _get_compatible_input_layer()
                    
                    # Build custom_objects dictionary
                    custom_objects = {}
                    if CompatibleInputLayer:
                        custom_objects['InputLayer'] = CompatibleInputLayer
                        custom_objects['CompatibleInputLayer'] = CompatibleInputLayer
                    
                    # Add common TensorFlow operations that might be in Lambda layers
                    if hasattr(tf, 'not_equal'):
                        custom_objects['NotEqual'] = tf.not_equal
                    if hasattr(tf, 'equal'):
                        custom_objects['Equal'] = tf.equal
                    if hasattr(tf, 'greater'):
                        custom_objects['Greater'] = tf.greater
                    if hasattr(tf, 'less'):
                        custom_objects['Less'] = tf.less
                    
                    # Try multiple loading strategies
                    load_success = False
                    last_error = None
                    
                    # Strategy 1: Load with custom_objects
                    if custom_objects:
                        try:
                            self.models["lstm"] = _KERAS.models.load_model(
                                lstm_path, compile=False, custom_objects=custom_objects
                            )
                            load_success = True
                        except Exception as e1:
                            last_error = e1
                    
                    # Strategy 2: Load with safe_mode=False (TF 2.13+)
                    if not load_success:
                        try:
                            self.models["lstm"] = _KERAS.models.load_model(
                                lstm_path, compile=False, safe_mode=False,
                                custom_objects=custom_objects if custom_objects else None
                            )
                            load_success = True
                        except (TypeError, AttributeError, Exception) as e2:
                            if last_error is None:
                                last_error = e2
                    
                    # Strategy 3: Normal load (fallback)
                    if not load_success:
                        try:
                            self.models["lstm"] = _KERAS.models.load_model(lstm_path, compile=False)
                            load_success = True
                        except Exception as e3:
                            if last_error is None:
                                last_error = e3
                    
                    if not load_success and last_error:
                        raise last_error
                    
                    # Verify model was actually loaded
                    if self.models["lstm"] is not None:
                        # Test that model can be called (basic sanity check)
                        try:
                            # Get input shape from model
                            input_shape = getattr(self.models["lstm"], "input_shape", None)
                            if input_shape and len(input_shape) >= 2:
                                # LSTM expects 3D input (batch, timesteps, features)
                                n_features = int(input_shape[-1]) if len(input_shape) > 1 else 100
                                n_timesteps = int(input_shape[1]) if len(input_shape) > 2 else 1
                                test_input = np.zeros((1, n_timesteps, n_features), dtype=float)
                            else:
                                # Fallback: assume standard shape
                                test_input = np.zeros((1, 1, 100), dtype=float)
                            _ = self.models["lstm"].predict(test_input, verbose=0)
                        except Exception as test_err:
                            warnings.warn(f"LSTM model loaded but prediction test failed: {test_err}")
                            self.models["lstm"] = None
                        
                except Exception as e:
                    warnings.warn(f"Failed to load Keras LSTM from {lstm_path}: {e}")
                    self.models["lstm"] = None
            else:
                warnings.warn(f"att_det_lstm.h5 not found at {lstm_path}; LSTM predictions will be unavailable.")

        # Load learned threshold if present (for LSTM)
        self.learned_threshold: float | None = None
        try:
            th_path = os.path.join(self.artifact_dir, os.path.splitext(DEFAULT_ARTIFACTS["lstm"])[0] + ".threshold.json")
            if os.path.exists(th_path):
                with open(th_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    t = float(meta.get("threshold", 0.9))
                    if 0.0 < t < 1.0:
                        self.learned_threshold = t
        except Exception as e:
            warnings.warn(f"Failed to read learned threshold: {e}")

    @staticmethod
    def required_input_columns() -> List[str]:
        return KDD99_COLUMNS[:41]

    def predict(self, X: pd.DataFrame | List[Dict[str, Any]] | np.ndarray
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_df = self._to_dataframe(X)
        X_enc = self._preprocess(X_df)

        # Collect probabilities from available neural models
        probas: Dict[str, np.ndarray] = {}
        # Convert to dense for Keras models
        try:
            X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)
        except Exception:
            X_dense = np.asarray(X_enc)
        for name in ["mlp", "lstm"]:
            model = self.models.get(name)
            if model is None:
                continue
            try:
                # Determine expected input rank (MLP expects 2D, LSTM expects 3D [N, 1, D])
                input_shape = getattr(model, "input_shape", None)
                needs_sequence = False
                if input_shape is not None:
                    # Keras shape like (None, D) or (None, 1, D)
                    needs_sequence = (len(input_shape) == 3)
                if not needs_sequence and getattr(model, "name", "").lower().find("lstm") != -1:
                    # Fallback heuristic by name
                    needs_sequence = True

                X_in = X_dense.reshape((-1, 1, X_dense.shape[1])) if needs_sequence else X_dense
                out = model.predict(X_in, verbose=0)
                # Ensure probabilities
                if out.ndim == 1:
                    proba = np.vstack([1 - out, out]).T
                else:
                    row_sums = out.sum(axis=1, keepdims=True)
                    if np.allclose(row_sums, 1.0, atol=1e-4):
                        proba = out
                    else:
                        exp = np.exp(out - np.max(out, axis=1, keepdims=True))
                        proba = exp / np.sum(exp, axis=1, keepdims=True)
                probas[name] = proba
            except Exception as e:
                warnings.warn(f"Inference failed for {name}: {e}")

        if not probas:
            # Provide helpful error message
            available_models = [name for name, model in self.models.items() if model is not None]
            if not available_models:
                raise RuntimeError(
                    f"No neural models (MLP/LSTM) loaded; cannot predict.\n"
                    f"Artifact directory: {self.artifact_dir}\n"
                    f"Expected MLP model at: {os.path.join(self.artifact_dir, DEFAULT_ARTIFACTS['mlp'])}\n"
                    f"Please ensure models are trained and available in the artifact directory."
                )
            else:
                raise RuntimeError(
                    f"Models loaded but prediction failed. Available models: {available_models}"
                )

        proba_list = list(probas.values())
        max_classes = max(p.shape[1] for p in proba_list)
        aligned = []
        for p in proba_list:
            if p.shape[1] == max_classes:
                aligned.append(p)
            else:
                pad = np.full((p.shape[0], max_classes), 1e-12)
                pad[:, :p.shape[1]] = p
                pad = pad / pad.sum(axis=1, keepdims=True)
                aligned.append(pad)
        y_proba = np.mean(np.stack(aligned, axis=0), axis=0)
        y_final = np.argmax(y_proba, axis=1)

        if self.label_binarizer is not None and hasattr(self.label_binarizer, "classes_"):
            classes = getattr(self.label_binarizer, "classes_", None)
            if classes is not None:
                y_labels = np.array([classes[i] for i in y_final])
                return y_labels, y_proba

        return y_final, y_proba

    def predict_with_revoke(self, X: pd.DataFrame | List[Dict[str, Any]] | np.ndarray,
                            threshold: float | None = None,
                            risky_labels: Optional[List[str]] = None
                           ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Predict and compute a revoke flag per row based on a probability threshold.

        - threshold: revoke if max class probability >= threshold and predicted label is risky
        - risky_labels: list of labels considered attacks; if None and label_binarizer is available,
          all classes except 'normal' (case-insensitive) are treated as risky. If no label names,
          all non-zero class indices are treated as risky.
        Returns: (y_pred, y_proba, revoke_flags[0/1])
        """
        y_pred, y_proba = self.predict(X)
        if y_proba is None:
            # Without probabilities, we cannot apply a probabilistic threshold; default to no revoke
            revoke = np.zeros_like(y_pred, dtype=int)
            return y_pred, None, revoke

        if threshold is None:
            threshold = self.learned_threshold if self.learned_threshold is not None else 0.9

        max_p = np.max(y_proba, axis=1)

        # Determine risky set
        if risky_labels is None:
            if self.label_binarizer is not None and hasattr(self.label_binarizer, "classes_"):
                classes = list(self.label_binarizer.classes_)
                risky_set = {c for c in classes if str(c).lower() != "normal"}
                y_labels = y_pred
            else:
                # Assume class index 0 is normal, others are risky
                risky_set = set(range(1, y_proba.shape[1]))
                y_labels = y_pred
        else:
            risky_set = set(risky_labels)
            y_labels = y_pred

        # Compute revoke: predicted label in risky set and high confidence
        revoke = np.array([
            1 if ((lbl in risky_set) and (p >= threshold)) else 0
            for lbl, p in zip(y_labels, max_p)
        ], dtype=int)

        return y_pred, y_proba, revoke

    def _to_dataframe(self, X: pd.DataFrame | List[Dict[str, Any]] | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            cols = self.required_input_columns()
            if X.shape[1] != len(cols):
                raise ValueError(f"Expected {len(cols)} columns, got {X.shape[1]}")
            df = pd.DataFrame(X, columns=cols)
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            raise TypeError("X must be DataFrame, ndarray, or list of dicts")

        req = self.required_input_columns()
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df[req]

    def _preprocess(self, df: pd.DataFrame):
        X_num = df[[c for c in NUMERIC if c in df.columns]].astype(float).to_numpy(copy=True)
        X_num = np.nan_to_num(X_num, copy=False, posinf=0.0, neginf=0.0)
        Xz = zscore(X_num, axis=0, nan_policy="omit") if X_num.size else X_num
        Xz = np.nan_to_num(Xz, copy=False)

        if self.onehot is not None and hasattr(self.onehot, "transform"):
            X_cat = self.onehot.transform(df[CATEGORICAL])
            if not sparse.issparse(X_cat):
                X_cat = sparse.csr_matrix(X_cat)
        else:
            X_cat_list = []
            for col in CATEGORICAL:
                vals, inv = np.unique(df[col].astype(str).values, return_inverse=True)
                rows = np.arange(df.shape[0])
                data = np.ones_like(inv, dtype=float)
                mat = sparse.coo_matrix((data, (rows, inv)), shape=(df.shape[0], len(vals))).tocsr()
                X_cat_list.append(mat)
            X_cat = sparse.hstack(X_cat_list, format="csr") if X_cat_list else sparse.csr_matrix((df.shape[0], 0))

        X_num_sparse = sparse.csr_matrix(Xz) if Xz.ndim == 2 else sparse.csr_matrix(Xz.reshape(-1, 1))
        X_enc = sparse.hstack([X_num_sparse, X_cat], format="csr")
        return X_enc

    @staticmethod
    def _majority_vote(pred_list: List[np.ndarray]) -> np.ndarray:
        # Deprecated in neural-only mode; kept for compatibility if needed elsewhere
        arr = np.stack(pred_list, axis=0)
        n_models, n = arr.shape
        y = np.empty(n, dtype=int)
        for i in range(n):
            vals, counts = np.unique(arr[:, i], return_counts=True)
            y[i] = vals[np.argmax(counts)]
        return y

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default=".")
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()

    det = AttackDetector(artifact_dir=args.artifact_dir)
    if args.csv:
        data = pd.read_csv(args.csv)
        y, proba = det.predict(data)
        print(json.dumps({"y_pred": y.tolist(), "proba_shape": None if proba is None else list(proba.shape)})[:2000])
    else:
        print("Loaded. Required columns:", det.required_input_columns())
