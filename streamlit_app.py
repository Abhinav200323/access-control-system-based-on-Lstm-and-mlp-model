import os
import io
import time
import subprocess
import sys
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict
from blocking_policy import BlockingPolicy

st.set_page_config(page_title="Attack Detection (Ensemble + SciPy)", layout="wide")

# ---------- Robust import for local pipeline module ----------
def _get_AttackDetector():
    """
    Try to import AttackDetector from attack_detection_pipeline.py either as a module
    on sys.path or from a file next to this streamlit_app.py.
    """
    try:
        from attack_detection_pipeline import AttackDetector  # type: ignore
        return AttackDetector
    except Exception:
        import importlib.util, sys
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "attack_detection_pipeline.py")
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("attack_detection_pipeline", cand)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore
            return mod.AttackDetector
        # If we get here, the file isn't present where we expect it.
        raise ImportError(
            "Could not import AttackDetector. Ensure attack_detection_pipeline.py "
            "is in the same folder as this file or installed on PYTHONPATH."
        )

@st.cache_resource
def load_detector(artifact_dir: str):
    """
    Load the AttackDetector with proper path resolution.
    """
    AttackDetector = _get_AttackDetector()
    # Resolve to absolute path for consistency
    abs_artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
    if not os.path.exists(abs_artifact_dir):
        raise FileNotFoundError(f"Artifact directory does not exist: {abs_artifact_dir}")
    return AttackDetector(artifact_dir=abs_artifact_dir)


@st.cache_resource
def load_behavioral_lstm(artifact_dir: str):
    """
    Lazy loader for optional behavioural LSTM model.
    Returns a BehavioralLSTM instance if available, otherwise raises.
    """
    from behavioral_lstm import get_behavioral_lstm

    return get_behavioral_lstm(artifact_dir=artifact_dir)


def _compute_adaptive_threshold(history, base: float = 0.9) -> float:
    """
    Compute a data-driven threshold from a history of max probabilities.
    Uses a high quantile to keep false positives low.
    """
    if not history:
        return float(base)
    arr = np.asarray(history, dtype=float)
    # 99th percentile, clipped to a reasonable range
    q = float(np.quantile(arr, 0.99))
    return float(np.clip(q, 0.5, 0.99))

def page_batch(det):
    st.subheader("📤 Upload CSV (Offline Mode)")
    st.caption("Upload a CSV file with KDD features for batch prediction. Uses MLP + LSTM ensemble.")
    
    # Check if models are loaded
    mlp_loaded = det.models.get("mlp") is not None
    lstm_loaded = det.models.get("lstm") is not None
    
    if not mlp_loaded:
        st.error("❌ **MLP model is required but not loaded!**")
        st.info("""
        **To fix this:**
        1. Go to the **Training** tab
        2. Train the MLP model
        3. Click **"Load / Reload Detector"** in the sidebar
        """)
        return
    
    # Initialize session state for batch predictions
    if "_batch_results" not in st.session_state:
        st.session_state["_batch_results"] = None
    if "_batch_df" not in st.session_state:
        st.session_state["_batch_df"] = None
    
    # Model status
    with st.expander("📊 Model Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            mlp_status = "✅ Loaded" if mlp_loaded else "❌ Not found"
            st.metric("MLP Model", mlp_status)
        with col2:
            lstm_status = "✅ Loaded" if lstm_loaded else "⚠️ Optional"
            st.metric("LSTM Model", lstm_status)
        with col3:
            st.metric("Ensemble Mode", "MLP + LSTM" if lstm_loaded else "MLP only")
    
    req_cols = det.required_input_columns()
    up = st.file_uploader("CSV with at least the 41 required columns", type=["csv"], key="u_csv")
    
    # Clear results if new file is uploaded
    if up is not None:
        # Check if this is a new file (different from stored one)
        if st.session_state.get("_uploaded_file_id") != id(up):
            st.session_state["_batch_results"] = None
            st.session_state["_batch_df"] = None
            st.session_state["_uploaded_file_id"] = id(up)
    
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.session_state["_batch_results"] = None
            return
        st.write("### Preview")
        st.dataframe(df.head(20), width='stretch')
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.session_state["_batch_results"] = None
            return
        X = df[req_cols].copy()
        dept_name = st.text_input("Department (optional)", value="", key="dept_name")
        
        col_thresh, col_info = st.columns([2, 1])
        with col_thresh:
            st.caption(f"Using threshold: {st.session_state.get('_threshold', 0.9):.3f}")
        with col_info:
            st.caption(f"Total rows: {len(X):,}")
        
        # Clear results button
        if st.session_state["_batch_results"] is not None:
            if st.button("🔄 Clear Results", key="clear_batch_results"):
                st.session_state["_batch_results"] = None
                st.session_state["_batch_df"] = None
                st.rerun()
        
        if st.button("🚀 Run Prediction", type="primary", key="run_csv"):
            with st.status("Running inference...", expanded=False):
                try:
                    t0 = time.time()
                    y_pred, y_proba, revoke = det.predict_with_revoke(X, threshold=st.session_state.get("_threshold", 0.9))
                    latency = time.time() - t0
                    st.success(f"Done in {latency:.2f}s ({len(X)/latency:.1f} rows/sec)")
                except RuntimeError as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("Please ensure models are properly loaded. Click 'Load / Reload Detector' in the sidebar.")
                    return
                except Exception as e:
                    st.error(f"❌ Unexpected error during prediction: {e}")
                    return
            
            # Store results in session state
            out = df.copy()
            out["prediction"] = y_pred
            out["revoke"] = revoke
            # Department-aware blocklist using risk scores if available
            policy = BlockingPolicy(default_threshold=st.session_state.get("_threshold", 0.9))
            dept = dept_name.strip() or None
            if dept:
                policy.set_department(dept, st.session_state.get("_threshold", 0.9))
            if y_proba is not None:
                try:
                    if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
                        classes = list(det.label_binarizer.classes_)
                        risky_idx = [i for i, c in enumerate(classes) if str(c).lower() != "normal"]
                    else:
                        risky_idx = list(range(1, y_proba.shape[1]))
                    risk_scores = np.max(y_proba[:, risky_idx], axis=1) if risky_idx else np.zeros((y_proba.shape[0],))
                except Exception:
                    risk_scores = np.max(y_proba, axis=1)
                out["risk_score"] = risk_scores
                out["block"] = policy.decide_revoke(risk_scores, dept=dept)
            else:
                out["block"] = out["revoke"]
            
            # Store in session state
            st.session_state["_batch_results"] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
                "revoke": revoke,
                "latency": latency,
                "out": out
            }
            st.session_state["_batch_df"] = df
            st.rerun()
    
    # Display results if they exist (prevents rerunning predictions on every widget change)
    if st.session_state["_batch_results"] is not None and st.session_state["_batch_df"] is not None:
        results = st.session_state["_batch_results"]
        y_pred = results["y_pred"]
        y_proba = results["y_proba"]
        revoke = results["revoke"]
        latency = results["latency"]
        out = results["out"]
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(y_pred))
        with col2:
            st.metric("Revoked", int(revoke.sum()), delta=f"{100*revoke.sum()/len(revoke):.1f}%")
        with col3:
            st.metric("Allowed", int((~revoke.astype(bool)).sum()), delta=f"{100*(1-revoke.sum()/len(revoke)):.1f}%")
        with col4:
            if y_proba is not None:
                avg_conf = np.max(y_proba, axis=1).mean()
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        res = pd.DataFrame({"prediction": y_pred, "revoke": revoke})
        st.write("### Predictions (first 100)")
        st.dataframe(res.head(100), width='stretch')
        
        # Aggregate visuals
        try:
            counts = pd.Series(revoke).value_counts().rename({0: "allow", 1: "revoke"})
            st.write("### Revoke vs Allow Distribution")
            st.bar_chart(counts)
            
            if y_proba is not None:
                import numpy as _np
                max_p = _np.max(y_proba, axis=1)
                
                # Prediction class distribution
                pred_counts = pd.Series(y_pred).value_counts()
                st.write("### Prediction Class Distribution")
                st.bar_chart(pred_counts)
                
                # Confidence histogram
                hist, edges = _np.histogram(max_p, bins=20, range=(0.0, 1.0))
                hist_df = pd.DataFrame({"prob_bin": edges[:-1], "count": hist})
                st.write("### Confidence Score Histogram")
                st.bar_chart(hist_df.set_index("prob_bin"))
                
                # Risk scores by prediction
                if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
                    classes = list(det.label_binarizer.classes_)
                    risky_idx = [i for i, c in enumerate(classes) if str(c).lower() != "normal"]
                    risk_scores = np.max(y_proba[:, risky_idx], axis=1) if risky_idx else np.max(y_proba, axis=1)
                else:
                    risk_scores = np.max(y_proba, axis=1)
                
                risk_by_pred = pd.DataFrame({
                    "prediction": y_pred,
                    "risk_score": risk_scores
                }).groupby("prediction")["risk_score"].mean().sort_values(ascending=False)
                st.write("### Average Risk Score by Prediction Class")
                st.bar_chart(risk_by_pred)
        except Exception as _:
            pass
        
        if y_proba is not None:
            st.write("### Detailed Probability Inspection")
            row_idx = st.number_input("Row index to inspect",
                                      min_value=0, max_value=len(res)-1, value=0, step=1, key="csv_row")
            proba_row = y_proba[int(row_idx)]
            proba_df = pd.DataFrame({"class_id": list(range(len(proba_row))), "prob": proba_row})
            if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
                classes = list(det.label_binarizer.classes_)
                proba_df["class_name"] = [classes[i] if i < len(classes) else f"class_{i}" for i in range(len(proba_row))]
                proba_df = proba_df.set_index("class_name")
            else:
                proba_df = proba_df.set_index("class_id")
            st.bar_chart(proba_df["prob"])
            st.caption(f"Row {row_idx}: Prediction = {y_pred[row_idx]}, Confidence = {np.max(proba_row):.3f}, Revoke = {revoke[row_idx]}")
        
        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download predictions CSV",
                           data=csv_buf.getvalue().encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

def page_live(det):
    st.subheader("🛰️ Live Capture - Online Mode")
    st.caption("Real-time network traffic capture and analysis. Use only on networks you own or have permission to monitor.")
    
    # Optional behavioural LSTM (per-user sequences) - load first
    beh_model = None
    artifact_dir = st.session_state.get(
        "_artifact_dir", os.path.dirname(os.path.abspath(__file__))
    )
    beh_model_status = "❌ Not available"
    try:
        beh_model = load_behavioral_lstm(artifact_dir)
        beh_model_status = f"✅ Loaded (T={beh_model.timesteps}, D={beh_model.features})"
    except FileNotFoundError:
        beh_model_status = "⚠️ beh_lstm.h5 not found (using flow-only mode)"
    except Exception as e:
        error_msg = str(e)
        if "NotEqual" in error_msg or "Unknown layer" in error_msg:
            beh_model_status = "❌ Compatibility error (retrain needed)"
        else:
            beh_model_status = f"❌ Error: {error_msg[:50]}"
    
    # Capture method selection
    capture_method = st.radio(
        "Capture Method",
        ["Wireshark (tshark)", "Scapy (Python)"],
        horizontal=True,
        help="Wireshark/tshark is more reliable and doesn't require special permissions. Scapy requires root/admin access."
    )
    
    feat = None
    wireshark_capture = None
    interfaces = []
    
    if capture_method == "Wireshark (tshark)":
        try:
            from wireshark_live import get_wireshark_capture, check_tshark_available, get_available_interfaces
            
            if not check_tshark_available():
                st.error("❌ tshark (Wireshark command-line tool) is not available.")
                st.info("""
                **To install Wireshark/tshark:**
                - **macOS**: `brew install wireshark`
                - **Linux**: `sudo apt-get install tshark` or `sudo yum install wireshark`
                - **Windows**: Download from https://www.wireshark.org/download.html
                
                After installation, make sure `tshark` is in your PATH.
                """)
                # Try to fall back to Scapy
                capture_method = "Scapy (Python)"
            else:
                # Get available interfaces
                interfaces = get_available_interfaces()
                if interfaces:
                    st.info(f"✅ tshark available. Found {len(interfaces)} interface(s).")
                else:
                    st.warning("⚠️ Could not detect network interfaces. Using default.")
                
                wireshark_capture = get_wireshark_capture()
                feat = None  # Use wireshark_capture instead
            
        except Exception as e:
            st.error(f"Wireshark capture not available: {e}")
            st.info("Falling back to Scapy method...")
            capture_method = "Scapy (Python)"
    
    if capture_method == "Scapy (Python)" or (wireshark_capture is None and feat is None):
        try:
            from scapy_featurizer import get_featurizer
            feat = get_featurizer()
            if wireshark_capture is None:
                wireshark_capture = None  # Explicitly set
        except Exception as e:
            st.error(f"Scapy featurizer not available: {e}")
            if wireshark_capture is None:
                st.warning("⚠️ No capture method available. Please install Wireshark/tshark or Scapy.")
                return

    # Optional behavioural LSTM (per-user sequences)
    beh_model = None
    artifact_dir = st.session_state.get(
        "_artifact_dir", os.path.dirname(os.path.abspath(__file__))
    )
    beh_model_status = "❌ Not available"
    try:
        beh_model = load_behavioral_lstm(artifact_dir)
        beh_model_status = f"✅ Loaded (T={beh_model.timesteps}, D={beh_model.features})"
    except FileNotFoundError:
        beh_model_status = "⚠️ beh_lstm.h5 not found (using flow-only mode)"
    except Exception as e:
        error_msg = str(e)
        if "NotEqual" in error_msg or "Unknown layer" in error_msg:
            beh_model_status = "❌ Compatibility error (retrain needed)"
        else:
            beh_model_status = f"❌ Error: {error_msg[:50]}"
    
    # Model status display
    with st.expander("📊 Model Status", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mlp_status = "✅ Loaded" if det.models.get("mlp") is not None else "❌ Not found"
            st.metric("MLP Model", mlp_status)
        with col2:
            lstm_status = "✅ Loaded" if det.models.get("lstm") is not None else "⚠️ Optional"
            st.metric("LSTM Model", lstm_status)
        with col3:
            st.metric("Behavioral LSTM", beh_model_status.split("(")[0] if "(" in beh_model_status else beh_model_status)
        with col4:
            mode = "Flow + Behavioral" if beh_model is not None else "Flow only"
            st.metric("Detection Mode", mode)

    now_ts = time.time()
    if "_live_capture_start" not in st.session_state:
        st.session_state["_live_capture_start"] = None
    if "_last_live_reload" not in st.session_state:
        st.session_state["_last_live_reload"] = now_ts
    if "_live_max_p_history" not in st.session_state:
        st.session_state["_live_max_p_history"] = []
    
    # Continuous learning state
    if "_continuous_learning_enabled" not in st.session_state:
        st.session_state["_continuous_learning_enabled"] = False
    if "_data_collection_start" not in st.session_state:
        st.session_state["_data_collection_start"] = None
    if "_collected_data" not in st.session_state:
        st.session_state["_collected_data"] = pd.DataFrame()
    if "_collection_duration_min" not in st.session_state:
        st.session_state["_collection_duration_min"] = 10

    # Risk combination weight (alpha)
    if beh_model is not None:
        alpha = st.slider(
            "Risk Combination Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="α=0.0: use only behavioural risk | α=1.0: use only flow risk | α=0.5: equal weight"
        )
        st.session_state["_risk_alpha"] = alpha
    else:
        alpha = 1.0  # Flow only
        st.info("ℹ️ Behavioral LSTM not available. Using flow risk only.")
        
        # Direct Training from Live Capture Section
        with st.expander("🎓 Train Behavioral LSTM from Live Capture", expanded=True):
            st.markdown("""
            **Train behavioral LSTM directly from live captured traffic:**
            1. Start live capture above
            2. Collect data for a specified duration
            3. Automatically build sequences and train model
            4. Model is ready to use immediately
            """)
            
            col_train1, col_train2 = st.columns(2)
            with col_train1:
                enable_direct_train = st.checkbox(
                    "Enable Direct Training from Live Capture",
                    value=st.session_state.get("_direct_training_enabled", False),
                    key="enable_direct_training"
                )
                st.session_state["_direct_training_enabled"] = enable_direct_train
                
                train_collection_duration = st.number_input(
                    "Collection Duration (minutes)",
                    min_value=1,
                    max_value=60,
                    value=10,
                    step=1,
                    key="direct_train_duration"
                )
            with col_train2:
                train_seq_len = st.number_input(
                    "Sequence Length",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                    key="direct_train_seq_len"
                )
                train_epochs = st.number_input(
                    "Training Epochs",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=1,
                    key="direct_train_epochs"
                )
            
            if enable_direct_train:
                direct_train_start = st.session_state.get("_direct_train_start", None)
                direct_train_data = st.session_state.get("_direct_train_data", pd.DataFrame())
                
                # Check if capture is running
                is_capturing = False
                capture_method_state = st.session_state.get("_capture_method", "scapy")
                if capture_method_state == "wireshark" and wireshark_capture:
                    is_capturing = wireshark_capture.is_capturing
                elif feat:
                    is_capturing = feat.is_capturing
                
                if not is_capturing:
                    st.warning("⚠️ Start live capture first to collect training data.")
                elif direct_train_start is None:
                    if st.button("🚀 Start Data Collection for Training", type="primary", key="start_direct_train"):
                        st.session_state["_direct_train_start"] = time.time()
                        st.session_state["_direct_train_data"] = pd.DataFrame()
                        st.success("Data collection started! Training will begin automatically when duration is reached.")
                        st.rerun()
                else:
                    elapsed_min = (time.time() - direct_train_start) / 60
                    remaining_min = train_collection_duration - elapsed_min
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Elapsed Time", f"{elapsed_min:.1f} min")
                    with col_stat2:
                        st.metric("Remaining", f"{max(0, remaining_min):.1f} min")
                    with col_stat3:
                        st.metric("Flows Collected", len(direct_train_data))
                    
                    if remaining_min <= 0:
                        st.info("⏰ Collection complete! Ready to train.")
                        if st.button("🎓 Train Model Now", type="primary", key="train_direct_now"):
                            with st.status("Training behavioral LSTM from live data...", expanded=True) as train_status:
                                try:
                                    from continuous_learning import build_sequences_from_live_data, train_behavioral_lstm_from_data
                                    
                                    if len(direct_train_data) < 50:
                                        raise ValueError(f"Not enough data collected ({len(direct_train_data)} flows). Need at least 50 flows.")
                                    
                                    train_status.update(label="Building sequences from collected data...", state="running")
                                    
                                    # Build sequences
                                    X_seq, y_seq = build_sequences_from_live_data(
                                        df=direct_train_data,
                                        user_id_col="user_id" if "user_id" in direct_train_data.columns else "src_ip",
                                        seq_len=train_seq_len,
                                        min_flows_per_user=5,
                                    )
                                    
                                    train_status.update(label=f"Built {len(X_seq)} sequences. Training model...", state="running")
                                    
                                    # Train model
                                    model_path = os.path.join(artifact_dir, "beh_lstm.h5")
                                    train_behavioral_lstm_from_data(
                                        X_seq=X_seq,
                                        y_seq=y_seq,
                                        model_path=model_path,
                                        epochs=train_epochs,
                                        batch_size=64,
                                    )
                                    
                                    train_status.update(label="Training completed!", state="complete")
                                    st.success(f"✅ Model trained successfully! {len(X_seq)} sequences, {train_epochs} epochs")
                                    
                                    # Check for threshold
                                    th_path = model_path.replace(".h5", ".threshold.json")
                                    if os.path.exists(th_path):
                                        import json
                                        with open(th_path, "r") as f:
                                            th_data = json.load(f)
                                        st.info(f"📊 Optimal Threshold: {th_data.get('threshold', 'N/A'):.3f} (F1={th_data.get('f1', 0):.4f})")
                                    
                                    # Clear collected data and reset
                                    st.session_state["_direct_train_data"] = pd.DataFrame()
                                    st.session_state["_direct_train_start"] = None
                                    
                                    # Reload detector
                                    st.info("🔄 Reloading detector...")
                                    try:
                                        det_new = load_detector(artifact_dir)
                                        st.session_state["_detector"] = det_new
                                        st.success("✅ Detector reloaded with new behavioral LSTM!")
                                        st.rerun()
                                    except Exception as e:
                                        st.warning(f"Model trained but detector reload failed: {e}. Click 'Load / Reload Detector' in sidebar.")
                                    
                                    # Auto-start next collection if enabled
                                    if enable_direct_train and is_capturing:
                                        st.session_state["_direct_train_start"] = time.time()
                                        st.info("🔄 Starting next collection cycle...")
                                        
                                except Exception as e:
                                    train_status.update(label="Training failed", state="error")
                                    st.error(f"❌ Error: {e}")
                    
                    if st.button("⏹ Stop Collection", key="stop_direct_train"):
                        st.session_state["_direct_train_start"] = None
                        st.session_state["_direct_train_data"] = pd.DataFrame()
                        st.info("Data collection stopped.")
        
        # Continuous Learning Section
        with st.expander("🔄 Continuous Learning (Auto-train from Live Data)", expanded=False):
            st.markdown("""
            **Automatically collect live data, train behavioral LSTM, and cycle continuously.**
            
            This feature will:
            1. Collect live captured data for a specified duration
            2. Build sequences from the collected data
            3. Train a behavioral LSTM model
            4. Delete the collected data
            5. Repeat the cycle automatically
            """)
            
            col_learn1, col_learn2 = st.columns(2)
            with col_learn1:
                enable_learning = st.checkbox(
                    "Enable Continuous Learning",
                    value=st.session_state.get("_continuous_learning_enabled", False),
                    key="enable_continuous_learning"
                )
                st.session_state["_continuous_learning_enabled"] = enable_learning
                
                collection_duration = st.number_input(
                    "Collection Duration (minutes)",
                    min_value=1,
                    max_value=60,
                    value=st.session_state.get("_collection_duration_min", 10),
                    step=1,
                    key="collection_duration"
                )
                st.session_state["_collection_duration_min"] = collection_duration
            
            with col_learn2:
                seq_len = st.number_input(
                    "Sequence Length (timesteps)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                    key="seq_len_learning"
                )
                training_epochs = st.number_input(
                    "Training Epochs",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                    key="training_epochs_learning"
                )
            
            if enable_learning:
                # Check if data collection is active
                collection_start = st.session_state.get("_data_collection_start")
                collected_data = st.session_state.get("_collected_data", pd.DataFrame())
                
                if collection_start is None:
                    if st.button("🚀 Start Data Collection", type="primary", key="start_collection"):
                        st.session_state["_data_collection_start"] = time.time()
                        st.session_state["_collected_data"] = pd.DataFrame()
                        st.success("Data collection started!")
                        st.rerun()
                else:
                    elapsed_min = (time.time() - collection_start) / 60
                    remaining_min = collection_duration - elapsed_min
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Elapsed Time", f"{elapsed_min:.1f} min")
                    with col_stat2:
                        st.metric("Remaining", f"{max(0, remaining_min):.1f} min")
                    with col_stat3:
                        st.metric("Flows Collected", len(collected_data))
                    
                    if remaining_min <= 0:
                        st.info("⏰ Collection period complete! Ready to train.")
                        if st.button("🎓 Train Model Now", type="primary", key="train_now"):
                            with st.status("Training behavioral LSTM...", expanded=True) as status:
                                try:
                                    from continuous_learning import collect_and_train_cycle
                                    
                                    status.update(label="Building sequences from collected data...", state="running")
                                    results = collect_and_train_cycle(
                                        collected_data=collected_data,
                                        artifact_dir=artifact_dir,
                                        collection_duration_min=collection_duration,
                                        seq_len=seq_len,
                                        epochs=training_epochs,
                                        batch_size=64,
                                        user_id_col="user_id" if "user_id" in collected_data.columns else "src_ip"
                                    )
                                    
                                    if results["success"]:
                                        status.update(label="Training completed successfully!", state="complete")
                                        st.success(f"✅ Model trained! {results['sequences_built']} sequences, {results['training_time']:.1f}s")
                                        
                                        # Clear collected data and reset collection
                                        st.session_state["_collected_data"] = pd.DataFrame()
                                        st.session_state["_data_collection_start"] = None
                                        
                                        # Reload detector to pick up new model
                                        st.info("🔄 Reloading detector to use new model...")
                                        try:
                                            det_new = load_detector(artifact_dir)
                                            st.session_state["_detector"] = det_new
                                            st.success("Detector reloaded with new behavioral LSTM!")
                                        except Exception as e:
                                            st.warning(f"Model trained but detector reload failed: {e}. Click 'Load / Reload Detector' in sidebar.")
                                        
                                        # Auto-start next collection cycle
                                        if enable_learning:
                                            st.session_state["_data_collection_start"] = time.time()
                                            st.info("🔄 Starting next collection cycle...")
                                    else:
                                        status.update(label="Training failed", state="error")
                                        st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                                except Exception as e:
                                    status.update(label="Error during training", state="error")
                                    st.error(f"❌ Error: {e}")
                    
                    if st.button("⏹ Stop Collection", key="stop_collection"):
                        st.session_state["_data_collection_start"] = None
                        st.session_state["_collected_data"] = pd.DataFrame()
                        st.info("Data collection stopped.")
        
        with st.expander("💡 Manual Training (Alternative)", expanded=False):
            st.markdown("""
            **To enable behavioral LSTM manually:**
            1. Build sequences: `python build_beh_sequences.py --csv KDDTest_plus.csv`
            2. Train model: `python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --out beh_lstm.h5`
            3. Ensure `beh_lstm.h5` is in your artifact directory
            4. Click "Load / Reload Detector" in the sidebar
            
            **Note:** Flow-only mode works perfectly for per-flow attack detection!
            """)
    
    col1, col2, col3 = st.columns(3)
    if capture_method == "Wireshark (tshark)":
        if wireshark_capture and interfaces:
            iface = col1.selectbox("Interface", options=[""] + interfaces, key="wireshark_iface")
        else:
            iface = col1.text_input("Interface (optional)", value="", placeholder="e.g., en0, eth0, wlan0", key="wireshark_iface")
        bpf = col2.text_input("BPF Filter (optional)", value="", placeholder="e.g., tcp or port 80", key="wireshark_bpf")
    else:
        iface = col1.text_input("Interface (optional)", value="", placeholder="e.g., eth0, wlan0, Ethernet", key="scapy_iface")
        bpf = col2.text_input("BPF Filter (optional)", value="", placeholder="e.g., tcp or port 80", key="scapy_bpf")
    
    interval = col3.number_input("Refresh (sec)", min_value=1, max_value=10, value=2, step=1)

    c1, c2 = st.columns(2)
    start = c1.button("▶️ Start Sniffing", type="primary")
    stop = c2.button("⏹ Stop Sniffing")
    
    if start:
        try:
            if capture_method == "Wireshark (tshark)" and wireshark_capture:
                wireshark_capture.start(interface=iface or None, bpf_filter=bpf or None)
                st.success("✅ Wireshark capture started.")
                st.session_state["_live_capture_start"] = time.time()
                st.session_state["_capture_method"] = "wireshark"
            else:
                feat.start(iface=iface or None, bpf_filter=bpf or None)
                st.success("✅ Scapy capture started.")
                st.session_state["_live_capture_start"] = time.time()
                st.session_state["_capture_method"] = "scapy"
                # Reset user sequences on new capture
                if hasattr(feat, "user_sequences"):
                    with feat.lock:
                        feat.user_sequences.clear()
        except PermissionError as e:
            error_msg = str(e)
            if "bpf" in error_msg.lower() or "permission denied" in error_msg.lower():
                st.error("❌ **Permission Denied: Network Access Required**")
                with st.expander("📱 How to Fix (macOS)", expanded=True):
                    st.markdown("""
                    **Option 1: Grant Network Permissions (Recommended) ⭐**
                    1. Open **System Settings** → **Privacy & Security** → **Network**
                    2. Click the lock icon and enter your password
                    3. Check the box next to **Terminal** (or your terminal app)
                    4. Restart Streamlit and try again
                    
                    **Option 2: Use Sudo**
                    ```bash
                    sudo env PATH="$PATH" streamlit run streamlit_app.py
                    ```
                    
                    **Option 3: Use Wireshark Method (No Permissions Needed)**
                    - Select **"Wireshark (tshark)"** in the capture method above
                    - Install Wireshark: `brew install wireshark`
                    - No sudo or permissions needed!
                    """)
            else:
                st.error(f"Permission error: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            if "bpf" in error_msg.lower() or "permission" in error_msg.lower():
                st.error("❌ **Permission Denied: Network Access Required**")
                with st.expander("📱 How to Fix (macOS)", expanded=True):
                    st.markdown("""
                    **Option 1: Grant Network Permissions (Recommended) ⭐**
                    1. Open **System Settings** → **Privacy & Security** → **Network**
                    2. Click the lock icon and enter your password
                    3. Check the box next to **Terminal** (or your terminal app)
                    4. Restart Streamlit and try again
                    
                    **Option 2: Use Sudo**
                    ```bash
                    sudo env PATH="$PATH" streamlit run streamlit_app.py
                    ```
                    
                    **Option 3: Use Wireshark Method (No Permissions Needed)**
                    - Select **"Wireshark (tshark)"** in the capture method above
                    - Install Wireshark: `brew install wireshark`
                    - No sudo or permissions needed!
                    """)
            else:
                st.error(f"Failed to start sniffing: {error_msg}")
    
    if stop:
        if capture_method == "Wireshark (tshark)" and wireshark_capture:
            wireshark_capture.stop()
        else:
            feat.stop()
        st.info("Sniffing stopped.")
        st.session_state["_capture_method"] = None

    # Auto-stop capture and refresh detector every 5 minutes for live mode
    live_start = st.session_state.get("_live_capture_start")
    capture_method_state = st.session_state.get("_capture_method", "scapy")
    
    is_capturing = False
    if capture_method_state == "wireshark" and wireshark_capture:
        is_capturing = wireshark_capture.is_capturing
    elif feat:
        is_capturing = feat.is_capturing
    
    if live_start and is_capturing and (now_ts - live_start) > 300:
        if capture_method_state == "wireshark" and wireshark_capture:
            wireshark_capture.stop()
        else:
            feat.stop()
        st.session_state["_live_capture_start"] = None
        st.info("Live capture stopped after 5 minutes. Refreshing detector and threshold.")
        art_dir = st.session_state.get("_artifact_dir", os.path.dirname(os.path.abspath(__file__)))
        try:
            det_new = load_detector(art_dir)
            st.session_state["_detector"] = det_new
            st.session_state["_last_live_reload"] = time.time()
            st.session_state["_live_max_p_history"] = []
            st.session_state["_adaptive_threshold_live"] = None
            det = det_new
        except Exception as e:
            st.warning(f"Could not refresh detector: {e}")

    req_cols = det.required_input_columns()

    # Single-pass render; use a button to refresh (no infinite loop that blocks Streamlit)
    colr1, colr2 = st.columns([1,1])
    do_refresh = colr1.button("🔄 Refresh now")
    colr2.caption(f"Auto-refresh by clicking; or rerun every ~{interval}s using your own scheduler.")

    # Get dataframe from appropriate capture method
    if capture_method_state == "wireshark" and wireshark_capture:
        df = wireshark_capture.dataframe()
    else:
        df = feat.dataframe() if feat else pd.DataFrame()
    if not df.empty:
        # Ensure required columns exist and order them
        for c in req_cols:
            if c not in df.columns:
                df[c] = 0
        live = df[req_cols].copy()
        
        # Get user sequence statistics
        user_seq_stats = {}
        if beh_model is not None:
            try:
                if capture_method_state == "wireshark" and wireshark_capture:
                    user_seqs = wireshark_capture.get_user_sequences(min_len=1)
                elif feat and hasattr(feat, "get_user_sequences"):
                    user_seqs = feat.get_user_sequences(min_len=1)
                else:
                    user_seqs = {}
                
                for uid, seq_arr in user_seqs.items():
                    user_seq_stats[uid] = {
                        "sequence_length": len(seq_arr),
                        "ready": len(seq_arr) >= getattr(beh_model, "timesteps", 5)
                    }
            except Exception:
                pass
        
        try:
            # Check if models are loaded before attempting prediction
            if det.models.get("mlp") is None:
                st.error("❌ **MLP model is required but not loaded!**")
                st.info("Please load models using 'Load / Reload Detector' in the sidebar.")
                return
            
            # Use model probabilities to derive an adaptive threshold for live traffic
            t0_inf = time.time()
            y_pred, y_proba = det.predict(live)
            inference_time = time.time() - t0_inf
            
            show = df.copy()
            show["prediction"] = y_pred
            
            # Collect data for continuous learning if enabled
            if st.session_state.get("_continuous_learning_enabled", False):
                collection_start = st.session_state.get("_data_collection_start")
                if collection_start is not None:
                    # Add timestamp and user_id if available
                    show_collect = show.copy()
                    show_collect["_collection_timestamp"] = time.time()
                    
                    # Use src_ip as user_id if user_id column doesn't exist
                    if "user_id" not in show_collect.columns and "src_ip" in show_collect.columns:
                        show_collect["user_id"] = show_collect["src_ip"].astype(str)
                    
                    # Append to collected data
                    collected = st.session_state.get("_collected_data", pd.DataFrame())
                    if collected.empty:
                        st.session_state["_collected_data"] = show_collect.copy()
                    else:
                        # Append new rows
                        st.session_state["_collected_data"] = pd.concat(
                            [collected, show_collect],
                            ignore_index=True
                        )
            
            # Collect data for direct training from live capture
            if st.session_state.get("_direct_training_enabled", False):
                direct_train_start = st.session_state.get("_direct_train_start")
                if direct_train_start is not None:
                    # Add timestamp and user_id if available
                    show_train = show.copy()
                    show_train["_collection_timestamp"] = time.time()
                    
                    # Use src_ip as user_id if user_id column doesn't exist
                    if "user_id" not in show_train.columns and "src_ip" in show_train.columns:
                        show_train["user_id"] = show_train["src_ip"].astype(str)
                    
                    # Append to training data
                    train_data = st.session_state.get("_direct_train_data", pd.DataFrame())
                    if train_data.empty:
                        st.session_state["_direct_train_data"] = show_train.copy()
                    else:
                        # Append new rows
                        st.session_state["_direct_train_data"] = pd.concat(
                            [train_data, show_train],
                            ignore_index=True
                        )

            flow_risk = None
            if y_proba is not None:
                max_p = np.max(y_proba, axis=1)
                flow_risk = max_p
                show["flow_risk"] = flow_risk

                hist = st.session_state.get("_live_max_p_history", [])
                hist.extend(max_p.tolist())
                # Keep a bounded history window
                hist = hist[-1000:]
                st.session_state["_live_max_p_history"] = hist

                base_thr = st.session_state.get("_threshold", 0.9)
                adaptive_thr = _compute_adaptive_threshold(hist, base_thr)
                st.session_state["_adaptive_threshold_live"] = adaptive_thr

                # Determine which labels are "risky" (attacks)
                if det.label_binarizer is not None and hasattr(det.label_binarizer, "classes_"):
                    classes = list(det.label_binarizer.classes_)
                    risky_set = {c for c in classes if str(c).lower() != "normal"}
                    labels = y_pred
                else:
                    risky_set = set(range(1, y_proba.shape[1]))
                    labels = y_pred

                # Optional behavioural risk per user (if model and user_id are available)
                beh_scores: Dict[str, float] = {}
                beh_inference_time = 0.0
                if beh_model is not None and "user_id" in show.columns:
                    try:
                        t0_beh = time.time()
                        if capture_method_state == "wireshark" and wireshark_capture:
                            user_seqs = wireshark_capture.get_user_sequences(min_len=getattr(beh_model, "timesteps", 5))
                        elif feat and hasattr(feat, "get_user_sequences"):
                            user_seqs = feat.get_user_sequences(min_len=getattr(beh_model, "timesteps", 5))
                        else:
                            user_seqs = {}
                        beh_scores = beh_model.score_users(user_seqs)
                        beh_inference_time = time.time() - t0_beh
                    except Exception as e:
                        beh_scores = {}
                        st.warning(f"Behavioral LSTM inference failed: {e}")

                # Combine risks
                combined_risk = []
                alpha = st.session_state.get("_risk_alpha", 0.5)
                for idx, (lbl, p) in enumerate(zip(labels, max_p)):
                    uid = show.iloc[idx].get("user_id") if "user_id" in show.columns else None
                    beh = beh_scores.get(str(uid), None) if uid is not None else None
                    if beh is not None:
                        r = alpha * float(p) + (1.0 - alpha) * float(beh)
                    else:
                        r = float(p)
                    combined_risk.append(r)

                combined_risk_arr = np.asarray(combined_risk, dtype=float)
                show["risk_combined"] = combined_risk_arr
                
                # Add behavioural risk column if available
                if beh_model is not None and "user_id" in show.columns:
                    beh_risk_col = []
                    for idx in range(len(show)):
                        uid = show.iloc[idx].get("user_id")
                        beh_risk_col.append(beh_scores.get(str(uid), np.nan) if uid is not None else np.nan)
                    show["behavioral_risk"] = beh_risk_col

                # Use adaptive threshold on combined risk to decide revoke
                revoke_adaptive = np.array([
                    1 if ((lbl in risky_set) and (r >= adaptive_thr)) else 0
                    for lbl, r in zip(labels, combined_risk_arr)
                ], dtype=int)
                show["revoke"] = revoke_adaptive
            else:
                # No probabilities available; default to no revoke
                show["revoke"] = 0
                show["flow_risk"] = 0.0
                show["risk_combined"] = 0.0
        except Exception as e:
            show = df.copy()
            show["prediction"] = f"ERR: {e}"
            show["revoke"] = 0
        
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Flows", len(df))
        with col2:
            if "revoke" in show.columns:
                revoked = int(show["revoke"].sum())
                st.metric("Revoked", revoked, delta=f"{100*revoked/len(show):.1f}%" if len(show) > 0 else "0%")
        with col3:
            if st.session_state.get("_adaptive_threshold_live") is not None:
                st.metric("Adaptive Threshold", f"{st.session_state['_adaptive_threshold_live']:.3f}")
        with col4:
            if "flow_risk" in show.columns and show["flow_risk"].notna().any():
                avg_flow_risk = show["flow_risk"].mean()
                st.metric("Avg Flow Risk", f"{avg_flow_risk:.3f}")
        with col5:
            if beh_model is not None and len(user_seq_stats) > 0:
                ready_users = sum(1 for s in user_seq_stats.values() if s["ready"])
                st.metric("Users Tracked", f"{len(user_seq_stats)} ({ready_users} ready)")
        
        # Timing information
        st.caption(
            f"Last update: {time.strftime('%H:%M:%S')} | "
            f"Inference: {inference_time*1000:.1f}ms" +
            (f" | Behavioral: {beh_inference_time*1000:.1f}ms" if beh_inference_time > 0 else "")
        )
        
        # Adaptive threshold info
        if st.session_state.get("_adaptive_threshold_live") is not None:
            st.info(
                f"📊 Live adaptive threshold: **{st.session_state['_adaptive_threshold_live']:.3f}** "
                f"(base: {st.session_state.get('_threshold', 0.9):.3f}, "
                f"α={alpha:.1f} for risk combination)"
            )
        
        # User sequence statistics (if behavioral model available)
        if beh_model is not None and len(user_seq_stats) > 0:
            with st.expander(f"👥 User Sequence Tracking ({len(user_seq_stats)} users)", expanded=False):
                user_df = pd.DataFrame([
                    {
                        "user_id": uid,
                        "sequence_length": stats["sequence_length"],
                        "ready": "✅" if stats["ready"] else "⏳",
                        "behavioral_risk": beh_scores.get(str(uid), np.nan) if str(uid) in beh_scores else np.nan
                    }
                    for uid, stats in user_seq_stats.items()
                ])
                st.dataframe(user_df, width='stretch')
        
        # Risk comparison visualization
        if "flow_risk" in show.columns and "risk_combined" in show.columns:
            with st.expander("📈 Risk Score Comparison", expanded=False):
                risk_comparison = pd.DataFrame({
                    "flow_risk": show["flow_risk"].head(50),
                    "risk_combined": show["risk_combined"].head(50)
                })
                st.line_chart(risk_comparison)
                if "behavioral_risk" in show.columns and show["behavioral_risk"].notna().any():
                    beh_risk_available = show[show["behavioral_risk"].notna()].head(50)
                    if len(beh_risk_available) > 0:
                        st.caption("Behavioral risk available for some users")
                        st.line_chart(beh_risk_available[["behavioral_risk", "risk_combined"]])
        
        # Main data display
        st.write("### Live Flow Data (last 50)")
        # Select columns to display
        display_cols = ["prediction", "flow_risk", "risk_combined", "revoke"]
        if "behavioral_risk" in show.columns:
            display_cols.insert(-1, "behavioral_risk")
        if "user_id" in show.columns:
            display_cols.insert(0, "user_id")
        
        # Add any other original columns user might want to see
        for col in ["src_ip", "dst_ip", "protocol_type", "service"]:
            if col in show.columns and col not in display_cols:
                display_cols.append(col)
        
        display_df = show[display_cols].tail(50) if all(c in show.columns for c in display_cols) else show.tail(50)
        st.dataframe(display_df, width='stretch')
        
        # Download option
        csv_buf = io.StringIO()
        show.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Live Data CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"live_capture_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📡 No flows captured yet.")
        with st.expander("🚀 How to start capturing", expanded=True):
            st.markdown("""
            **Steps to start live capture:**
            1. **Click "▶️ Start Sniffing"** button above
            2. **If you get permission errors**, run Streamlit with sudo:
               ```bash
               sudo env PATH="$PATH" streamlit run streamlit_app.py
               ```
            3. **Optional:** Specify network interface (e.g., `en0`, `eth0`, `wlan0`)
            4. **Optional:** Add BPF filter (e.g., `tcp`, `port 80`, `host 192.168.1.1`)
            5. **Click "🔄 Refresh now"** to see captured flows
            
            **Note:** 
            - On macOS, you may need to grant Terminal/Streamlit network permissions
            - Live capture auto-refreshes models every 5 minutes
            - User sequences build over time (requires behavioral LSTM)
            """)
        if beh_model is not None:
            st.success("✅ Behavioral LSTM is ready! User sequences will build as traffic is captured.")
        elif is_capturing:
            st.caption("💡 Capture is running. Waiting for network traffic...")

    if do_refresh:
        # Light delay to avoid rapid churn, then rerun
        time.sleep(interval)
        st.rerun()

def page_training():
    st.subheader("🎓 Model Training")
    st.caption("Train MLP, LSTM, or Behavioral LSTM models from your data. All training happens in the background.")
    
    # Check for required dependencies
    with st.expander("🔧 System Requirements & Setup", expanded=False):
        st.markdown("""
        **Required Dependencies:**
        - Python packages: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `scipy`
        - Training data files (KDD format)
        
        **Installing Dependencies:**
        ```bash
        # If you need to install system packages (may require sudo):
        # macOS:
        brew install python3  # if needed
        
        # Linux:
        sudo apt-get update
        sudo apt-get install python3-pip python3-venv
        
        # Install Python packages (usually no sudo needed):
        pip install -r requirements_streamlit.txt
        
        # If pip install fails, try:
        pip install --user -r requirements_streamlit.txt
        ```
        
        **Note:** Training models does NOT require sudo. Only live network capture requires elevated privileges.
        """)
    
    tab_mlp, tab_lstm, tab_beh_lstm = st.tabs(["MLP Training", "LSTM Training", "Behavioral LSTM Training"])
    
    with tab_mlp:
        st.markdown("### Train MLP Model")
        st.caption("Train a Multi-Layer Perceptron for attack detection")
        
        col1, col2 = st.columns(2)
        with col1:
            train_file = st.text_input("Training Data", value="archive/KDDTrain+_20Percent.txt", key="mlp_train")
            val_file = st.text_input("Validation Data", value="archive/KDDTest+.txt", key="mlp_val")
            artifact_dir = st.text_input("Artifact Directory", value=os.path.dirname(os.path.abspath(__file__)), key="mlp_art")
        with col2:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, key="mlp_epochs")
            batch_size = st.number_input("Batch Size", min_value=32, max_value=1024, value=256, step=32, key="mlp_batch")
            units = st.number_input("Hidden Units", min_value=64, max_value=1024, value=256, step=64, key="mlp_units")
            dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.3, step=0.1, key="mlp_dropout")
        
        use_tune = st.checkbox("Enable Hyperparameter Tuning", value=False, key="mlp_tune")
        output_file = st.text_input("Output Model File", value="att_det_mlp.h5", key="mlp_out")
        
        if st.button("🚀 Start MLP Training", type="primary", key="mlp_train_btn"):
            if not os.path.exists(train_file):
                st.error(f"Training file not found: {train_file}")
            elif not os.path.exists(val_file):
                st.error(f"Validation file not found: {val_file}")
            else:
                with st.status("Training MLP model...", expanded=True) as status:
                    try:
                        cmd = [
                            sys.executable, "train_mlp.py",
                            "--train", train_file,
                            "--val", val_file,
                            "--artifact_dir", artifact_dir,
                            "--epochs", str(epochs),
                            "--batch_size", str(batch_size),
                            "--units", str(units),
                            "--dropout", str(dropout),
                            "--output", output_file
                        ]
                        if use_tune:
                            cmd.append("--tune")
                        
                        status.update(label="Running training script...", state="running")
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                        
                        if result.returncode == 0:
                            status.update(label="Training completed successfully!", state="complete")
                            st.success("✅ MLP model trained successfully!")
                            st.code(result.stdout)
                            if os.path.exists(output_file):
                                st.info(f"Model saved to: {output_file}")
                        else:
                            status.update(label="Training failed", state="error")
                            st.error("❌ Training failed. Check error messages below.")
                            st.code(result.stderr)
                            st.code(result.stdout)
                    except Exception as e:
                        st.error(f"Failed to start training: {e}")
    
    with tab_lstm:
        st.markdown("### Train LSTM Model")
        st.caption("Train a Long Short-Term Memory network for sequence-based attack detection")
        
        col1, col2 = st.columns(2)
        with col1:
            train_file = st.text_input("Training Data", value="archive/KDDTrain+_20Percent.txt", key="lstm_train")
            val_file = st.text_input("Validation Data", value="archive/KDDTest+.txt", key="lstm_val")
            artifact_dir = st.text_input("Artifact Directory", value=os.path.dirname(os.path.abspath(__file__)), key="lstm_art")
        with col2:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=8, key="lstm_epochs")
            batch_size = st.number_input("Batch Size", min_value=32, max_value=1024, value=256, step=32, key="lstm_batch")
            lstm_units = st.number_input("LSTM Units", min_value=32, max_value=256, value=64, step=32, key="lstm_units")
        
        output_file = st.text_input("Output Model File", value="att_det_lstm.h5", key="lstm_out")
        
        if st.button("🚀 Start LSTM Training", type="primary", key="lstm_train_btn"):
            if not os.path.exists(train_file):
                st.error(f"Training file not found: {train_file}")
            elif not os.path.exists(val_file):
                st.error(f"Validation file not found: {val_file}")
            else:
                with st.status("Training LSTM model...", expanded=True) as status:
                    try:
                        cmd = [
                            sys.executable, "train_lstm.py",
                            "--train", train_file,
                            "--val", val_file,
                            "--artifact_dir", artifact_dir,
                            "--epochs", str(epochs),
                            "--batch_size", str(batch_size),
                            "--lstm_units", str(lstm_units),
                            "--output", output_file
                        ]
                        
                        status.update(label="Running training script...", state="running")
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                        
                        if result.returncode == 0:
                            status.update(label="Training completed successfully!", state="complete")
                            st.success("✅ LSTM model trained successfully!")
                            st.code(result.stdout)
                            if os.path.exists(output_file):
                                st.info(f"Model saved to: {output_file}")
                                # Check for threshold file
                                th_file = output_file.replace(".h5", ".threshold.json")
                                if os.path.exists(th_file):
                                    st.info(f"Threshold file saved to: {th_file}")
                        else:
                            status.update(label="Training failed", state="error")
                            st.error("❌ Training failed. Check error messages below.")
                            st.code(result.stderr)
                            st.code(result.stdout)
                    except Exception as e:
                        st.error(f"Failed to start training: {e}")
    
    with tab_beh_lstm:
        st.markdown("### Train Behavioral LSTM Model")
        st.caption("Train a behavioral LSTM model for per-user sequence analysis")
        
        # Wireshark PCAP Processing Option
        with st.expander("📡 Process Wireshark PCAP File (Alternative to Live Capture)", expanded=True):
            st.markdown("""
            **Use Wireshark to capture network traffic, then process it here:**
            1. Capture traffic in Wireshark and save as `.pcap` file
            2. Upload the PCAP file below
            3. Convert to KDD features automatically
            4. Build sequences and train behavioral LSTM
            """)
            
            pcap_file = st.file_uploader(
                "Upload Wireshark PCAP File",
                type=["pcap", "pcapng"],
                key="pcap_upload"
            )
            
            if pcap_file is not None:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp_file:
                    tmp_file.write(pcap_file.read())
                    tmp_pcap_path = tmp_file.name
                
                if st.button("🔄 Convert PCAP to KDD CSV", key="convert_pcap"):
                    with st.status("Converting PCAP to KDD features...", expanded=True) as status:
                        try:
                            from wireshark_to_kdd import pcap_to_kdd_csv
                            import tempfile
                            
                            # Create output CSV path
                            output_csv = f"wireshark_capture_{int(time.time())}.csv"
                            
                            status.update(label="Processing PCAP file...", state="running")
                            csv_path = pcap_to_kdd_csv(
                                pcap_path=tmp_pcap_path,
                                output_csv=output_csv,
                                time_window=2.0,
                                min_packets_per_flow=1
                            )
                            
                            status.update(label="Conversion complete!", state="complete")
                            st.success(f"✅ Converted PCAP to CSV: {csv_path}")
                            st.session_state["_wireshark_csv"] = csv_path
                            
                            # Show preview
                            df_preview = pd.read_csv(csv_path)
                            st.dataframe(df_preview.head(20), width='stretch')
                            st.info(f"Total flows: {len(df_preview)}")
                            
                            # Option to run complete workflow
                            st.markdown("---")
                            st.markdown("### 🚀 Complete Workflow")
                            st.caption("Automatically build sequences and train behavioral LSTM from this CSV")
                            
                            col_wf1, col_wf2 = st.columns(2)
                            with col_wf1:
                                wf_seq_len = st.number_input("Sequence Length", min_value=5, max_value=50, value=10, key="wf_seq_len")
                            with col_wf2:
                                wf_epochs = st.number_input("Training Epochs", min_value=5, max_value=50, value=20, key="wf_epochs")
                            
                            if st.button("🎓 Run Complete Workflow (Build Sequences + Train)", type="primary", key="run_complete_workflow"):
                                with st.status("Running complete workflow...", expanded=True) as wf_status:
                                    try:
                                        # Step 1: Build sequences
                                        wf_status.update(label="Building sequences...", state="running")
                                        seq_x = f"X_seq_{int(time.time())}.npy"
                                        seq_y = f"y_seq_{int(time.time())}.npy"
                                        
                                        cmd_seq = [
                                            sys.executable, "build_beh_sequences.py",
                                            "--csv", csv_path,
                                            "--out_x", seq_x,
                                            "--out_y", seq_y,
                                            "--seq_len", str(wf_seq_len)
                                        ]
                                        
                                        result_seq = subprocess.run(
                                            cmd_seq,
                                            capture_output=True,
                                            text=True,
                                            cwd=os.path.dirname(os.path.abspath(__file__))
                                        )
                                        
                                        if result_seq.returncode != 0:
                                            raise RuntimeError(f"Sequence building failed: {result_seq.stderr}")
                                        
                                        # Step 2: Train model
                                        wf_status.update(label="Training behavioral LSTM...", state="running")
                                        model_path = os.path.join(artifact_dir, "beh_lstm.h5")
                                        
                                        cmd_train = [
                                            sys.executable, "train_beh_lstm.py",
                                            "--x", seq_x,
                                            "--y", seq_y,
                                            "--epochs", str(wf_epochs),
                                            "--batch_size", "64",
                                            "--out", model_path
                                        ]
                                        
                                        result_train = subprocess.run(
                                            cmd_train,
                                            capture_output=True,
                                            text=True,
                                            cwd=os.path.dirname(os.path.abspath(__file__))
                                        )
                                        
                                        if result_train.returncode != 0:
                                            raise RuntimeError(f"Training failed: {result_train.stderr}")
                                        
                                        wf_status.update(label="Workflow completed!", state="complete")
                                        st.success("✅ Behavioral LSTM trained successfully!")
                                        st.code(result_train.stdout)
                                        
                                        # Check for threshold file
                                        th_path = model_path.replace(".h5", ".threshold.json")
                                        if os.path.exists(th_path):
                                            import json
                                            with open(th_path, "r") as f:
                                                th_data = json.load(f)
                                            st.info(f"📊 Optimal Threshold: {th_data.get('threshold', 'N/A'):.3f} (F1={th_data.get('f1', 0):.4f})")
                                        
                                        # Clean up temp sequence files
                                        for f in [seq_x, seq_y]:
                                            if os.path.exists(f):
                                                try:
                                                    os.unlink(f)
                                                except:
                                                    pass
                                        
                                        st.info("🔄 Click 'Load / Reload Detector' in sidebar to use the new model")
                                        
                                    except Exception as e:
                                        wf_status.update(label="Workflow failed", state="error")
                                        st.error(f"❌ Error: {e}")
                            
                        except Exception as e:
                            status.update(label="Conversion failed", state="error")
                            st.error(f"❌ Error: {e}")
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(tmp_pcap_path)
                            except:
                                pass
        
        st.info("""
        **Prerequisites:**
        1. First, build sequences from your data using `build_beh_sequences.py`
        2. This will create `X_seq.npy` and `y_seq.npy` files
        3. Then train the behavioral LSTM model using those files
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            x_seq_file = st.text_input("Sequence Features (X_seq.npy)", value="X_seq.npy", key="beh_x")
            y_seq_file = st.text_input("Sequence Labels (y_seq.npy)", value="y_seq.npy", key="beh_y")
        with col2:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20, key="beh_epochs")
            batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16, key="beh_batch")
        
        output_file = st.text_input("Output Model File", value="beh_lstm.h5", key="beh_out")
        
        # Option to build sequences first
        with st.expander("🔨 Build Sequences First", expanded=True):
            st.markdown("If you haven't created sequence files yet, build them here:")
            
            # Use Wireshark CSV if available
            wireshark_csv = st.session_state.get("_wireshark_csv")
            if wireshark_csv and os.path.exists(wireshark_csv):
                default_csv = wireshark_csv
                st.success(f"📡 Using Wireshark CSV: {wireshark_csv}")
            else:
                default_csv = "KDDTest_plus.csv"
            
            seq_csv = st.text_input("Input CSV File", value=default_csv, key="seq_csv")
            seq_timesteps = st.number_input("Sequence Length (Timesteps)", min_value=5, max_value=50, value=10, key="seq_t")
            
            if st.button("🔨 Build Sequences", key="build_seq_btn"):
                if not os.path.exists(seq_csv):
                    st.error(f"CSV file not found: {seq_csv}")
                else:
                    with st.status("Building sequences...", expanded=True) as status:
                        try:
                            cmd = [
                                sys.executable, "build_beh_sequences.py",
                                "--csv", seq_csv,
                                "--timesteps", str(seq_timesteps)
                            ]
                            
                            status.update(label="Running sequence builder...", state="running")
                            result = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))
                            )
                            
                            if result.returncode == 0:
                                status.update(label="Sequences built successfully!", state="complete")
                                st.success("✅ Sequences built successfully!")
                                st.code(result.stdout)
                                if os.path.exists("X_seq.npy") and os.path.exists("y_seq.npy"):
                                    st.info("Files created: X_seq.npy, y_seq.npy")
                            else:
                                status.update(label="Sequence building failed", state="error")
                                st.error("❌ Failed to build sequences. Check error messages below.")
                                st.code(result.stderr)
                                st.code(result.stdout)
                        except Exception as e:
                            st.error(f"Failed to build sequences: {e}")
        
        if st.button("🚀 Start Behavioral LSTM Training", type="primary", key="beh_train_btn"):
            if not os.path.exists(x_seq_file):
                st.error(f"Sequence features file not found: {x_seq_file}")
            elif not os.path.exists(y_seq_file):
                st.error(f"Sequence labels file not found: {y_seq_file}")
            else:
                with st.status("Training Behavioral LSTM model...", expanded=True) as status:
                    try:
                        cmd = [
                            sys.executable, "train_beh_lstm.py",
                            "--x", x_seq_file,
                            "--y", y_seq_file,
                            "--epochs", str(epochs),
                            "--batch_size", str(batch_size),
                            "--out", output_file
                        ]
                        
                        status.update(label="Running training script...", state="running")
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                        
                        if result.returncode == 0:
                            status.update(label="Training completed successfully!", state="complete")
                            st.success("✅ Behavioral LSTM model trained successfully!")
                            st.code(result.stdout)
                            if os.path.exists(output_file):
                                st.info(f"Model saved to: {output_file}")
                        else:
                            status.update(label="Training failed", state="error")
                            st.error("❌ Training failed. Check error messages below.")
                            st.code(result.stderr)
                            st.code(result.stdout)
                    except Exception as e:
                        st.error(f"Failed to start training: {e}")

def main():
    st.title("🔐 Attack Detection – Ensemble Inference (SciPy)")
    with st.sidebar:
        st.header("⚙️ Settings")
        # Default artifact dir: folder containing this script (where artifacts live)
        default_artifact_dir = os.path.dirname(os.path.abspath(__file__))
        artifact_dir = st.text_input("Artifact directory", value=default_artifact_dir)
        # Expose artifact dir to other parts of the app (e.g., live behavioural LSTM)
        st.session_state["_artifact_dir"] = artifact_dir
        # Default threshold: learned from LSTM if available, else 0.90
        det_for_default = st.session_state.get("_detector")
        learned_t = None
        if det_for_default is not None and hasattr(det_for_default, "learned_threshold") and det_for_default.learned_threshold:
            learned_t = float(det_for_default.learned_threshold)
        threshold_default = learned_t if learned_t is not None else 0.9
        threshold = st.slider("Revoke threshold", min_value=0.5, max_value=0.99, value=float(threshold_default), step=0.01)
        st.session_state["_threshold"] = float(threshold)
        if st.button("Load / Reload Detector", type="primary"):
            st.session_state["_reload"] = True
            st.rerun()
        st.markdown("---")
        st.caption("Expected artifacts:")
        st.code("\n".join([
            "att_det_mlp.h5 (required)",
            "att_det_lstm.h5 (optional)",
            "beh_lstm.h5 (optional, for live behavioral detection)",
            "onehot_encoder.pkl",
            "label_binarizer.pkl",
        ]))

    if st.session_state.get("_reload") or "_detector" not in st.session_state:
        with st.status("Loading detector...", expanded=True):
            try:
                # Resolve artifact directory to absolute path
                abs_artifact_dir = os.path.abspath(os.path.expanduser(artifact_dir))
                
                # Check if directory exists
                if not os.path.exists(abs_artifact_dir):
                    st.error(f"❌ Artifact directory does not exist: {abs_artifact_dir}")
                    st.info(f"Current working directory: {os.getcwd()}")
                    st.info(f"Please check the artifact directory path in the sidebar.")
                    return
                
                # Check for required model files
                required_files = {
                    "MLP Model": os.path.join(abs_artifact_dir, "att_det_mlp.h5"),
                    "OneHot Encoder": os.path.join(abs_artifact_dir, "onehot_encoder.pkl"),
                    "Label Binarizer": os.path.join(abs_artifact_dir, "label_binarizer.pkl")
                }
                
                missing_files = []
                for name, path in required_files.items():
                    if not os.path.exists(path):
                        missing_files.append(f"{name}: {path}")
                
                if missing_files:
                    st.warning("⚠️ Some required files are missing:")
                    for missing in missing_files:
                        st.text(missing)
                    st.info("💡 Go to the **Training** tab to train missing models.")
                
                # Load detector
                det = load_detector(abs_artifact_dir)
                st.session_state["_detector"] = det
                st.session_state["_reload"] = False  # Clear reload flag
                
                # Verify models loaded successfully
                mlp_loaded = det.models.get("mlp") is not None
                lstm_loaded = det.models.get("lstm") is not None
                
                if mlp_loaded:
                    st.success("✅ MLP model loaded successfully")
                else:
                    st.error("❌ MLP model failed to load")
                    st.info(f"Expected path: {os.path.join(abs_artifact_dir, 'att_det_mlp.h5')}")
                
                if lstm_loaded:
                    st.success("✅ LSTM model loaded successfully")
                else:
                    st.info("ℹ️ LSTM model not loaded (optional)")
                
                if not mlp_loaded:
                    with st.expander("🔧 Troubleshooting Model Loading", expanded=True):
                        st.markdown("""
                        **If models fail to load, try these solutions:**
                        
                        1. **Check TensorFlow installation**:
                           ```bash
                           python -c "import tensorflow as tf; print(tf.__version__)"
                           ```
                        
                        2. **Retrain the models** (recommended):
                           - Go to **Training** tab
                           - Retrain MLP and LSTM models
                           - This will create models compatible with your TensorFlow version
                        
                        3. **Update TensorFlow**:
                           ```bash
                           pip install --upgrade tensorflow
                           ```
                        
                        4. **Check file permissions**:
                           Ensure model files are readable
                        """)
                
                # Check for behavioral LSTM
                try:
                    beh_check = load_behavioral_lstm(artifact_dir)
                    st.success(f"Behavioral LSTM found (T={beh_check.timesteps}).")
                except FileNotFoundError:
                    st.info("Behavioral LSTM (beh_lstm.h5) not found. Live mode will use flow-only detection.")
                except Exception as e:
                    error_msg = str(e)
                    if "NotEqual" in error_msg or "Unknown layer" in error_msg:
                        st.warning("⚠️ Behavioral LSTM model compatibility issue detected.")
                        with st.expander("🔧 How to fix this", expanded=True):
                            st.markdown("""
                            **The model was saved with a different TensorFlow version.**
                            
                            **Solution:** Retrain the behavioral LSTM model:
                            1. Go to the **Training** tab → **Behavioral LSTM Training**
                            2. Build sequences (if needed): Click "🔨 Build Sequences"
                            3. Train the model: Click "🚀 Start Behavioral LSTM Training"
                            
                            Or from command line:
                            ```bash
                            python train_beh_lstm.py --x X_seq.npy --y y_seq.npy --out beh_lstm.h5
                            ```
                            """)
                    else:
                        st.warning(f"Behavioral LSTM check failed: {error_msg[:100]}")
            except Exception as e:
                st.error(f"Failed to initialize detector: {e}")
                return

    det = st.session_state["_detector"]
    
    # Model status summary in sidebar
    with st.expander("📊 Model Status Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Flow Models:**")
            st.write(f"- MLP: {'✅' if det.models.get('mlp') is not None else '❌'}")
            st.write(f"- LSTM: {'✅' if det.models.get('lstm') is not None else '⚠️'}")
        with col2:
            st.write("**Behavioral Model:**")
            try:
                beh_check = load_behavioral_lstm(artifact_dir)
                st.write(f"- Behavioral LSTM: ✅ (T={beh_check.timesteps})")
            except:
                st.write("- Behavioral LSTM: ❌")
        st.write("**Preprocessing:**")
        st.write(f"- OneHot Encoder: {'✅' if det.onehot is not None else '⚠️'}")
        st.write(f"- Label Binarizer: {'✅' if det.label_binarizer is not None else '⚠️'}")
    
    with st.expander("Required Input Columns (first 41 KDD features)"):
        req_cols = det.required_input_columns()
        st.write(pd.DataFrame({"#": list(range(1, len(req_cols)+1)), "column": req_cols}))

    tab_batch, tab_live, tab_training = st.tabs(["📦 Batch CSV", "🛰️ Live Capture (Scapy)", "🎓 Training"])
    with tab_batch:
        page_batch(det)
    with tab_live:
        page_live(det)
    with tab_training:
        page_training()

if __name__ == "__main__":
    main()
