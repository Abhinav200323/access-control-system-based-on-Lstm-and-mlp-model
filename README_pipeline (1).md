
# Attack Detection Pipeline (SciPy + Ensemble)

This single-file pipeline integrates your artifacts:

- `att_det_knn.pkl`, `att_det_svm.pkl`, optional `att_det_qsvm.pkl`
- `att_det_mlp.h5`
- `onehot_encoder.pkl`, `label_binarizer.pkl`

**SciPy usage:** `scipy.stats.zscore` for standardization and `scipy.sparse` for efficient concatenation.

## Install (example)
pip install numpy scipy scikit-learn pandas joblib tensorflow==2.*

## Run
python attack_detection_pipeline.py --artifact_dir . --csv your.csv --prefer_proba
