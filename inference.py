# --- utils_load_and_split.py (put in a cell) -------------------------------
import os, re, numpy as np, pandas as pd

def _basename_no_ext(p):
    b = os.path.basename(str(p))
    return re.sub(r"\.[A-Za-z0-9]+$", "", b)

def _load_npz_as_table(npz_path, source_tag):
    """
    Loads an NPZ where each KEY is an image path and each VALUE is a 1D embedding vector.
    Returns:
      df_npz: columns [source, img_key, emb_index, image_id, X_block, X_ptr]
      X: np.ndarray of shape [N, D]
    """
    npz = np.load(npz_path, allow_pickle=True)
    ids = list(npz.keys())
    X = np.vstack([npz[k] for k in ids])
    df_npz = pd.DataFrame({
        "source": source_tag,
        "img_key": [_basename_no_ext(k).lower() for k in ids],
        "emb_index": np.arange(len(ids), dtype=int),
        "image_id": [_basename_no_ext(k) for k in ids],
        "X_block": source_tag,
        "X_ptr": np.arange(len(ids), dtype=int)
    })
    return df_npz, X

def _attach_labels(df_npz, labels_csv, id_col_name, LABELS):
    """
    Merge NPZ table with label CSV on normalized image id.
    - id_col_name: the column in the CSV that holds image id (e.g., 'img_id' for PAD, 'image_id' for derm12345).
    Returns merged dataframe filtered to rows that have ANY of the LABELS.
    """
    lab = pd.read_csv(labels_csv)
    if id_col_name not in lab.columns:
        raise ValueError(f"'{id_col_name}' not in {labels_csv}. Available: {list(lab.columns)}")
    lab["img_key"] = lab[id_col_name].astype(str).map(lambda s: _basename_no_ext(s).lower())

    need_cols = ["patient_id","img_key"] + LABELS
    miss = [c for c in need_cols if c not in lab.columns]
    if miss:
        raise ValueError(f"Missing columns in {labels_csv}: {miss}")

    merged = df_npz.merge(lab[need_cols], on="img_key", how="left")
    keep_mask = merged[LABELS].notna().any(axis=1)
    merged = merged[keep_mask].reset_index(drop=True)
    # force ints for labels
    merged[LABELS] = merged[LABELS].fillna(0).astype(int)
    return merged

def _pull_X_from_blocks(df_all, X_blocks):
    """Row-wise gather embeddings from the correct block/index."""
    return np.vstack([X_blocks[row["X_block"]][row["X_ptr"]] for _, row in df_all.iterrows()])

def load_datasets(
    use_pad=True,
    use_derm12345=True,
    PAD_NPZ_PATH="dermatology_pad_precomputed_embeddings.npz",
    PAD_LABELS_CSV="PAD_labels.csv",
    DERM_NPZ_PATH="derm12345_google-derm-foundation-model_embeddings.npz",
    DERM_LABELS_CSV="derm12345_labels.csv",
    LABELS=("NEV","BCC","ACK","SEK","SCC","MEL"),
):
    """
    Loads selected datasets, maps embeddings to labels, and returns:
      X            : ndarray [N, D]
      y_onehot     : ndarray [N, C] in the order of LABELS
      df_all       : dataframe with columns (patient_id, image_id, source, LABELS, etc.)
      LABELS       : tuple of label names (for reference/order)
    """
    parts = []
    X_blocks = {}

    if use_pad:
        df_pad, X_pad = _load_npz_as_table(PAD_NPZ_PATH, source_tag="PAD")
        df_pad = _attach_labels(df_pad, PAD_LABELS_CSV, id_col_name="img_id", LABELS=list(LABELS))
        parts.append(df_pad)
        X_blocks["PAD"] = X_pad

    if use_derm12345:
        df_derm, X_derm = _load_npz_as_table(DERM_NPZ_PATH, source_tag="DERM")
        df_derm = _attach_labels(df_derm, DERM_LABELS_CSV, id_col_name="image_id", LABELS=list(LABELS))
        parts.append(df_derm)
        X_blocks["DERM"] = X_derm

    if not parts:
        raise ValueError("No dataset selected. Set use_pad and/or use_derm12345 to True.")

    df_all = pd.concat(parts, ignore_index=True)
    X = _pull_X_from_blocks(df_all, X_blocks)
    y_onehot = df_all[list(LABELS)].astype(int).values
    return X, y_onehot, df_all, tuple(LABELS)

def patient_level_split(df_all, LABELS=("NEV","BCC","ACK","SEK","SCC","MEL"), random_state=42):
    """
    Creates a 70/15/15 split at the PATIENT level using multilabel stratification if available;
    falls back to random patient split if iterstrat isn't installed.
    Returns:
      masks: dict with boolean masks { 'train', 'val', 'test' } for df_all rows
      per_patient: dataframe with patient-level multi-hot and assigned split
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        has_ml_strat = True
    except Exception:
        has_ml_strat = False

    # Build per-patient multi-hot
    per_patient = (
        df_all[["patient_id"] + list(LABELS)]
        .groupby("patient_id")[list(LABELS)].max()
        .reset_index()
    )
    patients = per_patient["patient_id"].astype(str).values
    Yp = per_patient[list(LABELS)].astype(int).values

    rng = np.random.RandomState(random_state)
    if has_ml_strat:
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        tr_pat_idx, temp_pat_idx = next(mskf.split(np.zeros((len(patients),1)), Yp))
        # split temp into val/test
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as MSKF2
        Ytemp = Yp[temp_pat_idx]
        mskf2 = MSKF2(n_splits=2, shuffle=True, random_state=random_state+1)
        val_rel, test_rel = next(mskf2.split(np.zeros((len(temp_pat_idx),1)), Ytemp))
        val_pat_idx = temp_pat_idx[val_rel]
        test_pat_idx = temp_pat_idx[test_rel]
    else:
        idx = np.arange(len(patients))
        rng.shuffle(idx)
        n = len(idx)
        tr_pat_idx = idx[:int(0.7*n)]
        val_pat_idx = idx[int(0.7*n):int(0.85*n)]
        test_pat_idx= idx[int(0.85*n):]

    # Map patient split back to row masks
    pat_to_split = {}
    for i in tr_pat_idx:   pat_to_split[patients[i]] = "train"
    for i in val_pat_idx:  pat_to_split[patients[i]] = "val"
    for i in test_pat_idx: pat_to_split[patients[i]] = "test"

    df_all = df_all.copy()
    df_all["split"] = df_all["patient_id"].astype(str).map(pat_to_split)

    masks = {
        "train": (df_all["split"] == "train").values,
        "val":   (df_all["split"] == "val").values,
        "test":  (df_all["split"] == "test").values,
    }

    # also return annotated per_patient table
    per_patient = per_patient.copy()
    per_patient["split"] = per_patient["patient_id"].astype(str).map(pat_to_split)
    return masks, per_patient
# --- example_load_call.py (put in the next cell) ---------------------------
# Config flags
USE_PAD = True
USE_DERM12345 = True

LABELS = ("BCC","ACK","SEK","SCC","MEL")

# Load data
X, y_onehot, df_all, LABELS = load_datasets(
    use_pad=USE_PAD,
    use_derm12345=USE_DERM12345,
    PAD_NPZ_PATH="dermatology_pad_precomputed_embeddings.npz",
    PAD_LABELS_CSV="PAD_labels.csv",
    DERM_NPZ_PATH="derm12345_google-derm-foundation-model_embeddings.npz",
    DERM_LABELS_CSV="derm12345_labels.csv",
    LABELS=LABELS
)

# Patient-level 70/15/15 split
masks, per_patient = patient_level_split(df_all, LABELS=LABELS, random_state=42)

print("Shapes: X =", X.shape, "| y_onehot =", y_onehot.shape)
print("Rows per split:", {k:int(v.sum()) for k,v in masks.items()})
print("Patients per split:", per_patient.groupby("split")["patient_id"].nunique().to_dict())

# (Optional) Indices for convenience
import numpy as np
train_idx = np.where(masks["train"])[0]
val_idx   = np.where(masks["val"])[0]
test_idx  = np.where(masks["test"])[0]

# (Optional) quick class counts by split
for split, mask in masks.items():
    counts = y_onehot[mask].sum(axis=0)
    print(f"{split} class counts:", dict(zip(LABELS, map(int, counts))))


# --- train_and_auc.py -------------------------------------------------------
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("⚠️ xgboost not installed; skipping XGBoost.")

# 1) Build y indices and split views
CLASS_NAMES = list(LABELS)  # e.g., ["NEV","BCC","ACK","SEK","SCC","MEL"]
y_idx = y_onehot.argmax(axis=1)

train_idx = np.where(masks["train"])[0]
val_idx   = np.where(masks["val"])[0]
test_idx  = np.where(masks["test"])[0]

X_train, y_train = X[train_idx], y_idx[train_idx]
X_val,   y_val   = X[val_idx],   y_idx[val_idx]
X_test,  y_test  = X[test_idx],  y_idx[test_idx]

# 2) Define models (with light, sensible defaults)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   multi_class="multinomial", random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))
    ]),
    "SVC (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42)

def per_class_auc(y_true_idx, proba, class_names):
    """Compute per-class ROC-AUC and macro AUC. Handles missing positive/negative cases safely."""
    n_classes = proba.shape[1]
    scores = {}
    vals = []
    for i, cname in enumerate(class_names):
        # Binary ground-truth for one-vs-rest
        y_bin = (y_true_idx == i).astype(int)
        # Skip if only one class present in y_bin
        if y_bin.min() == y_bin.max():
            scores[cname] = float("nan")
            continue
        try:
            auc = roc_auc_score(y_bin, proba[:, i])
        except Exception:
            auc = float("nan")
        scores[cname] = auc
        if not np.isnan(auc):
            vals.append(auc)
    macro = float(np.mean(vals)) if len(vals) else float("nan")
    return scores, macro

# 3) Train & evaluate
all_results = {}
for name, model in models.items():
    # Fit
    model.fit(X_train, y_train)
    # Predict probs on TEST
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
    else:
        # For pipelines, final step may be SVC etc.
        clf = model[-1] if isinstance(model, Pipeline) else model
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(model[:-1].transform(X_test) if isinstance(model, Pipeline) else X_test)
        else:
            # Fallback: use decision_function and squash via softmax
            decision = model.decision_function(X_test)
            # Softmax
            e = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)

    scores, macro = per_class_auc(y_test, proba, CLASS_NAMES)
    all_results[name] = (scores, macro)

# 4) Pretty print results
print("\n=== Per-class ROC-AUC (TEST) ===")
for name, (scores, macro) in all_results.items():
    line = f"{name:>20} | "
    parts = []
    for cname in CLASS_NAMES:
        val = scores.get(cname, float("nan"))
        parts.append(f"{cname}:{val:.3f}" if np.isfinite(val) else f"{cname}:nan")
    line += "  ".join(parts) + f"  ||  Macro:{macro:.3f}" if np.isfinite(macro) else f"  ||  Macro:nan"
    print(line)

# Optional: also report split sizes
print("\nImages per split ->",
      "Train:", len(train_idx), "| Val:", len(val_idx), "| Test:", len(test_idx))
