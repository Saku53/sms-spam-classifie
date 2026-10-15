import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Dataset Loading
# Replace with actual loading if spam.csv is in the correct path
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.drop_duplicates()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Advanced Preprocessing
def advanced_clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\d{10,}', ' PHONE ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[\£\$]', ' MONEY ', text)
    return text

df['clean_message'] = df['message'].apply(advanced_clean)

# 3. Feature Engineering Class
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame({
            'char_count': X.str.len(),
            'word_count': X.str.split().str.len(),
            'upper_ratio': X.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x)>0 else 0),
            'excl_count': X.str.count('!')
        })

# 4. Pipeline Construction
numeric_transformer = Pipeline([('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2), 'clean_message'),
        ('tfidf_char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5)), 'clean_message'),
        ('numeric', numeric_transformer, ['char_count', 'word_count', 'upper_ratio', 'excl_count'])
    ]
)

# NOTE: This requires a custom approach to combine features efficiently
# For simplicity in this script, we'll use a FeatureUnion-like approach implicitly or explicitly
# LinearSVC is the chosen baseline
model = Pipeline([
    ('features', preprocessor),
    ('clf', LinearSVC(class_weight='balanced', dual='auto'))
])

# 5. Training
X_train, X_test, y_train, y_test = train_test_split(df[['clean_message', 'char_count', 'word_count', 'upper_ratio', 'excl_count']], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Update train features to match the pipeline expectation if necessary, 
# but ColumnTransformer works on DataFrames, so it should be fine.

# Fit and Save
# (Add fitting logic here after refining the ColumnTransformer setup)
# joblib.dump(model, 'backend/model.joblib')
"""Train, tune, evaluate and export the SMS spam classifier.
Run from the project root:  python train_model.py
Needs spam.csv (Kaggle SMS Spam Collection) next to this file."""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from features import normalize_text, handcrafted_features  # noqa: E402

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

SEED = 42
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_data():
    path = ROOT / "spam.csv"
    if not path.exists():
        sys.exit(f"spam.csv not found. Put it here: {path}")
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    print(f"Rows: {len(df)} | spam: {(df.label == 'spam').sum()} | ham: {(df.label == 'ham').sum()}")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # BEFORE splitting -> no train/test leakage
    print(f"Removed {before - len(df)} duplicate messages -> {len(df)} left")
    df["y"] = (df.label == "spam").astype(int)
    return df


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True)),
        ("hand", Pipeline([("f", FunctionTransformer(handcrafted_features)),
                           ("s", StandardScaler())])),
    ])


def build_models():
    def pipe(clf):
        return Pipeline([("features", make_features()), ("clf", clf)])

    svm_cal = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    ens = VotingClassifier([
        ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ], voting="soft")
    return {
        "SVM (calibrated)": (pipe(svm_cal), {"clf__estimator__C": [0.3, 1, 3]}),
        "Logistic Regression": (pipe(lr), {"clf__C": [3, 10, 30]}),
        "Ensemble (SVM+LR)": (pipe(ens), {}),
    }


def metrics_at(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def pick_threshold(y_true, proba, min_recall=0.97):
    """Highest-precision threshold that still keeps recall >= min_recall (uses training data only)."""
    p, r, t = precision_recall_curve(y_true, proba)
    best_thr, best_prec = 0.5, -1
    for pi, ri, ti in zip(p[:-1], r[:-1], t):
        if ri >= min_recall and pi > best_prec:
            best_prec, best_thr = pi, ti
    return float(best_thr)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["y"], test_size=0.2, random_state=SEED, stratify=df["y"])
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, (pipeline, grid) in build_models().items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        gs = GridSearchCV(pipeline, grid or {}, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        train_time = time.time() - t0
        t1 = time.time()
        proba = gs.predict_proba(X_test)[:, 1]
        latency_ms = (time.time() - t1) / len(X_test) * 1000
        m = metrics_at(y_test, proba)
        m.update({"cv_f1": gs.best_score_, "best_params": gs.best_params_,
                  "train_time_s": round(train_time, 1), "latency_ms": round(latency_ms, 3)})
        results[name], fitted[name] = m, (gs.best_estimator_, proba)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    # Choose the model by cross-validation F1 (NOT by looking at the test set)
    best_name = max(results, key=lambda n: results[n]["cv_f1"])
    best_model, best_proba = fitted[best_name]
    print(f"\nBest by CV F1: {best_name}")

    # Threshold chosen from out-of-fold predictions on the TRAIN set only
    oof = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr = pick_threshold(y_train, oof, min_recall=0.97)
    final = metrics_at(y_test, best_proba, thr)
    print(f"Chosen threshold: {thr:.3f}")
    print("Final TEST metrics at that threshold:", {k: round(v, 4) for k, v in final.items()})
    print(classification_report(y_test, (best_proba >= thr).astype(int), target_names=["ham", "spam"]))

    # ---- Plots ----
    plt.figure(figsize=(6, 5))
    for n, (_, pr) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, pr)
        plt.plot(fpr, tpr, label=f"{n} (AUC={results[n]['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for n, (_, pr) in fitted.items():
        p, r, _ = precision_recall_curve(y_test, pr)
        plt.plot(r, p, label=f"{n} (AP={results[n]['pr_auc']:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "pr_curve.png", dpi=150); plt.close()

    cm = confusion_matrix(y_test, (best_proba >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

    p, r, t = precision_recall_curve(y_test, best_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(t, p[:-1], label="precision"); plt.plot(t, r[:-1], label="recall")
    plt.axvline(thr, color="k", ls="--", label=f"chosen threshold {thr:.2f}")
    plt.xlabel("Threshold"); plt.legend(); plt.title("Precision / recall vs threshold")
    plt.tight_layout(); plt.savefig(OUT / "threshold_curve.png", dpi=150); plt.close()

    # Misclassified messages for error analysis
    pred = (best_proba >= thr).astype(int)
    err = pd.DataFrame({"message": X_test.values, "true": y_test.values, "pred": pred, "spam_prob": best_proba})
    err[err.true != err.pred].to_csv(OUT / "misclassified.csv", index=False)

    # ---- Export (final model refit on ALL data) ----
    best_model.fit(df["message"], df["y"])
    joblib.dump({"pipeline": best_model, "threshold": thr, "model_name": best_name},
                ROOT / "backend" / "model.joblib")
    summary = {"best_model": best_name, "threshold": thr, "test_metrics_at_threshold": final,
               "all_models": results}
    (ROOT / "backend" / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: backend/model.joblib, backend/metrics.json, plots + misclassified.csv in {OUT}")


if __name__ == "__main__":
    main()