"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 5: Predictive Inspection Risk Model
=============================================================================
Trains and evaluates a risk-scoring model to predict:
  - fail_flag  : whether a vendor will fail/get conditional on next inspection
  - risk_score : probability score (0-100) used for dashboard display

Models trained:
  1. Random Forest Classifier  (primary - handles missing features well)
  2. Gradient Boosting (XGBoost-style via sklearn)  (secondary)
  3. Logistic Regression  (baseline)

Outputs:
  - suffolk_data/models/risk_model.pkl      <- trained Random Forest
  - suffolk_data/models/model_report.txt    <- classification report
  - suffolk_data/ml_ready/risk_scores.parquet <- scored vendor table
  - suffolk_data/reports/model_performance.png
=============================================================================
"""

import logging
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline         import Pipeline
from sklearn.impute            import SimpleImputer

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ML_DIR     = Path("suffolk_data/ml_ready")
MODEL_DIR  = Path("suffolk_data/models")
REPORT_DIR = Path("suffolk_data/reports")
for d in [MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BRAND_COLOR = "#1f4e79"

# -- Feature groups ------------------------------------------------------------
FEATURE_COLS = [
    # Violation history
    "critical_violations", "total_violations",
    "crit_viol_roll3m", "crit_viol_roll6m", "crit_viol_roll12m",
    "had_critical_last",
    # Temporal
    "days_since_last_inspection", "inspections_that_year",
    "is_summer", "is_weekend",
    # Vendor profile
    "is_mobile_vendor", "town_enc",
    # Weather (optional - may be absent without NOAA token)
    "tavg", "heat_risk", "rain_day",
]


def load_data() -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    feat_path  = ML_DIR / "features.parquet"
    label_path = ML_DIR / "labels.parquet"

    if not feat_path.exists() or not label_path.exists():
        log.error("features.parquet or labels.parquet not found. Run 04_feature_engineering.py first.")
        return None, None

    features = pd.read_parquet(feat_path)
    labels   = pd.read_parquet(label_path)

    df = features.merge(labels, on=["facility_name", "inspection_date"], how="inner")
    log.info(f"Loaded {len(df):,} labeled records.")

    # Keep only columns that exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Using {len(available)} features: {available}")

    X = df[available]
    y = df["fail_flag"]

    log.info(f"Class distribution - pass: {(y==0).sum():,}  |  fail/conditional: {(y==1).sum():,}")
    return X, y


def build_pipeline(model_type: str = "rf") -> Pipeline:
    """Build sklearn Pipeline with imputation -> scaling -> model."""
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    else:  # logistic regression baseline
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train all three models, evaluate, save best one."""
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model_type in [("RandomForest", "rf"), ("GradientBoosting", "gb"), ("LogisticRegression", "lr")]:
        log.info(f"Training {name}...")
        pipe = build_pipeline(model_type)
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_prob  = pipe.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_prob)
        cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        report = classification_report(y_test, y_pred, target_names=["pass", "fail"])
        log.info(f"{name} - AUC: {auc:.3f}  |  CV AUC: {cv_aucs.mean():.3f} x {cv_aucs.std():.3f}")

        results[name] = {
            "pipe":    pipe,
            "auc":     auc,
            "cv_auc":  cv_aucs.mean(),
            "report":  report,
            "y_test":  y_test,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
            "X_test":  X_test,
        }

    # -- Save best model (highest CV AUC) -------------------------------------
    best_name = max(results, key=lambda k: results[k]["cv_auc"])
    best      = results[best_name]
    log.info(f"Best model: {best_name}  (CV AUC = {best['cv_auc']:.3f})")

    model_path = MODEL_DIR / "risk_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best["pipe"], "features": list(X.columns), "model_name": best_name}, f)
    log.info(f"[OK] Model saved -> {model_path}")

    # -- Write text report -----------------------------------------------------
    report_lines = [
        "=" * 60,
        "  SUFFOLK COUNTY COMPLIANCE - MODEL PERFORMANCE REPORT",
        "=" * 60, "",
    ]
    for name, res in results.items():
        report_lines += [
            f"-- {name} ------------------------------",
            f"  Test AUC:   {res['auc']:.4f}",
            f"  CV AUC:     {res['cv_auc']:.4f}",
            "",
            "  Classification Report:",
            res["report"],
            "",
        ]
    report_lines.append(f"?  Best model selected: {best_name}")

    with open(MODEL_DIR / "model_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    log.info(f"[OK] Report -> {MODEL_DIR}/model_report.txt")

    return results, best_name


def plot_model_performance(results: dict, best_name: str):
    """4-panel model performance chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Suffolk County Inspection Risk Model - Performance", fontsize=14,
                 fontweight="bold", color=BRAND_COLOR)

    colors = {"RandomForest": BRAND_COLOR, "GradientBoosting": "#c7522a", "LogisticRegression": "#78909c"}

    # 1. ROC Curves
    ax = axes[0, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=colors[name], lw=2)
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_title("ROC Curves"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. CV AUC comparison
    ax = axes[0, 1]
    names  = list(results.keys())
    aucs   = [results[n]["cv_auc"] for n in names]
    bars   = ax.bar(names, aucs, color=[colors[n] for n in names], alpha=0.85)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("5-Fold CV AUC Comparison"); ax.set_ylabel("Mean AUC")
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

    # 3. Confusion matrix for best model
    ax = axes[1, 0]
    best = results[best_name]
    cm = confusion_matrix(best["y_test"], best["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Pass", "Fail"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {best_name}")

    # 4. Feature importances (RF only)
    ax = axes[1, 1]
    rf_res = results.get("RandomForest")
    if rf_res:
        clf       = rf_res["pipe"].named_steps["clf"]
        feat_names = list(rf_res["X_test"].columns)
        importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
        top = importances.tail(12)
        ax.barh(top.index, top.values, color=BRAND_COLOR, alpha=0.85)
        ax.set_title("Top Feature Importances (Random Forest)")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    out = REPORT_DIR / "model_performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[OK] Performance chart -> {out}")


def score_all_vendors(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Apply the saved model to score every vendor in the dataset."""
    model_path = MODEL_DIR / "risk_model.pkl"
    if not model_path.exists():
        log.error("risk_model.pkl not found.")
        return pd.DataFrame()

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    pipe         = bundle["model"]
    model_feats  = bundle["features"]

    # Align columns
    X_aligned = X.reindex(columns=model_feats)
    probs = pipe.predict_proba(X_aligned)[:, 1]

    feat_path = ML_DIR / "features.parquet"
    df        = pd.read_parquet(feat_path)
    df["risk_score"]   = (probs * 100).round(1)
    df["risk_tier"]    = pd.cut(df["risk_score"],
                                bins=[0, 30, 60, 80, 100],
                                labels=["Low", "Medium", "High", "Critical"])
    df["model_name"]   = bundle["model_name"]

    out = ML_DIR / "risk_scores.parquet"
    df.to_parquet(out, index=False)
    log.info(f"[OK] Risk scores -> {out}  ({len(df):,} records)")
    return df


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        if y.nunique() < 2:
            log.warning("Only one class in labels - cannot train model. Need more data.")
        else:
            results, best_name = train_and_evaluate(X, y)
            plot_model_performance(results, best_name)
            scored = score_all_vendors(X, list(X.columns))
            if not scored.empty:
                print("\n-- Risk Score Preview (top 10 highest risk) ----------")
                cols = ["facility_name", "inspection_date", "risk_score", "risk_tier"]
                cols = [c for c in cols if c in scored.columns]
                print(scored.nlargest(10, "risk_score")[cols].to_string(index=False))
