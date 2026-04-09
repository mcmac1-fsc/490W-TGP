"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 4: Feature Engineering - ML-Ready Dataset
=============================================================================
Builds the final feature set for:
  - Predictive inspection model (risk score / pass-fail prediction)
  - Violation frequency forecasting
  - Permit renewal risk flagging
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_DIR = Path("suffolk_data/clean")
ML_DIR    = Path("suffolk_data/ml_ready")
ML_DIR.mkdir(parents=True, exist_ok=True)


def build_features() -> pd.DataFrame:
    master_path = CLEAN_DIR / "master_feature_table.parquet"
    if not master_path.exists():
        log.error("master_feature_table.parquet not found. Run 02_data_cleaning.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(master_path)
    log.info(f"Loaded master table: {df.shape}")

    # -- 1. Rolling violation averages per vendor ------------------------------
    df = df.sort_values(["facility_name", "inspection_date"])

    for window in [3, 6, 12]:
        col = f"critical_violations"
        if col in df.columns:
            df[f"crit_viol_roll{window}m"] = (
                df.groupby("facility_name")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    # -- 2. Days since last violation -----------------------------------------
    if "critical_violations" in df.columns:
        df["had_critical_last"] = (
            df.groupby("facility_name")["critical_violations"]
            .transform(lambda x: (x.shift(1) > 0).astype(int))
        )

    # -- 3. Inspection frequency (inspections per year per vendor) -------------
    if "inspection_date" in df.columns:
        df["inspection_year"] = pd.to_datetime(df["inspection_date"]).dt.year
        freq = (
            df.groupby(["facility_name", "inspection_year"])
            .size()
            .reset_index(name="inspections_that_year")
        )
        df = df.merge(freq, on=["facility_name", "inspection_year"], how="left")

    # -- 4. Seasonal features --------------------------------------------------
    if "inspection_month" in df.columns:
        df["is_summer"]  = df["inspection_month"].isin([6, 7, 8]).astype(int)
        df["is_weekend"] = df.get("inspection_dow", pd.Series(0)).isin([5, 6]).astype(int)

    # -- 5. Encode categorical columns -----------------------------------------
    cat_cols = ["facility_type", "town", "inspection_type"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))

    # -- 6. Binary target: fail_flag (fail or conditional = 1) -----------------
    if "result_label" in df.columns:
        df["fail_flag"] = (df["result_label"].astype(str).isin(["fail", "conditional"])).astype(int)

    # -- 7. Select final feature columns ---------------------------------------
    numeric_features = [
        "critical_violations", "non_critical_violations", "total_violations",
        "days_since_last_inspection", "crit_viol_roll3m", "crit_viol_roll6m",
        "crit_viol_roll12m", "had_critical_last", "inspections_that_year",
        "is_summer", "is_weekend", "is_mobile_vendor",
        "tavg", "tmax", "heat_risk", "rain_day", "heavy_rain", "awnd",
        "facility_type_enc", "town_enc", "inspection_type_enc",
    ]
    available = [c for c in numeric_features if c in df.columns]
    target    = "fail_flag" if "fail_flag" in df.columns else None

    # -- 8. Save ML-ready tables -----------------------------------------------
    features_df = df[["facility_name", "inspection_date"] + available].copy()
    features_df.to_parquet(ML_DIR / "features.parquet", index=False)
    log.info(f"[OK] Features -> {ML_DIR}/features.parquet  ({len(available)} features)")

    if target:
        labels_df = df[["facility_name", "inspection_date", target]].copy()
        labels_df.to_parquet(ML_DIR / "labels.parquet", index=False)
        log.info(f"[OK] Labels  -> {ML_DIR}/labels.parquet   (target: {target})")

    # -- 9. Feature summary ----------------------------------------------------
    summary = pd.DataFrame({
        "feature":  available,
        "null_%":   [round(df[c].isna().mean() * 100, 1) for c in available],
        "mean":     [round(df[c].mean(), 3) for c in available],
        "std":      [round(df[c].std(), 3) for c in available],
        "category": [
            "violation_history" if "viol" in c or "critical" in c else
            "temporal"          if any(x in c for x in ["day", "month", "year", "season", "summer", "weekend"]) else
            "weather"           if any(x in c for x in ["t_", "tmax", "tavg", "rain", "heat", "awnd"]) else
            "vendor_profile"    if "mobile" in c or "enc" in c or "freq" in c or "inspections" in c else
            "other"
            for c in available
        ],
    })
    summary.to_csv(ML_DIR / "feature_summary.csv", index=False)
    log.info(f"[OK] Feature summary -> {ML_DIR}/feature_summary.csv")
    return features_df


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    features = build_features()
    if not features.empty:
        print("\n-- ML Feature Table Preview ------------------------------")
        print(features.head(5).to_string())
        print(f"\nShape: {features.shape}")
