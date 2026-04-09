"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 2: Data Cleaning & Standardization
=============================================================================
Cleans, normalizes, and validates:
  - Inspection records (dates, scores, violation codes, vendor type)
  - Violation detail records (code mapping, severity tagging)
  - Weather data (unit conversion, gap-filling)
  - Business pattern data (NAICS alignment)
Outputs: cleaned Parquet files in suffolk_data/clean/
=============================================================================
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DIR   = Path("suffolk_data/raw")
CLEAN_DIR = Path("suffolk_data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# -- Known mobile food vendor type keywords ------------------------------------
MOBILE_KEYWORDS = [
    "food truck", "mobile", "cart", "catering truck", "food cart",
    "pushcart", "hotdog cart", "ice cream truck", "concession trailer",
    "roving", "itinerant", "temporary food"
]

# -- NYS Violation Code Severity Map (Part 14 / Suffolk Article 13) ------------
# Critical = immediate health risk | Non-Critical = administrative
VIOLATION_SEVERITY = {
    # Critical violations
    "1A": "critical", "1B": "critical", "2A": "critical", "2B": "critical",
    "3A": "critical", "4A": "critical", "5A": "critical", "5B": "critical",
    "5C": "critical", "5D": "critical", "5E": "critical", "6A": "critical",
    "7A": "critical", "8A": "critical", "8B": "critical", "8C": "critical",
    # Non-critical violations
    "9A": "non_critical", "9B": "non_critical", "9C": "non_critical",
    "10A": "non_critical", "10B": "non_critical", "11A": "non_critical",
    "11B": "non_critical", "11C": "non_critical", "11D": "non_critical",
    "12A": "non_critical", "12B": "non_critical", "12C": "non_critical",
    "12D": "non_critical", "12E": "non_critical",
}

# -- Violation descriptions per NYS Part 14 -----------------------------------
VIOLATION_DESCRIPTIONS = {
    "1A": "Food from unapproved source",
    "2A": "Food temperature abuse (hot/cold holding)",
    "2B": "Inadequate cooling procedures",
    "3A": "Inadequate cooking temperatures",
    "4A": "Cross-contamination / improper food handling",
    "5A": "Hands not washed properly",
    "5B": "Ill food worker not excluded",
    "5C": "Bare-hand contact with ready-to-eat food",
    "6A": "Water supply / plumbing deficiency",
    "8A": "Pest evidence (rodent/insect)",
    "9A": "Non-food contact surface unclean",
    "10A": "Improper equipment / utensil washing",
    "11A": "Inadequate ventilation / lighting",
    "12A": "Permit not posted / expired permit",
}


# ???????????????????????????????????????????????????????????????????????????????
# INSPECTION RECORD CLEANER
# ???????????????????????????????????????????????????????????????????????????????

def _flatten_date_dicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    When pandas reads JSON containing ISO date strings, it sometimes parses them
    into nested dicts {year:X, month:Y, day:Z} stored as object columns.
    This helper detects ALL such columns (regardless of name) and reassembles
    them into plain "YYYY-MM-DD" strings. Run this before AND after renaming.
    """
    for col in list(df.columns):
        series = df[col]
        # Skip if it is already a proper Series (not a DataFrame sub-object)
        if not isinstance(series, pd.Series):
            continue
        sample = series.dropna()
        if len(sample) == 0:
            continue
        first = sample.iloc[0]
        # Case A: nested dict with date components
        if isinstance(first, dict) and set(first.keys()) >= {"year", "month", "day"}:
            try:
                df[col] = pd.to_datetime({
                    "year":  series.apply(lambda x: x.get("year")  if isinstance(x, dict) else None),
                    "month": series.apply(lambda x: x.get("month") if isinstance(x, dict) else None),
                    "day":   series.apply(lambda x: x.get("day")   if isinstance(x, dict) else None),
                }, errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        # Case B: integer epoch milliseconds stored as int64
        elif pd.api.types.is_integer_dtype(series) and series.max() > 1e10:
            try:
                df[col] = pd.to_datetime(series, unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return df


def _safe_read_json(path: Path) -> pd.DataFrame:
    """
    Read a JSON file and immediately:
      1. Flatten any nested date-dict columns
      2. Deduplicate column names (keeps first occurrence)
      3. Ensure no column is a DataFrame (can happen with duplicate names)
    """
    df = pd.read_json(path)
    df = _flatten_date_dicts(df)
    # Deduplicate column names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    # Force any remaining DataFrame-typed columns to string
    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].astype(str)
    return df


def clean_inspections(raw_path: Path) -> pd.DataFrame:
    log.info(f"Cleaning inspections: {raw_path}")
    df = _safe_read_json(raw_path)

    # -- Normalize column names ------------------------------------------------
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # -- Key column aliases ----------------------------------------------------
    # Suffolk County Open Data (ArcGIS) field names:
    #   FacilityID, FacilityName, Address, City, Zip,
    #   InspectionDate, SanitaryCodeSection, ViolationText
    col_map = {
        "facilityid":            "facility_id",
        "facilityname":          "facility_name",
        "address":               "address",
        "city":                  "town",
        "zip":                   "zip",
        "inspectiondate":        "inspection_date",
        "sanitarycodesection":   "violation_code",
        "violationtext":         "violation_text",
        "source_year":           "source_year",
        # fallback lower-underscore variants
        "facility_id":           "facility_id",
        "facility_name":         "facility_name",
        "inspection_date":       "inspection_date",
        "sanitary_code_section": "violation_code",
        "violation_text":        "violation_text",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # -- Date parsing ----------------------------------------------------------
    # -- Robust date parsing: handles strings, epoch-ms ints, nested dicts --
    date_col = None
    for candidate in ["inspection_date", "inspectiondate", "InspectionDate",
                      "inspection date", "insp_date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    # Nuclear fix: ensure inspection_date is a single datetime Series
    # regardless of how many columns share that name or what type they are
    if "inspection_date" in df.columns:
        col = df["inspection_date"]
        # If it's a DataFrame (duplicate col names merged), take first column
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        # Convert to datetime
        if pd.api.types.is_integer_dtype(col):
            df["inspection_date"] = pd.to_datetime(col, unit="ms", errors="coerce")
        elif pd.api.types.is_float_dtype(col):
            df["inspection_date"] = pd.to_datetime(col.astype("Int64"), unit="ms", errors="coerce")
        else:
            df["inspection_date"] = pd.to_datetime(col.astype(str).str.strip(), errors="coerce")
    elif date_col and date_col != "inspection_date":
        df = df.rename(columns={date_col: "inspection_date"})
        col = df["inspection_date"]
        df["inspection_date"] = pd.to_datetime(col.astype(str).str.strip(), errors="coerce")
        # Guarantee inspection_date is a proper datetime Series before .dt access
        d = df["inspection_date"]
        if isinstance(d, pd.DataFrame):
            d = pd.to_datetime(d.iloc[:, 0].astype(str), errors="coerce")
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
            df["inspection_date"] = d
        if not pd.api.types.is_datetime64_any_dtype(df["inspection_date"]):
            df["inspection_date"] = pd.to_datetime(df["inspection_date"].astype(str), errors="coerce")
        df["inspection_year"]    = df["inspection_date"].dt.year
        df["inspection_month"]   = df["inspection_date"].dt.month
        df["inspection_dow"]     = df["inspection_date"].dt.dayofweek
        df["inspection_quarter"] = df["inspection_date"].dt.quarter

    # -- Flag mobile food vendors ----------------------------------------------
    name_col  = "facility_name"  if "facility_name"  in df.columns else None
    type_col  = "facility_type"  if "facility_type"  in df.columns else None

    def is_mobile(row):
        text = " ".join(filter(None, [
            str(row.get(name_col, "")).lower() if name_col else "",
            str(row.get(type_col, "")).lower() if type_col else "",
        ]))
        return any(kw in text for kw in MOBILE_KEYWORDS)

    df["is_mobile_vendor"] = df.apply(is_mobile, axis=1)

    # -- Resolve violation code column (real CSV uses SanitaryCodeSection) ------
    for _vc in ["sanitarycodesection", "sanitary_code_section", "violationcode",
                "violation_item", "violation_code"]:
        if _vc in df.columns and _vc != "violation_code":
            df["violation_code"] = df[_vc].astype(str).str.strip().str.upper()
            break

    if "violation_code" in df.columns:
        df["violation_code"] = df["violation_code"].astype(str).str.strip().str.upper()
        df["severity"] = df["violation_code"].map(VIOLATION_SEVERITY).fillna("unknown")
    else:
        df["violation_code"] = "NONE"
        df["severity"] = "unknown"

    # -- Count violations per inspection event --------------------------------
    if "facility_name" in df.columns and "inspection_date" in df.columns:
        viol_counts = (
            df.groupby(["facility_name", "inspection_date"])
            .agg(
                total_violations    = ("violation_code", "count"),
                critical_violations = ("severity",
                                       lambda x: int((x == "critical").sum())),
            )
            .reset_index()
        )
        df = df.merge(viol_counts, on=["facility_name", "inspection_date"], how="left")
        df["critical_violations"] = df["critical_violations"].fillna(0).astype(int)
        df["total_violations"]    = df["total_violations"].fillna(0).astype(int)

    # -- Simple pass/fail label -----------------------------------------------
    if "critical_violations" in df.columns:
        df["result_label"] = df["critical_violations"].apply(
            lambda x: "fail" if x >= 3 else ("conditional" if x >= 1 else "pass")
        )

    # -- Days since last inspection --------------------------------------------
    if "inspection_date" in df.columns:
        df = df.sort_values(["facility_name", "inspection_date"])
        df["days_since_last_inspection"] = (
            df.groupby("facility_name")["inspection_date"]
            .diff().dt.days
        )

    # -- Drop rows with no facility name or date -------------------------------
    before = len(df)
    df = df.dropna(subset=[c for c in ["facility_name", "inspection_date"] if c in df.columns])
    log.info(f"Dropped {before - len(df)} rows missing name/date. Remaining: {len(df):,}")

    # -- Save ------------------------------------------------------------------
    # -- Sanitize columns before parquet save (pyarrow is strict about types) --
    for col in df.columns:
        if df[col].dtype == object:
            # Check if the column has mixed str/int/float - convert all to str
            try:
                df[col] = df[col].astype(str).replace("nan", "").replace("<NA>", "")
            except Exception:
                df[col] = df[col].astype(str)
        elif str(df[col].dtype) == "category":
            df[col] = df[col].astype(str)

    out = CLEAN_DIR / "inspections_clean.parquet"
    df.to_parquet(out, index=False)
    mobile_count = int(df["is_mobile_vendor"].sum()) if "is_mobile_vendor" in df.columns else 0
    log.info(f"[OK] Inspections saved -> {out}  ({len(df):,} rows, {mobile_count} mobile vendors)")
    return df


# ???????????????????????????????????????????????????????????????????????????????
# VIOLATION DETAIL CLEANER
# ???????????????????????????????????????????????????????????????????????????????

def clean_violations(raw_path: Path) -> pd.DataFrame:
    log.info(f"Cleaning violations: {raw_path}")
    df = _safe_read_json(raw_path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Rename date column immediately after normalization (before col_map)
    # Real Suffolk CSV: "inspectiondate" -> "inspection_date"
    if "inspectiondate" in df.columns and "inspection_date" not in df.columns:
        df = df.rename(columns={"inspectiondate": "inspection_date"})
    # Rename facility name column
    if "facilityname" in df.columns and "facility_name" not in df.columns:
        df = df.rename(columns={"facilityname": "facility_name"})

    # Map both NYS DOH and Suffolk County / synthetic field names
    col_map = {
        # NYS DOH format
        "facilityname":          "facility_name",
        "inspectiondate":        "inspection_date",
        "violationcode":         "violation_code",
        "violationitem":         "violation_item",
        "violationmemo":         "violation_memo",
        # Suffolk County Open Data / synthetic format
        "facilityid":            "facility_id",
        "facility_name":         "facility_name",
        "sanitarycodesection":   "violation_code",
        "violationtext":         "violation_text",
        "inspection_date":       "inspection_date",
        "source_year":           "source_year",
        # already-lowercased variants
        "sanitary_code_section": "violation_code",
        "violation_text":        "violation_text",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = _flatten_date_dicts(df)   # run again after rename so inspection_date is resolved

    # -- Robust date parsing: handles strings, epoch-ms ints, nested dicts --
    date_col = None
    for candidate in ["inspection_date", "inspectiondate", "InspectionDate",
                      "inspection date", "insp_date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    # Convert inspection_date to proper datetime regardless of source format
    if "inspection_date" not in df.columns and date_col:
        df = df.rename(columns={date_col: "inspection_date"})
    if "inspection_date" in df.columns:
        col = df["inspection_date"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if pd.api.types.is_integer_dtype(col):
            df["inspection_date"] = pd.to_datetime(col, unit="ms", errors="coerce")
        else:
            df["inspection_date"] = pd.to_datetime(col.astype(str).str.strip(), errors="coerce")

    # Derive violation_code from any available column
    if "violation_code" not in df.columns:
        for fallback in ["sanitarycodesection", "sanitary_code_section", "violation_item"]:
            if fallback in df.columns:
                df["violation_code"] = df[fallback].astype(str)
                break

    if "violation_code" in df.columns:
        df["violation_code"] = df["violation_code"].astype(str).str.strip().str.upper()
        df["severity"]       = df["violation_code"].map(VIOLATION_SEVERITY).fillna("unknown")
        df["violation_desc"] = df["violation_code"].map(VIOLATION_DESCRIPTIONS).fillna(
            df.get("violation_text", pd.Series("", index=df.index)).astype(str)
        )
    else:
        df["violation_code"] = "UNKNOWN"
        df["severity"]       = "unknown"
        df["violation_desc"] = ""

    # Ensure facility_name exists
    if "facility_name" not in df.columns:
        for fallback in ["facilityname", "FacilityName"]:
            if fallback in df.columns:
                df["facility_name"] = df[fallback]
                break

    # -- Sanitize columns before parquet save --
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype(str).replace("nan", "").replace("<NA>", "")
            except Exception:
                df[col] = df[col].astype(str)

    out = CLEAN_DIR / "violations_clean.parquet"
    df.to_parquet(out, index=False)
    log.info(f"[OK] Violations saved -> {out}  ({len(df):,} rows)")
    return df


# ???????????????????????????????????????????????????????????????????????????????
# WEATHER DATA CLEANER
# ???????????????????????????????????????????????????????????????????????????????

def clean_weather(raw_path: Path) -> pd.DataFrame:
    log.info(f"Cleaning weather: {raw_path}")
    df = pd.read_json(raw_path)
    df.columns = [c.lower().strip() for c in df.columns]

    # NOAA CDO response shape: {results: [...]} or flat list
    if "results" in df.columns:
        df = pd.json_normalize(df["results"].explode())
        df.columns = [c.lower() for c in df.columns]

    # Pivot: one row per date, one column per datatype
    if "datatype" in df.columns and "value" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        df.columns.name = None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

    # Unit conversions
    # NOAA CDO with units=standard returns:
    #   TMAX/TMIN in tenths of degrees Celsius -> convert to Fahrenheit
    #   PRCP/SNOW in tenths of mm -> convert to inches
    #   AWND in tenths of m/s -> m/s
    # Synthetic data already has clean Fahrenheit values -- detect and skip.
    if "tmax" in df.columns and df["tmax"].dropna().max() < 200:
        # Looks like tenths-Celsius (real NOAA data): convert
        for col in ["tmax", "tmin"]:
            if col in df.columns:
                df[col] = df[col] / 10 * 9/5 + 32
        for col in ["prcp", "snow"]:
            if col in df.columns:
                df[col] = df[col] / 10 / 25.4
        if "awnd" in df.columns:
            df["awnd"] = df["awnd"] / 10
    # else: synthetic data already in Fahrenheit/inches -- no conversion needed

    # Derived features
    if "tmax" in df.columns and "tmin" in df.columns:
        df["tavg"]          = (df["tmax"] + df["tmin"]) / 2
        df["temp_range"]    = df["tmax"] - df["tmin"]
        df["heat_risk"]     = (df["tmax"] >= 90).astype(int)  # food safety concern >90?F
        df["freeze_risk"]   = (df["tmin"] <= 32).astype(int)

    if "prcp" in df.columns:
        df["rain_day"]      = (df["prcp"] > 0).astype(int)
        df["heavy_rain"]    = (df["prcp"] > 0.5).astype(int)

    # Forward-fill gaps x 3 days
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")
        df = df.resample("D").mean()
        df = df.interpolate(method="time", limit=3)
        df = df.reset_index()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace("nan", "").replace("<NA>", "")

    out = CLEAN_DIR / "weather_clean.parquet"
    df.to_parquet(out, index=False)
    log.info(f"[OK] Weather saved -> {out}  ({len(df):,} days)")
    return df


# ???????????????????????????????????????????????????????????????????????????????
# MASTER FEATURE TABLE
# ???????????????????????????????????????????????????????????????????????????????

def build_master_table() -> pd.DataFrame:
    """
    Joins cleaned inspections + weather into a single ML-ready feature table.
    Each row = one inspection event with associated weather on that day.
    """
    log.info("Building master feature table...")

    insp_path    = CLEAN_DIR / "inspections_clean.parquet"
    weather_path = CLEAN_DIR / "weather_clean.parquet"

    if not insp_path.exists() or not weather_path.exists():
        log.error("Run clean_inspections() and clean_weather() first.")
        return pd.DataFrame()

    insp    = pd.read_parquet(insp_path)
    weather = pd.read_parquet(weather_path)

    if "inspection_date" in insp.columns:
        insp["inspection_date"] = pd.to_datetime(insp["inspection_date"])
    if "date" in weather.columns:
        weather["date"] = pd.to_datetime(weather["date"])

    master = insp.merge(
        weather,
        left_on="inspection_date",
        right_on="date",
        how="left"
    )

    # -- Violation frequency features per vendor -------------------------------
    viol_path = CLEAN_DIR / "violations_clean.parquet"
    if viol_path.exists():
        viol = pd.read_parquet(viol_path)
        viol["inspection_date"] = pd.to_datetime(viol.get("inspection_date"), errors="coerce")

        # Ensure required columns exist before aggregating
        if "violation_code" not in viol.columns:
            viol["violation_code"] = "UNKNOWN"
        if "severity" not in viol.columns:
            viol["severity"] = "unknown"
        if "facility_name" not in viol.columns:
            viol["facility_name"] = viol.get("facilityname", pd.Series("", index=viol.index))

        viol_summary = (
            viol.groupby(["facility_name", "inspection_date"])
            .agg(
                n_violations   = ("violation_code", "count"),
                critical_count = ("severity", lambda x: (x == "critical").sum()),
                unique_codes   = ("violation_code", "nunique"),
            )
            .reset_index()
        )
        master = master.merge(
            viol_summary,
            on=["facility_name", "inspection_date"],
            how="left",
            suffixes=("", "_viol")
        )

    # -- Repeat offender flag --------------------------------------------------
    if "critical_violations" in master.columns:
        master["repeat_critical"] = (
            master.groupby("facility_name")["critical_violations"]
            .transform(lambda x: (x > 0).shift(1).fillna(False).astype(int))
        )

    for col in master.columns:
        if master[col].dtype == object:
            master[col] = master[col].astype(str).replace("nan", "").replace("<NA>", "")
        elif str(master[col].dtype) == "category":
            master[col] = master[col].astype(str)

    out = CLEAN_DIR / "master_feature_table.parquet"
    master.to_parquet(out, index=False)
    log.info(f"[OK] Master table -> {out}  ({len(master):,} rows x {master.shape[1]} features)")
    return master


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    # Suffolk County combined violations file (from 01_data_ingestion.py)
    suffolk_combined = RAW_DIR / "inspections" / "suffolk_violations_all.json"
    wthr = RAW_DIR / "weather" / "noaa_islip_daily.json"

    if suffolk_combined.exists():
        clean_inspections(suffolk_combined)
        clean_violations(suffolk_combined)   # same file - dual clean
    else:
        log.warning("suffolk_violations_all.json not found - run 01_data_ingestion.py first")

    if wthr.exists():
        clean_weather(wthr)

    master = build_master_table()
    if not master.empty:
        print("\n-- Master Table Preview ---------------------------------")
        print(master.head(3).to_string())
        print(f"\nShape: {master.shape}")
        mobile_col = "is_mobile_vendor"
        print(f"Mobile vendors: {master[mobile_col].sum() if mobile_col in master.columns else 'N/A'}")
