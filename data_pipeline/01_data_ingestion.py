"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM -- DATA ARCHITECT MODULE
Agent 1: Data Ingestion (v4 -- robust multi-strategy fetch)
=============================================================================
Strategy for Suffolk County violation data (3 attempts in order):
  1. ArcGIS Hub API  -- query the Hub metadata endpoint to get the real
                        FeatureServer URL dynamically, then paginate it.
  2. Hub GeoJSON     -- direct GeoJSON export link from the Hub API response.
  3. ECO portal      -- Suffolk County's own restaurant search portal
                        https://eco.suffolkcountyny.gov (scrape JSON API).

NOAA fix: CDO API max date range per request is 1 year. We chunk by year.

Other sources:
  - US Census CBP 2022 (food service business counts, Suffolk FIPS 103)
=============================================================================
"""

import io
import os
import sys
import time
import logging
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# -- Force UTF-8 stdout -------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# -- Directories --------------------------------------------------------------
BASE_DIR = Path("suffolk_data")
RAW_DIR  = BASE_DIR / "raw"
META_DIR = BASE_DIR / "metadata"
for d in [RAW_DIR / "inspections", RAW_DIR / "weather",
          RAW_DIR / "regulations", RAW_DIR / "census", META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NOAA_TOKEN = "ROFECOOtQsJObXMRDAblIUkIvgWOdepd"

# -- Dataset slugs on opendata.suffolkcountyny.gov ----------------------------
# The Hub API lets us resolve the real FeatureServer URL from the slug.
SUFFOLK_HUB_SLUGS = {
    2024: "restaurant-violations-2024",
    2023: "restaurant-violations-2023-1",
    2022: "restaurant-violations-2022",
    2021: "restaurant-violations-2021",
    2020: "restaurant-violations-2020",
}

HUB_API = "https://opendata.suffolkcountyny.gov/api/v3/datasets/{slug}?f=json"

# =============================================================================
# STRATEGY 1 + 2: ArcGIS Hub API -> resolve FeatureServer -> paginate
# =============================================================================

def resolve_feature_server(slug: str) -> tuple:
    """
    Call the Hub API for the dataset slug and return:
      (featureserver_query_url, geojson_url)
    Returns (None, None) on failure.
    """
    url = HUB_API.format(slug=slug)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        meta = r.json()

        # The 'url' field in the layer contains the FeatureServer base URL
        # Example: "https://services1.arcgis.com/XYZ/arcgis/rest/services/NAME/FeatureServer/0"
        layers = meta.get("data", {}).get("attributes", {}).get("layer", {})
        fs_url = layers.get("url", "")

        if fs_url and "FeatureServer" in fs_url:
            # Ensure it ends at the layer index (not the query endpoint)
            if not fs_url.rstrip("/").endswith("/query"):
                fs_url = fs_url.rstrip("/") + "/query"
            log.info(f"  Resolved FeatureServer: {fs_url[:80]}")
        else:
            fs_url = None

        # GeoJSON export URL is also in the Hub response
        geojson_url = None
        access = meta.get("data", {}).get("attributes", {}).get("access", {})
        for link in access.get("download", []):
            if link.get("format", "").lower() == "geojson":
                geojson_url = link.get("href")
                break

        return fs_url, geojson_url

    except Exception as e:
        log.warning(f"  Hub API resolve failed for {slug}: {e}")
        return None, None


def paginate_feature_server(query_url: str, year: int) -> list:
    """Paginate an ArcGIS FeatureServer query endpoint, 1000 records at a time."""
    all_records = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        try:
            r = requests.get(query_url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                log.warning(f"  [suffolk_{year}] ArcGIS error: {data['error']}")
                break

            features = data.get("features", [])
            if not features:
                break

            for feat in features:
                all_records.append(feat.get("attributes", {}))

            if not data.get("exceededTransferLimit", False):
                break
            offset += page_size

        except Exception as e:
            log.warning(f"  [suffolk_{year}] Pagination error at offset {offset}: {e}")
            break

    return all_records


def fetch_geojson(url: str, year: int) -> list:
    """Download a GeoJSON and flatten to a list of attribute dicts."""
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        records = []
        for feat in data.get("features", []):
            rec = feat.get("properties", {})
            records.append(rec)
        log.info(f"  [suffolk_{year}] GeoJSON got {len(records):,} records")
        return records
    except Exception as e:
        log.warning(f"  [suffolk_{year}] GeoJSON fetch failed: {e}")
        return []


# =============================================================================
# STRATEGY 3: Suffolk County ECO portal JSON API
# eco.suffolkcountyny.gov serves inspection data via a REST-like JSON endpoint
# =============================================================================

def fetch_eco_portal(year: int) -> list:
    """
    Fetch from Suffolk County's ECO portal.
    The ECO portal at apps.suffolkcountyny.gov/health/Restaurant exposes
    a search endpoint that returns JSON.
    We page through by record offset.
    """
    log.info(f"  [suffolk_{year}] Trying ECO portal fallback...")
    base = "https://apps.suffolkcountyny.gov/health/Restaurant/Results"
    all_records = []
    page = 1
    page_size = 200

    # ECO portal search params - blank query returns all, filtered by year
    # via InspectionDate range
    start = f"01/01/{year}"
    end   = f"12/31/{year}"

    while True:
        params = {
            "EstablishmentName": "",
            "Address": "",
            "City": "",
            "InspectionDateFrom": start,
            "InspectionDateTo": end,
            "Page": page,
            "PageSize": page_size,
            "format": "json",
        }
        try:
            r = requests.get(base, params=params, timeout=60,
                             headers={"Accept": "application/json",
                                      "X-Requested-With": "XMLHttpRequest"})
            r.raise_for_status()
            data = r.json()

            items = data if isinstance(data, list) else data.get("results", data.get("data", []))
            if not items:
                break

            all_records.extend(items)
            log.info(f"  [suffolk_{year}] ECO page {page}: {len(items)} records")

            if len(items) < page_size:
                break
            page += 1

        except Exception as e:
            log.warning(f"  [suffolk_{year}] ECO portal error (page {page}): {e}")
            break

    return all_records


# =============================================================================
# MASTER SUFFOLK VIOLATIONS FETCHER
# =============================================================================

def fetch_suffolk_violations(years=None) -> pd.DataFrame:
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    frames = []
    for year in years:
        out_path = RAW_DIR / "inspections" / f"suffolk_violations_{year}.json"

        # Use cache if available
        if out_path.exists():
            log.info(f"[suffolk_{year}] Loading from cache")
            frames.append(pd.read_json(out_path))
            continue

        log.info(f"[suffolk_{year}] Fetching data for {year}...")
        slug      = SUFFOLK_HUB_SLUGS.get(year, f"restaurant-violations-{year}")
        all_records = []

        # -- Strategy 1: Hub API -> FeatureServer pagination ------------------
        fs_url, geojson_url = resolve_feature_server(slug)
        if fs_url:
            all_records = paginate_feature_server(fs_url, year)
            if all_records:
                log.info(f"[suffolk_{year}] FeatureServer got {len(all_records):,} records")

        # -- Strategy 2: GeoJSON export from Hub API --------------------------
        if not all_records and geojson_url:
            all_records = fetch_geojson(geojson_url, year)

        # -- Strategy 3: ECO portal fallback ----------------------------------
        if not all_records:
            all_records = fetch_eco_portal(year)

        if all_records:
            df = pd.DataFrame(all_records)
            df["source_year"] = year
            df.to_json(out_path, orient="records", indent=2, date_format="iso")
            log.info(f"[suffolk_{year}] [OK] {len(df):,} records saved")
            frames.append(df)
        else:
            log.error(
                f"[suffolk_{year}] All 3 strategies failed. "
                f"Manual download instructions:"
            )
            log.error(
                f"  1. Open: https://opendata.suffolkcountyny.gov/datasets/{slug}"
            )
            log.error(
                f"  2. Click 'Download' -> 'CSV'"
            )
            log.error(
                f"  3. Save as: suffolk_data/raw/inspections/suffolk_violations_{year}.json"
            )

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined_path = RAW_DIR / "inspections" / "suffolk_violations_all.json"
        combined.to_json(combined_path, orient="records", indent=2, date_format="iso")
        log.info(f"Combined: {len(combined):,} rows -> {combined_path}")
        return combined
    return pd.DataFrame()


# =============================================================================
# NOAA WEATHER -- chunked by year (CDO max range = 1 year per request)
# =============================================================================

def fetch_noaa_weather(token=None) -> pd.DataFrame:
    tok      = token or os.getenv("NOAA_TOKEN", "") or NOAA_TOKEN
    out_path = RAW_DIR / "weather" / "noaa_islip_daily.json"

    if out_path.exists():
        log.info(f"[noaa_weather] Loading from cache")
        return pd.read_json(out_path)

    log.info(f"[noaa_weather] Token: {tok[:8]}{'*'*16}")
    headers  = {"token": tok}
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    all_results = []

    # CDO hard limit: max 1 year per request. Chunk 2020-2024 year by year.
    # Also: CDO data lags ~6-12 months; cap each year end at Dec 31.
    today = datetime.today()
    for yr in range(2020, today.year + 1):
        start = f"{yr}-01-01"
        # Cap end: don't ask for dates beyond today minus 30 days
        yr_end   = datetime(yr, 12, 31)
        safe_end = min(yr_end, today - timedelta(days=30))
        if safe_end < datetime(yr, 1, 1):
            log.info(f"[noaa_weather] Skipping {yr} (too recent)")
            continue
        end = safe_end.strftime("%Y-%m-%d")

        log.info(f"[noaa_weather] Fetching {start} -> {end}...")
        offset = 1
        while True:
            params = {
                "datasetid":  "GHCND",
                "stationid":  "GHCND:USW00094741",
                "datatypeid": "TMAX,TMIN,PRCP,SNOW,AWND",
                "startdate":  start,
                "enddate":    end,
                "limit":      1000,
                "units":      "standard",
                "offset":     offset,
            }
            try:
                r = requests.get(base_url, headers=headers, params=params, timeout=60)
                r.raise_for_status()
                data    = r.json()
                results = data.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                meta  = data.get("metadata", {}).get("resultset", {})
                total = meta.get("count", 0)
                limit = meta.get("limit", 1000)
                if offset + limit > total:
                    break
                offset += limit
                time.sleep(0.2)  # be polite to NOAA API
            except Exception as e:
                log.error(f"[noaa_weather] Error for {yr}: {e}")
                break

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_json(out_path, orient="records", indent=2)
        log.info(f"[noaa_weather] [OK] {len(df):,} records saved")
        return df

    log.error("[noaa_weather] No data returned. Check token or try later.")
    return pd.DataFrame()


# =============================================================================
# CENSUS BUSINESS PATTERNS
# =============================================================================

def fetch_census() -> pd.DataFrame:
    out_path = RAW_DIR / "census" / "suffolk_food_business_patterns.json"
    if out_path.exists():
        log.info("[census] Loading from cache")
        return pd.read_json(out_path)

    log.info("[census] Fetching Census Business Patterns...")
    url = (
        "https://api.census.gov/data/2022/cbp"
        "?get=NAICS2017,ESTAB,EMP,PAYANN,NAME"
        "&for=county:103&in=state:36&NAICS2017=7225"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        raw = r.json()
        if isinstance(raw, list) and isinstance(raw[0], list):
            headers = []
            seen = {}
            for h in raw[0]:
                if h in seen:
                    seen[h] += 1
                    headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    headers.append(h)
            df = pd.DataFrame(raw[1:], columns=headers)
        else:
            df = pd.DataFrame(raw if isinstance(raw, list) else [raw])
        df.to_json(out_path, orient="records", indent=2)
        log.info(f"[census] [OK] {len(df):,} rows saved")
        return df
    except Exception as e:
        log.error(f"[census] Failed: {e}")
        return pd.DataFrame()


# =============================================================================
# MASTER INGEST
# =============================================================================

def ingest_all(noaa_token=None):
    results = {}

    log.info("-- Ingesting Suffolk County violation records --")
    viol = fetch_suffolk_violations()
    if not viol.empty:
        results["suffolk_violations"] = viol

    log.info("-- Ingesting NOAA weather --")
    weather = fetch_noaa_weather(token=noaa_token)
    if not weather.empty:
        results["noaa_weather_islip"] = weather

    log.info("-- Ingesting Census business patterns --")
    census = fetch_census()
    if not census.empty:
        results["census_business_patterns"] = census

    return results


def write_manifest(results: dict) -> pd.DataFrame:
    sources = {
        "suffolk_violations":      ("Suffolk County restaurant violations 2020-2024",
                                    str(RAW_DIR / "inspections" / "suffolk_violations_all.json")),
        "noaa_weather_islip":      ("NOAA daily weather Islip Airport",
                                    str(RAW_DIR / "weather" / "noaa_islip_daily.json")),
        "census_business_patterns":("Census CBP food service Suffolk County",
                                    str(RAW_DIR / "census" / "suffolk_food_business_patterns.json")),
    }
    rows = []
    for key, (desc, path) in sources.items():
        df = results.get(key)
        rows.append({
            "source_key":  key,
            "description": desc,
            "rows":        len(df) if df is not None else 0,
            "status":      "[OK]" if (df is not None and not df.empty) else "[--]",
            "ingested_at": datetime.now().isoformat(),
            "local_path":  path,
        })
    mdf = pd.DataFrame(rows)
    mdf.to_csv(META_DIR / "ingestion_manifest.csv", index=False)
    log.info(f"Manifest -> {META_DIR}/ingestion_manifest.csv")
    return mdf


# -- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaa-token", default=NOAA_TOKEN)
    args    = parser.parse_args()
    results = ingest_all(noaa_token=args.noaa_token)
    mdf     = write_manifest(results)
    print("\n-- Ingestion Summary --")
    print(mdf[["source_key", "rows", "status"]].to_string(index=False))


# =============================================================================
# MANUAL CSV LOADER
# If all API strategies fail, user can manually download CSV files from:
#   https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-20XX
# and drop them in:  suffolk_data/raw/inspections/manual/
# This function picks them up automatically on the next run.
# =============================================================================

def load_manual_csvs() -> pd.DataFrame:
    """
    Load any manually downloaded CSV files from suffolk_data/raw/inspections/manual/
    Expected filenames: violations_2020.csv, violations_2021.csv, etc.
    Also accepts any .csv file dropped in that folder.
    """
    manual_dir = RAW_DIR / "inspections" / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(manual_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            # Try UTF-8 first, then latin-1 (some Suffolk exports use latin-1)
            try:
                df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
            except Exception:
                df = pd.read_csv(f, encoding="latin-1")

            # URL-decode filename for year detection (handles %3A etc.)
            from urllib.parse import unquote
            stem_clean = unquote(f.stem).replace(":", "_").replace(" ", "_")

            # Try to detect year from filename
            yr_match = [s for s in stem_clean.split("_") if s.isdigit() and len(s) == 4]
            df["source_year"] = int(yr_match[0]) if yr_match else 0

            # Normalise column names to lowercase
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Convert date columns to ISO strings to prevent epoch-ms serialisation
            for date_candidate in ["inspectiondate", "inspection_date", "InspectionDate",
                                   "inspection date", "insp_date", "date"]:
                if date_candidate in df.columns:
                    df[date_candidate] = pd.to_datetime(
                        df[date_candidate].astype(str).str.strip(),
                        errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                    break

            frames.append(df)
            log.info(f"[manual_csv] Loaded {len(df):,} rows from {f.name} (year={df['source_year'].iloc[0]})")
        except Exception as e:
            log.error(f"[manual_csv] Could not load {f.name}: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Ensure all remaining date-like columns are strings before JSON save
        for col in combined.columns:
            if hasattr(combined[col], 'dt'):
                combined[col] = combined[col].astype(str)
        out = RAW_DIR / "inspections" / "suffolk_violations_all.json"
        combined.to_json(out, orient="records", indent=2, date_format="iso")
        log.info(f"[manual_csv] Combined {len(combined):,} rows saved")
        return combined
    return pd.DataFrame()
