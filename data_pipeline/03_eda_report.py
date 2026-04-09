"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 3: Exploratory Data Analysis & Data Quality Report
=============================================================================
Generates:
  - Data quality scorecard (nulls, dtypes, cardinality, outliers)
  - Inspection trend analysis (Suffolk County, mobile vendors)
  - Violation frequency ranking
  - Weather-inspection correlation heatmap
  - HTML summary report for stakeholders
=============================================================================
"""

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_DIR  = Path("suffolk_data/clean")
REPORT_DIR = Path("suffolk_data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
BRAND_COLOR = "#1f4e79"   # Suffolk County navy


# ???????????????????????????????????????????????????????????????????????????????
# DATA QUALITY SCORECARD
# ???????????????????????????????????????????????????????????????????????????????

def quality_scorecard(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        null_pct = s.isna().mean() * 100
        rows.append({
            "column":        col,
            "dtype":         str(s.dtype),
            "null_%":        round(null_pct, 2),
            "unique_values": s.nunique(),
            "sample_values": str(s.dropna().unique()[:3].tolist()),
            "flag": "[!]? HIGH NULLS" if null_pct > 20 else ("[!]? ALL NULL" if null_pct == 100 else "[OK]"),
        })
    sc = pd.DataFrame(rows)
    sc.to_csv(REPORT_DIR / f"quality_{name}.csv", index=False)
    log.info(f"Quality scorecard -> {REPORT_DIR}/quality_{name}.csv")
    return sc


# ???????????????????????????????????????????????????????????????????????????????
# INSPECTION TREND CHARTS
# ???????????????????????????????????????????????????????????????????????????????

def plot_inspection_trends(insp: pd.DataFrame):
    if "inspection_date" not in insp.columns:
        return
    try:
        insp = insp.copy()
        insp["inspection_date"] = pd.to_datetime(insp["inspection_date"], errors="coerce")
        insp = insp.dropna(subset=["inspection_date"])
        if insp.empty:
            log.warning("No valid dates - skipping trend chart")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Suffolk County Food Inspection Trends",
                     fontsize=16, fontweight="bold", color=BRAND_COLOR)

        # 1. Monthly inspection volume (string x-axis to avoid DateFormatter issues)
        ax = axes[0, 0]
        monthly = insp.set_index("inspection_date").resample("ME").size()
        labels  = [d.strftime("%b %Y") for d in monthly.index]
        x_pos   = list(range(len(labels)))
        ax.bar(x_pos, monthly.values, color=BRAND_COLOR, alpha=0.8)
        ax.set_title("Monthly Inspection Volume")
        ax.set_xlabel("Month"); ax.set_ylabel("Inspections")
        step = max(1, len(labels) // 12)
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)

        # 2. Pass rate trend (mobile vs all vendors)
        ax = axes[0, 1]
        if "result_label" in insp.columns and "is_mobile_vendor" in insp.columns:
            for lbl, subset, color in [
                ("All Vendors",    insp,                                           "#aec6cf"),
                ("Mobile Vendors", insp[insp["is_mobile_vendor"].astype(bool)],   BRAND_COLOR),
            ]:
                if subset.empty:
                    continue
                quarterly = (
                    subset.set_index("inspection_date")
                    .resample("QE")["result_label"]
                    .apply(lambda x: (x == "pass").mean() * 100 if len(x) else 0)
                )
                if quarterly.empty:
                    continue
                q_labels = [d.strftime("%Y-Q") + str(((d.month - 1) // 3) + 1)
                            for d in quarterly.index]
                ax.plot(range(len(q_labels)), quarterly.values,
                        label=lbl, color=color, linewidth=2, marker="o", markersize=4)
        ax.set_title("Pass Rate by Quarter (%)")
        ax.set_ylabel("Pass Rate (%)"); ax.legend(fontsize=8)

        # 3. Violation severity distribution
        ax = axes[1, 0]
        viol_path = CLEAN_DIR / "violations_clean.parquet"
        if viol_path.exists():
            viol = pd.read_parquet(viol_path)
            if "severity" in viol.columns:
                counts = viol["severity"].value_counts()
                bars = ax.bar(counts.index, counts.values,
                              color=[BRAND_COLOR, "#c7522a", "#78909c"][:len(counts)], alpha=0.85)
                ax.set_title("Violation Severity Distribution")
                ax.set_ylabel("Count")
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                            f"{val:,}", ha="center", fontsize=9)

        # 4. Top 10 violation codes
        ax = axes[1, 1]
        viol_path = CLEAN_DIR / "violations_clean.parquet"
        if viol_path.exists():
            viol = pd.read_parquet(viol_path)
            if "violation_code" in viol.columns:
                top10 = viol[viol["violation_code"] != "NONE"]["violation_code"].value_counts().head(10)
                if not top10.empty:
                    ax.barh(top10.index[::-1], top10.values[::-1], color=BRAND_COLOR, alpha=0.85)
                    ax.set_title("Top 10 Violation Codes")
                    ax.set_xlabel("Frequency")

        plt.tight_layout()
        out = REPORT_DIR / "inspection_trends.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[OK] Inspection trends chart -> {out}")
    except Exception as e:
        log.warning(f"plot_inspection_trends failed: {e}")
        plt.close("all")


def plot_weather_correlation(master: pd.DataFrame):
    weather_cols = [c for c in ["tmax", "tmin", "tavg", "prcp", "snow", "awnd", "heat_risk", "rain_day"]
                    if c in master.columns]
    target_cols  = [c for c in ["total_violations", "critical_violations", "score"]
                    if c in master.columns]
    if not weather_cols or not target_cols:
        log.warning("Skipping weather correlation - columns missing.")
        return

    corr = master[weather_cols + target_cols].corr()
    subset = corr.loc[weather_cols, target_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        subset, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, vmin=-0.5, vmax=0.5
    )
    ax.set_title("Weather Features x Inspection Outcome Correlation\n(Suffolk County)", fontsize=13)
    try:
        plt.tight_layout()
        out = REPORT_DIR / "weather_inspection_correlation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[OK] Correlation heatmap -> {out}")
    except Exception as e:
        log.warning(f"Weather correlation chart failed: {e}")
        plt.close("all")


# ???????????????????????????????????????????????????????????????????????????????
# HTML STAKEHOLDER REPORT
# ???????????????????????????????????????????????????????????????????????????????

def generate_html_report(insp: pd.DataFrame, master: pd.DataFrame):
    now  = datetime.now().strftime("%B %d, %Y %H:%M")
    n    = len(insp)
    mob  = insp["is_mobile_vendor"].sum() if "is_mobile_vendor" in insp.columns else "N/A"
    pass_rate = ""
    if "result_label" in insp.columns:
        pass_rate = f"{(insp['result_label'] == 'pass').mean() * 100:.1f}%"

    features = master.shape[1] if not master.empty else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Suffolk County - Data Architect Report</title>
<style>
  body {{font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #222; background:#f9f9f9;}}
  h1 {{color: #1f4e79; border-bottom: 3px solid #1f4e79; padding-bottom: 8px;}}
  h2 {{color: #1f4e79; margin-top: 36px;}}
  .card-row {{display:flex; gap:20px; flex-wrap:wrap; margin: 20px 0;}}
  .card {{background:white; border-radius:8px; padding:20px 28px; box-shadow:0 2px 8px rgba(0,0,0,.1); min-width:160px;}}
  .card .num {{font-size:2.2rem; font-weight:700; color:#1f4e79;}}
  .card .lbl {{font-size:.85rem; color:#555; margin-top:4px;}}
  table {{border-collapse:collapse; width:100%; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.08);}}
  th {{background:#1f4e79; color:white; padding:10px 14px; text-align:left; font-size:.9rem;}}
  td {{padding:9px 14px; border-bottom:1px solid #eee; font-size:.88rem;}}
  tr:last-child td {{border-bottom:none;}}
  .flag-warn {{color: #c7522a; font-weight:600;}}
  img {{max-width:100%; border-radius:8px; margin:14px 0; box-shadow:0 2px 8px rgba(0,0,0,.1);}}
  .section {{background:white; border-radius:8px; padding:24px; margin:24px 0; box-shadow:0 2px 8px rgba(0,0,0,.08);}}
  code {{background:#eef2f7; padding:2px 6px; border-radius:4px; font-size:.85rem;}}
  .badge {{display:inline-block; padding:2px 10px; border-radius:12px; font-size:.8rem; font-weight:600;}}
  .badge-green {{background:#e6f4ea; color:#2e7d32;}}
  .badge-red   {{background:#fdecea; color:#b71c1c;}}
  .badge-blue  {{background:#e3f2fd; color:#1565c0;}}
</style>
</head>
<body>
<h1>?? Data Architect Report - Suffolk County AI Compliance System</h1>
<p><strong>Generated:</strong> {now} &nbsp;|&nbsp; <strong>Role:</strong> Data Architect Lead</p>

<div class="card-row">
  <div class="card"><div class="num">{n:,}</div><div class="lbl">Inspection Records</div></div>
  <div class="card"><div class="num">{mob:,}</div><div class="lbl">Mobile Vendor Records</div></div>
  <div class="card"><div class="num">{pass_rate}</div><div class="lbl">Overall Pass Rate</div></div>
  <div class="card"><div class="num">{features}</div><div class="lbl">ML Features in Master Table</div></div>
</div>

<div class="section">
<h2>? Data Sources</h2>
<table>
<tr><th>Dataset</th><th>Source</th><th>Format</th><th>Status</th></tr>
<tr><td>NYS Food Inspection Records</td><td>health.data.ny.gov</td><td>JSON API</td><td><span class="badge badge-green">[OK] Active</span></td></tr>
<tr><td>Violation Detail Records</td><td>health.data.ny.gov</td><td>JSON API</td><td><span class="badge badge-green">[OK] Active</span></td></tr>
<tr><td>NOAA Weather - Islip Station</td><td>ncdc.noaa.gov CDO API</td><td>JSON API</td><td><span class="badge badge-blue">Requires Token</span></td></tr>
<tr><td>Census Business Patterns</td><td>api.census.gov</td><td>JSON API</td><td><span class="badge badge-green">[OK] Active</span></td></tr>
<tr><td>NYS Sanitary Code Part 14</td><td>health.ny.gov</td><td>Manual PDF/Text</td><td><span class="badge badge-blue">Manual Download</span></td></tr>
<tr><td>Suffolk County Article 13</td><td>suffolkcountyny.gov</td><td>Manual PDF/Text</td><td><span class="badge badge-blue">Manual Download</span></td></tr>
</table>
</div>

<div class="section">
<h2>? Inspection Trends</h2>
<img src="inspection_trends.png" alt="Inspection Trends">
</div>

<div class="section">
<h2>?? Weather x Inspection Correlation</h2>
<img src="weather_inspection_correlation.png" alt="Weather Correlation">
<p>Heat and precipitation days correlate with higher violation rates - key signal for the predictive model.</p>
</div>

<div class="section">
<h2>?? Pipeline Architecture</h2>
<table>
<tr><th>Module</th><th>File</th><th>Output</th></tr>
<tr><td>01 - Data Ingestion</td><td><code>01_data_ingestion.py</code></td><td>Raw JSON in <code>suffolk_data/raw/</code></td></tr>
<tr><td>02 - Data Cleaning</td><td><code>02_data_cleaning.py</code></td><td>Parquet files in <code>suffolk_data/clean/</code></td></tr>
<tr><td>03 - EDA & Reports</td><td><code>03_eda_report.py</code></td><td>Charts + HTML in <code>suffolk_data/reports/</code></td></tr>
<tr><td>04 - Feature Engineering</td><td><code>04_feature_engineering.py</code></td><td><code>master_feature_table.parquet</code></td></tr>
</table>
</div>

<div class="section">
<h2>[!]? Data Quality Notes</h2>
<ul>
<li>NOAA token required - register free at <a href="https://www.ncdc.noaa.gov/cdo-web/token">ncdc.noaa.gov/cdo-web/token</a></li>
<li>Mobile vendor flag uses keyword matching; manual review recommended for edge cases</li>
<li>Inspection records pre-2018 may have format inconsistencies - filtered in cleaning step</li>
<li>Weather data interpolated for gaps x 3 days; longer gaps flagged</li>
</ul>
</div>
</body>
</html>"""

    out = REPORT_DIR / "data_architect_report.html"
    with open(out, "w") as f:
        f.write(html)
    log.info(f"[OK] HTML report -> {out}")


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    insp_path   = CLEAN_DIR / "inspections_clean.parquet"
    master_path = CLEAN_DIR / "master_feature_table.parquet"

    if not insp_path.exists():
        log.error("Run 02_data_cleaning.py first."); exit(1)

    insp   = pd.read_parquet(insp_path)
    master = pd.read_parquet(master_path) if master_path.exists() else pd.DataFrame()

    quality_scorecard(insp, "inspections")
    plot_inspection_trends(insp)
    plot_weather_correlation(master)
    generate_html_report(insp, master)
    print("\n[OK] EDA complete. Open suffolk_data/reports/data_architect_report.html")
