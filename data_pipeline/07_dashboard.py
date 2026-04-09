"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 7: Web Dashboard (Flask)
=============================================================================
A lightweight Flask web application providing:

  GET  /                    -> Dashboard home (vendor risk overview)
  GET  /vendor/<name>       -> Vendor detail page (history + risk + rules)
  GET  /regulations         -> Searchable regulation reference
  GET  /permit-checklist    -> Interactive permit checklist
  GET  /api/risk-scores     -> JSON API: all vendor risk scores
  GET  /api/vendor/<name>   -> JSON API: single vendor detail
  GET  /api/violations      -> JSON API: recent violations
  GET  /api/weather         -> JSON API: current weather risk

Run:
  python 07_dashboard.py
  Then open: http://localhost:5000
=============================================================================
"""

import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -- Data paths ----------------------------------------------------------------
CLEAN_DIR   = Path("suffolk_data/clean")
ML_DIR      = Path("suffolk_data/ml_ready")
REG_DIR     = Path("suffolk_data/regulations")
MODEL_DIR   = Path("suffolk_data/models")

TIER_COLORS = {"Low": "#2e7d32", "Medium": "#f57c00", "High": "#c62828", "Critical": "#880e4f"}
BRAND       = "#1f4e79"


# -- Data loader (cached in memory) --------------------------------------------
_cache: dict = {}

def load_data():
    if _cache:
        return _cache

    # Inspections
    insp_path = CLEAN_DIR / "inspections_clean.parquet"
    _cache["inspections"] = pd.read_parquet(insp_path) if insp_path.exists() else pd.DataFrame()

    # Risk scores
    risk_path = ML_DIR / "risk_scores.parquet"
    _cache["risk_scores"] = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()

    # Regulations
    rules_path = REG_DIR / "mobile_vendor_rules.json"
    _cache["rules"] = json.loads(rules_path.read_text()) if rules_path.exists() else []

    checklist_path = REG_DIR / "permit_checklist.json"
    _cache["checklist"] = json.loads(checklist_path.read_text()) if checklist_path.exists() else []

    vmap_path = REG_DIR / "violation_rule_map.json"
    _cache["vmap"] = json.loads(vmap_path.read_text()) if vmap_path.exists() else {}

    log.info(f"Data loaded: {len(_cache['inspections']):,} inspections, "
             f"{len(_cache['risk_scores']):,} risk scores, "
             f"{len(_cache['rules'])} rules")
    return _cache


# -- HTML page builder helpers -------------------------------------------------

def base_html(title: str, body: str, active_nav: str = "") -> str:
    nav_items = [
        ("Dashboard",       "/",               "?"),
        ("Regulations",     "/regulations",    "?"),
        ("Permit Checklist","/permit-checklist","?"),
        ("API Docs",        "/api-docs",       "?"),
    ]
    nav_html = "".join(
        f'<a href="{url}" class="nav-link {"active" if active_nav==label else ""}">{icon} {label}</a>'
        for label, url, icon in nav_items
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - Suffolk AI Compliance</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#222;}}
  nav{{background:{BRAND};display:flex;align-items:center;padding:0 24px;height:56px;gap:4px;}}
  nav .logo{{color:white;font-weight:700;font-size:1.1rem;margin-right:24px;white-space:nowrap;}}
  .nav-link{{color:rgba(255,255,255,.8);text-decoration:none;padding:8px 14px;border-radius:6px;font-size:.9rem;transition:.2s;}}
  .nav-link:hover,.nav-link.active{{background:rgba(255,255,255,.15);color:white;}}
  .page{{max-width:1200px;margin:28px auto;padding:0 20px;}}
  h1{{color:{BRAND};font-size:1.6rem;margin-bottom:6px;}}
  h2{{color:{BRAND};font-size:1.15rem;margin:24px 0 12px;}}
  .subtitle{{color:#666;font-size:.9rem;margin-bottom:24px;}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px;}}
  .card{{background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
  .card .num{{font-size:2rem;font-weight:700;color:{BRAND};}}
  .card .lbl{{font-size:.85rem;color:#666;margin-top:4px;}}
  table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;
         box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px;}}
  th{{background:{BRAND};color:white;padding:11px 14px;text-align:left;font-size:.88rem;}}
  td{{padding:10px 14px;border-bottom:1px solid #eee;font-size:.88rem;}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:#f9f9f9;}}
  .tier{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:600;color:white;}}
  .tier-Low{{background:#2e7d32;}}
  .tier-Medium{{background:#f57c00;}}
  .tier-High{{background:#c62828;}}
  .tier-Critical{{background:#880e4f;}}
  .search-bar{{width:100%;max-width:420px;padding:9px 14px;border:1px solid #ccc;
               border-radius:8px;font-size:.95rem;margin-bottom:16px;}}
  .rule-card{{background:white;border-radius:8px;padding:16px 20px;margin:10px 0;
              box-shadow:0 1px 4px rgba(0,0,0,.08);border-left:4px solid {BRAND};}}
  .rule-id{{font-weight:700;color:{BRAND};font-size:.82rem;background:#e3f2fd;
             padding:2px 8px;border-radius:10px;}}
  .rule-section{{font-size:.8rem;color:#777;margin-left:6px;}}
  .rule-obligation{{margin-top:6px;line-height:1.6;}}
  .rule-condition{{color:#555;font-size:.88rem;margin-top:5px;}}
  .rule-penalty{{color:#b71c1c;font-size:.88rem;margin-top:4px;}}
  .checklist-item{{background:white;border-radius:8px;padding:14px 18px;margin:8px 0;
                   display:flex;align-items:flex-start;gap:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);}}
  .checklist-item label{{cursor:pointer;line-height:1.5;}}
  .checklist-item .deadline{{font-size:.8rem;color:#888;margin-top:3px;}}
  .step-num{{background:{BRAND};color:white;border-radius:50%;width:28px;height:28px;
              display:flex;align-items:center;justify-content:center;font-size:.8rem;
              font-weight:700;flex-shrink:0;}}
  .progress-bar{{background:#e0e0e0;border-radius:8px;height:12px;margin:8px 0 20px;overflow:hidden;}}
  .progress-fill{{background:{BRAND};height:100%;border-radius:8px;transition:.4s;}}
  .hidden{{display:none;}}
  a{{color:{BRAND};text-decoration:none;}}
  a:hover{{text-decoration:underline;}}
  code{{background:#eef2f7;padding:2px 6px;border-radius:4px;font-size:.85rem;}}
  .api-endpoint{{background:white;border-radius:8px;padding:16px 20px;margin:10px 0;
                 box-shadow:0 1px 4px rgba(0,0,0,.06);}}
  .method{{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:700;
            font-size:.8rem;margin-right:8px;}}
  .GET{{background:#e3f2fd;color:#1565c0;}}
</style>
</head>
<body>
<nav>
  <span class="logo">? Suffolk AI Compliance</span>
  {nav_html}
</nav>
<div class="page">
{body}
</div>
</body>
</html>"""


def dashboard_page() -> str:
    data       = load_data()
    insp       = data["inspections"]
    scores     = data["risk_scores"]

    # Summary stats
    n_vendors  = insp["facility_name"].nunique() if "facility_name" in insp.columns else 0
    n_mobile   = int(insp["is_mobile_vendor"].sum()) if "is_mobile_vendor" in insp.columns else 0
    n_scores   = len(scores)
    high_risk  = int((scores["risk_tier"].isin(["High","Critical"])).sum()) if "risk_tier" in scores.columns else 0

    cards = f"""
<div class="cards">
  <div class="card"><div class="num">{n_vendors:,}</div><div class="lbl">Total Vendors (2020-2024)</div></div>
  <div class="card"><div class="num">{n_mobile:,}</div><div class="lbl">Mobile Vendor Records</div></div>
  <div class="card"><div class="num">{n_scores:,}</div><div class="lbl">Scored Inspections</div></div>
  <div class="card"><div class="num" style="color:#c62828">{high_risk:,}</div><div class="lbl">High/Critical Risk</div></div>
</div>"""

    # Top risk vendors table
    if not scores.empty and "risk_score" in scores.columns:
        top = scores.nlargest(20, "risk_score")[
            [c for c in ["facility_name","inspection_date","risk_score","risk_tier","is_mobile_vendor"] if c in scores.columns]
        ]
        rows = "".join(
            f"""<tr>
              <td><a href="/vendor/{row.get('facility_name','').replace(' ','%20')}">{row.get('facility_name','')}</a></td>
              <td>{str(row.get('inspection_date',''))[:10]}</td>
              <td>{row.get('risk_score','')}</td>
              <td><span class="tier tier-{row.get('risk_tier','')}">{row.get('risk_tier','')}</span></td>
              <td>{"? Yes" if row.get("is_mobile_vendor") else "No"}</td>
            </tr>"""
            for _, row in top.iterrows()
        )
        table = f"""
<h2>[!]? Highest Risk Vendors</h2>
<table>
  <tr><th>Facility</th><th>Last Inspected</th><th>Risk Score</th><th>Risk Tier</th><th>Mobile</th></tr>
  {rows}
</table>"""
    else:
        table = "<p style='color:#888;margin:20px 0'>Run Steps 4-5 of the pipeline to generate risk scores.</p>"

    body = f"""
<h1>Suffolk County Food Vendor Compliance Dashboard</h1>
<p class="subtitle">AI-powered inspection risk analysis x Data: Suffolk County Open Data 2020-2024</p>
{cards}
{table}
<h2>? Data Pipeline Status</h2>
<table>
  <tr><th>Step</th><th>File</th><th>Status</th></tr>
  <tr><td>01 Ingestion</td><td><code>suffolk_violations_all.json</code></td>
      <td>{"[OK] Present" if (Path("suffolk_data/raw/inspections/suffolk_violations_all.json")).exists() else "[X] Missing - run 01_data_ingestion.py"}</td></tr>
  <tr><td>02 Cleaning</td><td><code>inspections_clean.parquet</code></td>
      <td>{"[OK] Present" if (CLEAN_DIR/"inspections_clean.parquet").exists() else "[X] Missing - run 02_data_cleaning.py"}</td></tr>
  <tr><td>04 Features</td><td><code>features.parquet</code></td>
      <td>{"[OK] Present" if (ML_DIR/"features.parquet").exists() else "[X] Missing - run 04_feature_engineering.py"}</td></tr>
  <tr><td>05 Model</td><td><code>risk_scores.parquet</code></td>
      <td>{"[OK] Present" if (ML_DIR/"risk_scores.parquet").exists() else "[X] Missing - run 05_risk_model.py"}</td></tr>
  <tr><td>06 Regulations</td><td><code>mobile_vendor_rules.json</code></td>
      <td>{"[OK] Present" if (REG_DIR/"mobile_vendor_rules.json").exists() else "[X] Missing - run 06_regulation_extractor.py"}</td></tr>
</table>"""
    return base_html("Dashboard", body, "Dashboard")


def regulations_page() -> str:
    data  = load_data()
    rules = data["rules"]

    cards = "\n".join(f"""
<div class="rule-card">
  <div>
    <span class="rule-id">{r.get('rule_id','')}</span>
    <span class="rule-section">{r.get('section','')}</span>
  </div>
  <div style="font-weight:600;margin-top:6px">{r.get('topic','')}</div>
  <div class="rule-obligation">{r.get('obligation','')}</div>
  {"<div class='rule-condition'>? " + r['condition'] + "</div>" if r.get('condition') else ""}
  {"<div class='rule-penalty'>[!] Penalty: " + r['penalty'] + "</div>" if r.get('penalty') else ""}
</div>""" for r in rules)

    body = f"""
<h1>? Mobile Food Vendor Regulation Reference</h1>
<p class="subtitle">Suffolk County Sanitary Code + NYS Part 14 - {len(rules)} rules</p>
<input class="search-bar" type="text" placeholder="Search regulations..." oninput="
  const q=this.value.toLowerCase();
  document.querySelectorAll('.rule-card').forEach(c=>{{
    c.classList.toggle('hidden', !c.innerText.toLowerCase().includes(q))
  }})">
{cards}"""
    return base_html("Regulations", body, "Regulations")


def checklist_page() -> str:
    data      = load_data()
    checklist = data["checklist"]
    items_html = "\n".join(f"""
<div class="checklist-item">
  <div class="step-num">{item['step']}</div>
  <div>
    <label><input type="checkbox" onchange="updateProgress()"> {item['item']}</label>
    <div class="deadline">? {item['deadline']} {"? <strong>Required</strong>" if item.get('required') else ""}</div>
  </div>
</div>""" for item in checklist)

    body = f"""
<h1>? Mobile Food Vendor Permit Checklist</h1>
<p class="subtitle">Suffolk County Department of Health Services - {len(checklist)} required steps</p>
<div class="progress-bar"><div class="progress-fill" id="prog" style="width:0%"></div></div>
<p id="prog-label" style="color:#666;font-size:.88rem;margin-bottom:20px">0 of {len(checklist)} completed</p>
{items_html}
<script>
function updateProgress(){{
  const boxes = document.querySelectorAll('input[type=checkbox]');
  const done  = [...boxes].filter(b=>b.checked).length;
  const pct   = Math.round(done/boxes.length*100);
  document.getElementById('prog').style.width = pct+'%';
  document.getElementById('prog-label').textContent = done+' of '+boxes.length+' completed';
}}
</script>"""
    return base_html("Permit Checklist", body, "Permit Checklist")


def api_docs_page() -> str:
    endpoints = [
        ("GET", "/api/risk-scores",     "All vendor risk scores (JSON)"),
        ("GET", "/api/vendor/<name>",   "Single vendor inspection history + rules"),
        ("GET", "/api/violations",      "Recent violations with severity"),
        ("GET", "/api/weather",         "Current weather risk flag (Islip station)"),
        ("GET", "/api/rules",           "All extracted regulation rules"),
        ("GET", "/api/permit-checklist","Permit checklist items"),
    ]
    items = "\n".join(f"""
<div class="api-endpoint">
  <span class="method {method}">{method}</span>
  <code>{path}</code>
  <p style="color:#666;font-size:.88rem;margin-top:6px">{desc}</p>
</div>""" for method, path, desc in endpoints)

    body = f"""
<h1>? API Reference</h1>
<p class="subtitle">All endpoints return JSON. Base URL: <code>http://localhost:5000</code></p>
{items}"""
    return base_html("API Docs", body, "API Docs")


# -- Flask app -----------------------------------------------------------------

def create_app():
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        log.error("Flask not installed. Run: pip install flask")
        return None

    app = Flask(__name__)

    @app.route("/")
    def index():
        from flask import Response
        return Response(dashboard_page(), mimetype="text/html")

    @app.route("/regulations")
    def regulations():
        from flask import Response
        return Response(regulations_page(), mimetype="text/html")

    @app.route("/permit-checklist")
    def checklist():
        from flask import Response
        return Response(checklist_page(), mimetype="text/html")

    @app.route("/api-docs")
    def api_docs():
        from flask import Response
        return Response(api_docs_page(), mimetype="text/html")

    # -- JSON APIs -------------------------------------------------------------

    @app.route("/api/risk-scores")
    def api_risk_scores():
        data   = load_data()
        scores = data["risk_scores"]
        if scores.empty:
            return jsonify({"error": "No risk scores available. Run 05_risk_model.py first."}), 404
        mobile_only = request.args.get("mobile_only", "false").lower() == "true"
        if mobile_only and "is_mobile_vendor" in scores.columns:
            scores = scores[scores["is_mobile_vendor"] == 1]
        # Return top N
        n   = int(request.args.get("limit", 100))
        out = scores.nlargest(n, "risk_score") if "risk_score" in scores.columns else scores.head(n)
        return jsonify(out[[c for c in ["facility_name","inspection_date","risk_score","risk_tier"] if c in out.columns]]
                       .to_dict(orient="records"))

    @app.route("/api/vendor/<path:name>")
    def api_vendor(name: str):
        data = load_data()
        insp = data["inspections"]
        if insp.empty:
            return jsonify({"error": "No inspection data."}), 404
        vendor_insp = insp[insp["facility_name"].str.lower() == name.lower()]
        if vendor_insp.empty:
            return jsonify({"error": f"Vendor '{name}' not found."}), 404

        scores = data["risk_scores"]
        risk   = {}
        if not scores.empty and "facility_name" in scores.columns:
            v_scores = scores[scores["facility_name"].str.lower() == name.lower()]
            if not v_scores.empty:
                risk = v_scores.nlargest(1, "risk_score").iloc[0].to_dict()

        return jsonify({
            "facility_name": name,
            "inspection_count": len(vendor_insp),
            "latest_inspection": str(vendor_insp["inspection_date"].max())[:10] if "inspection_date" in vendor_insp.columns else None,
            "risk": {k: str(v) for k, v in risk.items()},
            "inspections": vendor_insp.tail(10).to_dict(orient="records"),
        })

    @app.route("/api/violations")
    def api_violations():
        data = load_data()
        insp = data["inspections"]
        if insp.empty:
            return jsonify([])
        cols = [c for c in ["facility_name","inspection_date","violation_code","severity","violation_text"] if c in insp.columns]
        recent = insp.sort_values("inspection_date", ascending=False).head(200) if "inspection_date" in insp.columns else insp.head(200)
        return jsonify(recent[cols].to_dict(orient="records"))

    @app.route("/api/rules")
    def api_rules():
        return jsonify(load_data()["rules"])

    @app.route("/api/permit-checklist")
    def api_checklist():
        return jsonify(load_data()["checklist"])

    @app.route("/api/weather")
    def api_weather():
        weather_path = Path("suffolk_data/clean/weather_clean.parquet")
        if not weather_path.exists():
            return jsonify({"status": "unavailable", "message": "Run pipeline with NOAA token to get weather data."})
        weather = pd.read_parquet(weather_path)
        latest  = weather.sort_values("date").iloc[-1].to_dict() if "date" in weather.columns else {}
        heat    = bool(latest.get("heat_risk", 0))
        rain    = bool(latest.get("rain_day", 0))
        return jsonify({
            "date":       str(latest.get("date",""))[:10],
            "tmax_f":     latest.get("tmax"),
            "prcp_in":    latest.get("prcp"),
            "heat_risk":  heat,
            "rain_day":   rain,
            "advisory":   "[!] Heat advisory - monitor cold-holding temps" if heat else
                          ("? Rain advisory - check food protection" if rain else "[OK] No weather advisory")
        })

    return app


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    if app:
        log.info("Starting Suffolk County AI Compliance Dashboard...")
        log.info("Open: http://localhost:5000")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Install Flask first:  pip install flask")
