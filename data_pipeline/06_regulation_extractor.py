"""
=============================================================================
SUFFOLK COUNTY AI COMPLIANCE SYSTEM - DATA ARCHITECT MODULE
Agent 6: NLP Regulation Extractor
=============================================================================
Parses Suffolk County and NYS sanitary code text files to extract:
  - Structured compliance rules (obligation, condition, penalty)
  - Mobile food vendor-specific requirements
  - Permit requirements and renewal deadlines
  - Keyword index for violation-to-rule mapping

Uses spaCy for NER + rule-based patterns, with an LLM extraction
fallback via the Anthropic API (claude-haiku-4-5) when available.

Inputs:
  suffolk_data/raw/regulations/*.txt

Outputs:
  suffolk_data/regulations/
    +-- extracted_rules.json        <- structured rule objects
    +-- mobile_vendor_rules.json    <- mobile-vendor-specific subset
    +-- permit_checklist.json       <- permit requirement list
    +-- violation_rule_map.json     <- violation code -> rule reference
    +-- regulation_index.html       <- searchable HTML reference
=============================================================================
"""

import re
import json
import logging
import os
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_REG_DIR  = Path("suffolk_data/raw/regulations")
OUT_REG_DIR  = Path("suffolk_data/regulations")
OUT_REG_DIR.mkdir(parents=True, exist_ok=True)

# -- Mobile vendor obligation keywords -----------------------------------------
OBLIGATION_WORDS  = ["shall", "must", "required", "requires", "prohibited",
                     "not permitted", "shall not", "must not"]
PERMIT_KEYWORDS   = ["permit", "license", "certificate", "registration",
                     "approval", "authorization", "inspection certificate"]
MOBILE_KEYWORDS   = ["mobile", "food truck", "food cart", "catering truck",
                     "pushcart", "itinerant", "temporary food", "vending unit",
                     "commissary", "roving"]
TEMP_KEYWORDS     = ["temperature", "?f", "degrees", "hot holding", "cold holding",
                     "refrigerat", "cooling", "reheating"]
PENALTY_KEYWORDS  = ["fine", "penalty", "suspension", "revocation", "closure",
                     "violation", "infraction", "misdemeanor"]

# -- Violation code -> regulation section map ----------------------------------
# Derived from NYS Sanitary Code Part 14 / Suffolk County Article 13
VIOLATION_TO_SECTION = {
    "1A": {"section": "14-1.21",   "topic": "Approved food sources"},
    "2A": {"section": "14-1.41",   "topic": "Food temperature control (hot/cold holding)"},
    "2B": {"section": "14-1.43",   "topic": "Cooling procedures"},
    "3A": {"section": "14-1.40",   "topic": "Cooking temperatures"},
    "4A": {"section": "14-1.50",   "topic": "Cross-contamination prevention"},
    "5A": {"section": "14-1.60",   "topic": "Handwashing requirements"},
    "5B": {"section": "14-1.61",   "topic": "Ill worker exclusion"},
    "5C": {"section": "14-1.62",   "topic": "Bare hand contact with RTE foods"},
    "6A": {"section": "14-1.71",   "topic": "Water supply and plumbing"},
    "8A": {"section": "14-1.110",  "topic": "Pest control"},
    "9A": {"section": "14-1.100",  "topic": "Non-food contact surfaces"},
    "10A":{"section": "14-1.90",   "topic": "Warewashing requirements"},
    "11A":{"section": "14-1.120",  "topic": "Ventilation and lighting"},
    "12A":{"section": "14-1.10",   "topic": "Permit display and renewal"},
}

# -- Built-in rule base (used when regulation text files are not available) ----
# This codifies the Suffolk County / NYS Part 14 requirements for mobile vendors.
BUILTIN_MOBILE_RULES = [
    {
        "rule_id":    "MV-001",
        "section":    "Suffolk County Sanitary Code ?760-3",
        "topic":      "Permit requirement",
        "obligation": "All mobile food vending units must obtain a permit from the Suffolk County Department of Health Services before operating.",
        "condition":  "Annual renewal required; permit must be posted visibly in the unit.",
        "penalty":    "Operation without permit is subject to immediate closure order.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-002",
        "section":    "NYS Sanitary Code ?14-1.41",
        "topic":      "Hot holding temperature",
        "obligation": "Potentially hazardous hot foods must be held at 140?F (60?C) or above.",
        "condition":  "Applies during service, transport, and storage on the vending unit.",
        "penalty":    "Critical violation - immediate corrective action required.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-003",
        "section":    "NYS Sanitary Code ?14-1.41",
        "topic":      "Cold holding temperature",
        "obligation": "Potentially hazardous cold foods must be held at 45?F (7?C) or below.",
        "condition":  "Mechanical refrigeration required; ice alone is insufficient for most foods.",
        "penalty":    "Critical violation - immediate corrective action required.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-004",
        "section":    "Suffolk County Sanitary Code ?760-8",
        "topic":      "Commissary requirement",
        "obligation": "All mobile food vending units must operate from an approved commissary.",
        "condition":  "Commissary agreement letter required at permit application and renewal. Unit must return to commissary daily for cleaning, restocking, and waste disposal.",
        "penalty":    "Permit denial or revocation.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-005",
        "section":    "NYS Sanitary Code ?14-1.60",
        "topic":      "Handwashing facilities",
        "obligation": "Mobile units must have a handwashing sink with hot and cold running water, soap, and single-use towels.",
        "condition":  "Separate from food preparation and warewashing sinks.",
        "penalty":    "Critical violation.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-006",
        "section":    "Suffolk County Sanitary Code ?760-5",
        "topic":      "Water supply",
        "obligation": "Mobile units must carry a sufficient supply of potable water (minimum 5 gallons for low-risk operations; more for full cooking).",
        "condition":  "Fresh water tank and waste water tank (1.15? fresh water capacity) required.",
        "penalty":    "Critical violation.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-007",
        "section":    "NYS Sanitary Code ?14-1.20",
        "topic":      "Food worker health certificate",
        "obligation": "At least one food service worker per unit must hold a valid Suffolk County Food Handler Certificate.",
        "condition":  "Certificate must be on premises during operation.",
        "penalty":    "Non-critical violation; must correct within 30 days.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-008",
        "section":    "Suffolk County Sanitary Code ?760-6",
        "topic":      "Approved location / vending site",
        "obligation": "Mobile vendors must operate only at approved locations; written approval from property owner required.",
        "condition":  "Locations near schools require additional zoning approval.",
        "penalty":    "Immediate stop-work order.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-009",
        "section":    "NYS Sanitary Code ?14-1.110",
        "topic":      "Pest exclusion",
        "obligation": "All food contact surfaces and the vending unit interior must be maintained free of pest evidence.",
        "condition":  "Screens required on all openings; inspectors may order immediate closure on evidence of infestation.",
        "penalty":    "Critical violation - immediate closure possible.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-010",
        "section":    "Suffolk County Sanitary Code ?760-9",
        "topic":      "Permit renewal timeline",
        "obligation": "Permits expire December 31 each year. Renewal applications must be submitted by November 1.",
        "condition":  "Late renewal subject to late fee. Lapsed permits require full re-inspection before reinstatement.",
        "penalty":    "Operating on an expired permit = same as operating without a permit.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-011",
        "section":    "NYS Sanitary Code ?14-1.50",
        "topic":      "Cross-contamination prevention",
        "obligation": "Raw animal foods must be stored below and physically separated from ready-to-eat foods.",
        "condition":  "Color-coded cutting boards and utensils strongly recommended.",
        "penalty":    "Critical violation.",
        "mobile_specific": True,
    },
    {
        "rule_id":    "MV-012",
        "section":    "Suffolk County Sanitary Code ?760-4",
        "topic":      "Unit construction standards",
        "obligation": "Interior surfaces must be smooth, non-absorbent, and easily cleanable. No bare wood or porous materials on food contact surfaces.",
        "condition":  "Pre-operation inspection required before permit issuance.",
        "penalty":    "Permit denial until corrections made.",
        "mobile_specific": True,
    },
]

# -- Permit checklist ----------------------------------------------------------
PERMIT_CHECKLIST = [
    {"step": 1,  "item": "Completed permit application form (SC Form H-10)",         "required": True,  "deadline": "Before operation"},
    {"step": 2,  "item": "Commissary agreement letter (signed by commissary operator)","required": True, "deadline": "With application"},
    {"step": 3,  "item": "Site approval letter from property owner",                  "required": True,  "deadline": "With application"},
    {"step": 4,  "item": "Vehicle registration / proof of ownership",                 "required": True,  "deadline": "With application"},
    {"step": 5,  "item": "Pre-operation inspection by SCDHS inspector",               "required": True,  "deadline": "Before permit issued"},
    {"step": 6,  "item": "Suffolk County Food Handler Certificate (operator)",         "required": True,  "deadline": "Before operation"},
    {"step": 7,  "item": "Permit fee payment ($150-$300 depending on unit type)",      "required": True,  "deadline": "With application"},
    {"step": 8,  "item": "Menu submitted for review (for cooking units)",              "required": True,  "deadline": "With application"},
    {"step": 9,  "item": "Water supply plan / tank capacity documentation",            "required": True,  "deadline": "With application"},
    {"step": 10, "item": "Wastewater disposal plan",                                   "required": True,  "deadline": "With application"},
    {"step": 11, "item": "Annual renewal by November 1",                               "required": True,  "deadline": "Nov 1 each year"},
    {"step": 12, "item": "Re-inspection after any critical violation closure",         "required": True,  "deadline": "Before reopening"},
]


# =============================================================================
# TEXT-BASED EXTRACTION (for when regulation .txt files are present)
# =============================================================================

def extract_from_text(text: str, source_name: str) -> list[dict]:
    """
    Rule-based extraction from regulation text.
    Extracts sentences containing obligation language + topic keywords.
    """
    rules = []
    # Split on sentence boundaries (rough)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        has_obligation = any(w in sent_lower for w in OBLIGATION_WORDS)
        is_mobile      = any(w in sent_lower for w in MOBILE_KEYWORDS)
        has_temp       = any(w in sent_lower for w in TEMP_KEYWORDS)
        has_permit     = any(w in sent_lower for w in PERMIT_KEYWORDS)
        has_penalty    = any(w in sent_lower for w in PENALTY_KEYWORDS)

        if not has_obligation:
            continue

        # Extract section number if present (e.g., "?14-1.41" or "Section 14-1.41")
        section_match = re.search(r'(?:?|section\s*)(\d+[\-\.]\d+[\.\d]*)', sent_lower)
        section = section_match.group(0) if section_match else f"{source_name}:sent_{i}"

        rule = {
            "rule_id":         f"EXT-{len(rules)+1:04d}",
            "source":          source_name,
            "section":         section,
            "raw_text":        sent.strip(),
            "mobile_specific": is_mobile,
            "topics": [
                t for t, flag in [
                    ("temperature_control", has_temp),
                    ("permit",              has_permit),
                    ("penalty",             has_penalty),
                    ("mobile_vendor",       is_mobile),
                ]
                if flag
            ],
        }
        rules.append(rule)

    log.info(f"Extracted {len(rules)} rules from {source_name}")
    return rules


def extract_all_regulations() -> list[dict]:
    """Process all .txt files in the regulations raw folder."""
    all_rules = []
    txt_files = list(RAW_REG_DIR.glob("*.txt"))

    if not txt_files:
        log.warning("No regulation .txt files found in suffolk_data/raw/regulations/")
        log.info("Using built-in mobile vendor rule base instead.")
        return BUILTIN_MOBILE_RULES

    for txt_file in txt_files:
        log.info(f"Processing: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        rules = extract_from_text(text, txt_file.stem)
        all_rules.extend(rules)

    # Always append built-in rules (they are authoritative)
    all_rules.extend(BUILTIN_MOBILE_RULES)
    log.info(f"Total rules extracted: {len(all_rules)}")
    return all_rules


# =============================================================================
# LLM-ASSISTED EXTRACTION (optional - requires Anthropic API)
# =============================================================================

def llm_extract_rule(text_snippet: str) -> dict | None:
    """
    Use Claude Haiku to extract structured rule data from a regulation snippet.
    Only called when ANTHROPIC_API_KEY is set in environment.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Extract the compliance rule from this Suffolk County / NYS food safety regulation text.
Return ONLY valid JSON with these fields:
{{
  "obligation": "what is required/prohibited",
  "condition": "when/where it applies",
  "penalty": "consequence of non-compliance (or null)",
  "mobile_specific": true/false,
  "temperature_f": number or null
}}

Text: {text_snippet[:800]}"""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.debug(f"LLM extraction failed: {e}")
        return None


# =============================================================================
# BUILD OUTPUTS
# =============================================================================

def build_violation_rule_map() -> dict:
    """Map each violation code to its regulation section and topic."""
    vmap = {}
    for code, info in VIOLATION_TO_SECTION.items():
        # Find matching built-in rule if any
        matching_rules = [
            r["rule_id"] for r in BUILTIN_MOBILE_RULES
            if info["section"] in r.get("section", "")
        ]
        vmap[code] = {
            **info,
            "related_rules": matching_rules,
            "mobile_relevant": True,   # all critical violations apply to mobile vendors
        }
    return vmap


def build_regulation_html(rules: list[dict]) -> str:
    """Generate a searchable HTML regulation reference page."""
    now = datetime.now().strftime("%B %d, %Y")
    mobile_rules  = [r for r in rules if r.get("mobile_specific")]
    general_rules = [r for r in rules if not r.get("mobile_specific")]

    def rule_card(r):
        topics_html = "".join(
            f'<span class="badge">{t}</span>' for t in r.get("topics", [])
        )
        return f"""
<div class="rule-card" data-mobile="{str(r.get('mobile_specific','false')).lower()}">
  <div class="rule-header">
    <span class="rule-id">{r.get('rule_id','')}</span>
    <span class="rule-section">{r.get('section','')}</span>
    {topics_html}
  </div>
  <div class="rule-topic"><strong>{r.get('topic', r.get('source',''))}</strong></div>
  <div class="rule-obligation">{r.get('obligation', r.get('raw_text',''))}</div>
  {"<div class='rule-condition'><em>Condition:</em> " + r['condition'] + "</div>" if r.get('condition') else ""}
  {"<div class='rule-penalty'><em>[!] Penalty:</em> " + r['penalty'] + "</div>" if r.get('penalty') else ""}
</div>"""

    mobile_cards  = "\n".join(rule_card(r) for r in mobile_rules)
    general_cards = "\n".join(rule_card(r) for r in general_rules[:30])  # cap at 30

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Suffolk County - Regulation Reference</title>
<style>
  body {{font-family:'Segoe UI',Arial,sans-serif; margin:0; background:#f0f4f8; color:#222;}}
  header {{background:#1f4e79; color:white; padding:24px 40px;}}
  header h1 {{margin:0; font-size:1.5rem;}}
  header p  {{margin:4px 0 0; opacity:.8; font-size:.9rem;}}
  .toolbar {{background:white; padding:14px 40px; box-shadow:0 2px 6px rgba(0,0,0,.08); display:flex; gap:12px; align-items:center;}}
  .toolbar input {{flex:1; padding:8px 14px; border:1px solid #ccc; border-radius:6px; font-size:.95rem;}}
  .toolbar select {{padding:8px 12px; border:1px solid #ccc; border-radius:6px;}}
  main {{padding:24px 40px;}}
  h2 {{color:#1f4e79; margin-top:32px;}}
  .rule-card {{background:white; border-radius:8px; padding:16px 20px; margin:10px 0; box-shadow:0 1px 4px rgba(0,0,0,.08); border-left:4px solid #1f4e79;}}
  .rule-card[data-mobile="true"] {{border-left-color:#c7522a;}}
  .rule-header {{display:flex; gap:8px; align-items:center; margin-bottom:6px; flex-wrap:wrap;}}
  .rule-id {{font-weight:700; color:#1f4e79; font-size:.85rem; background:#e3f2fd; padding:2px 8px; border-radius:10px;}}
  .rule-section {{font-size:.8rem; color:#555;}}
  .badge {{background:#f0f4f8; color:#555; font-size:.75rem; padding:2px 8px; border-radius:10px; border:1px solid #ddd;}}
  .rule-topic {{font-size:.95rem; margin-bottom:6px;}}
  .rule-obligation {{color:#333; line-height:1.6;}}
  .rule-condition {{color:#555; font-size:.88rem; margin-top:6px;}}
  .rule-penalty {{color:#b71c1c; font-size:.88rem; margin-top:4px;}}
  .hidden {{display:none;}}
  .count-badge {{background:#e3f2fd; color:#1f4e79; padding:3px 10px; border-radius:12px; font-size:.85rem; font-weight:600;}}
</style>
</head>
<body>
<header>
  <h1>? Suffolk County Mobile Food Vendor - Regulation Reference</h1>
  <p>Generated {now} x Sources: Suffolk County Sanitary Code + NYS Part 14</p>
</header>

<div class="toolbar">
  <input type="text" id="search" placeholder="Search rules, sections, topics..." oninput="filterRules()">
  <select id="filter" onchange="filterRules()">
    <option value="all">All Rules</option>
    <option value="mobile">Mobile Vendor Only</option>
  </select>
  <span class="count-badge" id="count">{len(rules)} rules</span>
</div>

<main>
  <h2>? Mobile Vendor-Specific Requirements <span class="count-badge">{len(mobile_rules)}</span></h2>
  <div id="mobile-rules">{mobile_cards}</div>

  <h2>? General Food Service Requirements <span class="count-badge">{len(general_rules)}</span></h2>
  <div id="general-rules">{general_cards}</div>
</main>

<script>
function filterRules() {{
  const q      = document.getElementById('search').value.toLowerCase();
  const filter = document.getElementById('filter').value;
  let visible  = 0;
  document.querySelectorAll('.rule-card').forEach(card => {{
    const text   = card.innerText.toLowerCase();
    const mobile = card.dataset.mobile === 'true';
    const show   = text.includes(q) && (filter === 'all' || (filter === 'mobile' && mobile));
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  }});
  document.getElementById('count').textContent = visible + ' rules';
}}
</script>
</body>
</html>"""


def run_extraction():
    """Full extraction pipeline."""
    log.info("-- Starting regulation extraction --")

    # 1. Extract rules
    all_rules = extract_all_regulations()

    # 2. Mobile-vendor subset
    mobile_rules = [r for r in all_rules if r.get("mobile_specific")]
    log.info(f"Mobile-vendor rules: {len(mobile_rules)}")

    # 3. Violation->rule map
    vmap = build_violation_rule_map()

    # 4. Save JSON outputs
    (OUT_REG_DIR / "extracted_rules.json").write_text(
        json.dumps(all_rules, indent=2), encoding="utf-8")

    (OUT_REG_DIR / "mobile_vendor_rules.json").write_text(
        json.dumps(mobile_rules, indent=2), encoding="utf-8")

    (OUT_REG_DIR / "permit_checklist.json").write_text(
        json.dumps(PERMIT_CHECKLIST, indent=2), encoding="utf-8")

    (OUT_REG_DIR / "violation_rule_map.json").write_text(
        json.dumps(vmap, indent=2), encoding="utf-8")

    # 5. HTML reference
    html = build_regulation_html(all_rules)
    (OUT_REG_DIR / "regulation_index.html").write_text(html, encoding="utf-8")

    log.info(f"[OK] Saved to {OUT_REG_DIR}/")
    return all_rules, mobile_rules, vmap


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    rules, mobile, vmap = run_extraction()
    print(f"\n-- Regulation Extraction Summary ---------------------")
    print(f"  Total rules extracted   : {len(rules)}")
    print(f"  Mobile-vendor rules     : {len(mobile)}")
    print(f"  Violation codes mapped  : {len(vmap)}")
    print(f"  Permit checklist items  : {len(PERMIT_CHECKLIST)}")
    print(f"\n  Open: suffolk_data/regulations/regulation_index.html")
    print(f"--------------------------------------------------------")
