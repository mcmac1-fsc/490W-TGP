import gradio as gr
import time

suffolk_text = """
Suffolk County Code:
Cold PHF must be maintained at 41°F or lower.
Unless otherwise approved by the Department.
"""

nys_text = """
NYS 14-1.40:
Cold PHF must be maintained at 45°F or below,
except during necessary preparation or a limited
service period not exceeding 2 hours.
"""

def extract_rules(text):
    rules = {}
    if "41" in text:
        rules["cold_temp"] = 41
    if "45" in text:
        rules["cold_temp"] = 45
    if "Unless otherwise approved" in text:
        rules["inspector_override"] = True
    if "2 hours" in text:
        rules["service_window"] = "2h"
    return rules

def detect_ambiguity(rules_a, rules_b):
    alerts = []
    if rules_a.get("cold_temp") != rules_b.get("cold_temp"):
        alerts.append("Conflicting cold-holding thresholds (41°F vs 45°F).")
    if rules_a.get("inspector_override") or rules_b.get("inspector_override"):
        alerts.append("Inspector override clause detected.")
    
    if alerts:
        return True, alerts
    return False, []

def build_popup(alerts):
    alert_list = "<br>".join(f"• {a}" for a in alerts)
    return f"""
    <div class="popup-overlay" id="popup-overlay">
      <div class="popup-card">
        <button class="popup-close"
                onclick="document.getElementById('popup-overlay').style.display='none'">✕</button>
        <h3 style="margin-top:0;margin-bottom:12px;color:#0f172a;">Ambiguity Detected</h3>
        <div style='background:#fff3cd;border-left:6px solid #ffecb5;
                    padding:14px 16px;border-radius:6px;
                    font-family:Inter, sans-serif;font-size:14px;
                    line-height:1.45;color:#856404;'>
          <b>The system found conflicting regulatory requirements:</b><br><br>
          {alert_list}<br><br>
          <b>Action:</b> Human review required before rule extraction.
        </div>
        <div style="margin-top:10px;text-align:right">
          <button onclick="document.getElementById('popup-overlay').style.display='none'"
                  style="padding:8px 12px;border-radius:8px;border:none;
                         background:#0f172a;color:white;font-weight:600;cursor:pointer;">
            Close
          </button>
        </div>
      </div>
    </div>
    """

css = """
.popup-overlay {
  position: fixed;
  left: 0; top: 0;
  width: 100%; height: 100%;
  background: rgba(15,23,42,0.55);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
}
.popup-card {
  background: white;
  border-radius: 12px;
  padding: 22px;
  width: min(650px, 92%);
  box-shadow: 0 12px 40px rgba(2,6,23,0.35);
  position: relative;
  max-height: 80vh;
  overflow: auto;
}
.popup-close {
  position: absolute;
  right: 14px;
  top: 14px;
  border: none;
  background: transparent;
  font-size: 20px;
  color: #374151;
  cursor: pointer;
}
"""

def run_check():
    time.sleep(1.0)
    rules_s = extract_rules(suffolk_text)
    rules_n = extract_rules(nys_text)
    ambiguous, alerts = detect_ambiguity(rules_s, rules_n)
    
    if ambiguous:
        return build_popup(alerts)
    else:
        return """
        <div style='background:#d4edda; border-left:6px solid #c3e6cb; 
                    padding:14px 16px; border-radius:6px; color:#155724; 
                    font-family:Inter, sans-serif; font-size:14px; margin-top:15px;'>
            <b>No ambiguity detected.</b> All loaded regulations are aligned.
        </div>
        """

with gr.Blocks(css=css, title="Regulatory Ambiguity Demo") as demo:
    gr.Markdown("### Compare Suffolk County vs NYS Regulations")
    btn = gr.Button("Run Ambiguity Check", variant="primary")
    popup = gr.HTML(value="", elem_id="popup_html")
    btn.click(run_check, outputs=popup)

if __name__ == "__main__":
    demo.launch(share=True)
