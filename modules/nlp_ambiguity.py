import gradio as gr
import time

ambiguous_text = """
"Food must be kept at a safe temperature, ideally below 45°F,
unless otherwise permitted by the inspector. In some cases,
cold holding may be considered acceptable at 50°F."
"""

def build_ambiguity_popup():
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
          <b>The NLP pipeline flagged unclear regulatory language:</b><br><br>
          <i>{ambiguous_text}</i><br><br>
          <b>Reason:</b> Conflicting temperature thresholds (45°F vs 50°F).<br>
          <b>Action:</b> Requires human review before rule extraction.
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

def trigger_popup():
    time.sleep(1.0)
    return build_ambiguity_popup()

with gr.Blocks(css=css, title="Ambiguity Detection Demo") as demo:
    gr.Markdown("### Click to Simulate an Ambiguity Alert")
    btn = gr.Button("Run Ambiguity Check", variant="primary")
    popup = gr.HTML(value="", elem_id="popup_html")
    btn.click(trigger_popup, outputs=popup)

if __name__ == "__main__":
    demo.launch(share=True)
