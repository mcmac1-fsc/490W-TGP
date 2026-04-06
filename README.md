# Compliance Intelligence Prototype

This repository contains the technical deliverables for our AIM 490W Term Group Project.  
The project simulates an AI-driven compliance assistant for mobile food vendors.

---

## Project Schema

All modules output data in the following structure:

```json
{
  "foods": [],
  "equipment": [],
  "allergens": [],
  "event_location": "",
  "event_date": "",
  "permits": [],
  "staff_count": 0
}
```
---

## Normalization example

```def normalize_vendor(text):
    return {
        "foods": [w.strip() for w in text.split(",")],
        "equipment": [],
        "allergens": [],
        "event_location": "",
        "event_date": "",
        "permits": [],
        "staff_count": 0
    }

normalize_vendor("tacos, birria, rice")
```

-----


## Technical Execution & Modules

This repository utilizes a modular multi-agent structure to simulate advanced compliance extraction and ambiguity checking. The scripts are located in the `modules/` directory and can be executed via any Python environment with Gradio installed.

### Core Modules:
1. **`allergy_notice_protocol.py`**
   * **Purpose:** Simulates document-level verification to ensure vendor menus comply with mandatory notice requirements. 
   * **Logic:** Triggers a targeted front-end alert if specific regulatory text mandates are not met on customer-facing assets.

2. **`nlp_ambiguity.py`**
   * **Purpose:** Acts as a simulated Natural Language Processing (NLP) pipeline checking for conflicting regulatory text.
   * **Logic:** Scans unstructured paragraphs for conflicting numerical thresholds (e.g., conflicting holding temperatures) and holds the operation for human-in-the-loop review.

3. **`jurisdiction_ref.py`**
   * **Purpose:** Conducts real cross-referencing between disjointed local and state codes.
   * **Logic:** Evaluates Suffolk County sanitation codes against New York State (NYS) 14-1.40 codes, detecting extraction conflicts and establishing a strict compliance ceiling.

## Local Knowledge Base
The `suffolk_health/` directory contains the source PDF data files downloaded from the official Suffolk County Health Services portal. These are used to ground the retrieval mechanisms and prevent LLM hallucinations.
