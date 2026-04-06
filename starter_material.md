# Starter Material
This document contains reference code, schema definitions, and starter kits used during the development of the Compliance Intelligence prototype.
It is not part of the runnable system — it serves as supporting documentation for the project.

## 1. Embedding Starter Code
```import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load models
food_tok = AutoTokenizer.from_pretrained("nlpconnect/foodbert-base")
food_mod = AutoModel.from_pretrained("nlpconnect/foodbert-base")

legal_tok = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_mod = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

e5_tok = AutoTokenizer.from_pretrained("intfloat/e5-small")
e5_mod = AutoModel.from_pretrained("intfloat/e5-small")

def embed(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def blended_embedding(text):
    e_food = embed(food_mod, food_tok, text)
    e_legal = embed(legal_mod, legal_tok, text)
    e_general = embed(e5_mod, e5_tok, text)

    # Normalize
    e_food = e_food / np.linalg.norm(e_food)
    e_legal = e_legal / np.linalg.norm(e_legal)
    e_general = e_general / np.linalg.norm(e_general)

    # Weighted blend
    final = (
        0.50 * e_food +
        0.25 * e_legal +
        0.25 * e_general
    )

    return final / np.linalg.norm(final)
```

---

## 2. Model Notes

### FoodBERT (50%)
- understands ingredients  
- understands cooking methods  
- understands allergens  
- understands cuisines  
- understands food‑risk semantics  

### LegalBERT (25%)
- understands regulatory language  
- understands permit requirements  
- understands violation categories  
- understands legal phrasing  

### E5 / TF‑IDF (25%)
- stabilizes embeddings  
- handles slang, typos, multilingual text  
- provides general semantic grounding  


---

## 3. Streamlit Starter Kit (Laiba)

```import streamlit as st
import json

st.title("Vendor Intake Form")

foods = st.text_input("Foods served")
equipment = st.text_input("Equipment used")
event_location = st.text_input("Event location")
event_date = st.date_input("Event date")
permits = st.text_input("Permits (comma separated)")
staff_count = st.number_input("Staff count", min_value=0)

if st.button("Submit"):
    output = {
        "foods": foods.split(","),
        "equipment": equipment.split(","),
        "event_location": event_location,
        "event_date": str(event_date),
        "permits": [p.strip() for p in permits.split(",")],
        "staff_count": staff_count
    }
    st.json(output)
```

---

## 4. Regulatory Extraction Starter Kit (Esra)

### Useful Sources
- Suffolk County Open Data  
- NYS Food Service Establishment Inspection Data  
- NYS Sanitary Code PDFs  

### Prompts
- “Summarize all food safety rules from this PDF into bullet points.”  
- “Extract all rules related to temperature control, sanitation, allergens, and equipment.”  
- “Rewrite these rules in simple language for food vendors.”  
- “Convert these rules into a structured JSON-like format.”  


```{
  "temperature_control": [
    "Hot foods must be held at 140°F or above",
    "Cold foods must be held at 41°F or below"
  ],
  "permits": [
    "Temporary event permit required for public events",
    "Propane requires fire safety certification"
  ]
}
```

---

## 5. Data Ingestion Starter Kit (Aisha)

### Tasks
- Download the inspection CSV  
- Open in Excel  
- Identify columns  
- Clean obvious issues  
- Save a cleaned CSV  

### Useful Prompt
- “Clean this CSV: normalize violation categories, fix date formats, remove duplicates, and summarize the top 10 violation types.”  

```import pandas as pd

df = pd.read_csv("your_file.csv")
df['violation_category'] = df['violation_category'].str.lower().str.strip()
df['violation_category'].value_counts().head(10)
```

## 6. Logic Starter Kit (Craig)

Simple Scoring Logic
```
score = 100
if "meat" in foods and "thermometer" not in equipment:
    score -= 20
```

## 7. Example Tasks for Each Role

### Aisha — Data Ingestion
- Clean CSV  
- Normalize categories  
- Summarize top violations  

### Esra — Regulatory Extraction
- Extract rules from PDFs  
- Convert to JSON format  

### Laiba — UI
- Build Streamlit form  
- Output JSON in schema  

### Craig — Logic
- Normalize vendor descriptions  
- Build rule‑based scoring  
- Prototype compliance output  


## End of Starter Material
This file documents the exploratory and planning code used during early development.
It is not part of the final runnable system but is included for completeness and transparency.


