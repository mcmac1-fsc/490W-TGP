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

---

## Normalization example

def normalize_vendor(text):
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

