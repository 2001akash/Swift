# 🚚 SWIFT Assignment – Transit Performance Analysis

This project analyzes shipment tracking data to evaluate **transit performance** and **delivery efficiency** using a provided JSON dataset.

---

## 📁 Files
transit_performance_analysis.py # Main script
Swift Assignment 4 - Dataset (1).json # Input dataset (place in same folder)
transit_performance_detailed.csv # Shipment-level output
transit_performance_summary.csv # Overall summary output
transit_service_comparison.csv # Optional service-type summary

yaml
Copy code

---

## ⚙️ How to Run

1. Place the JSON file in the **same folder** as the script.
2. Install dependencies:
   ```bash
   pip install pandas numpy
Run:

bash
Copy code
python transit_performance_analysis.py
📊 Output Files
File	Description
transit_performance_detailed.csv	Shipment-level detailed metrics
transit_performance_summary.csv	Overall transit performance summary
transit_service_comparison.csv	Service-type comparison summary

🧮 Metrics Calculated
Total transit time (in hours)

Facilities visited

In-transit events

Average hours per facility

Out-for-delivery attempts

First-attempt delivery status

Service type (Express/Standard)

✅ Features
Handles missing/null data

Supports multiple timestamp formats ($numberLong, ISO)

Removes duplicate or invalid events

Exports clean, analysis-ready CSVs
