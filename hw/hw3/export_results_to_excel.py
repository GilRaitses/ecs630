#!/usr/bin/env python3
import json
import pandas as pd
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS_JSON = os.path.join(THIS_DIR, "results_week3.json")
TIMELINE_CSV = os.path.join(THIS_DIR, "timeline_week3.csv")
OUTPUT_XLSX = os.path.join(THIS_DIR, "results_week3.xlsx")


def main() -> None:
    # Load metrics
    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)
    metrics_items = sorted(metrics.items())
    df_metrics = pd.DataFrame(metrics_items, columns=["metric", "value"])

    # Load timeline
    df_timeline = pd.read_csv(TIMELINE_CSV)

    # Write workbook
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        df_timeline.to_excel(writer, sheet_name="Timeline", index=False)

    print(f"Wrote {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()

