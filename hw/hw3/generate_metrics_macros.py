#!/usr/bin/env python3
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(THIS_DIR, "results_week3.json")
TEX_PATH = os.path.join(THIS_DIR, "metrics_macros.tex")


def main() -> None:
    with open(JSON_PATH, "r") as f:
        m = json.load(f)
    def line(name, value):
        return f"\\newcommand{{\\{name}}}{{{value:.3f}}}\n"
    content = (
        line("AvgTsys", m["avg_total_time_in_system"]) +
        line("LSys", m["l_sys"]) +
        line("WqPaint", m["avg_wq_paint"]) +
        line("LqPaint", m["lq_paint"]) +
        line("UtilPaint", m["util_paint"]) +
        line("WqFinish", m["avg_wq_finish"]) +
        line("LqFinish", m["lq_finish"]) +
        line("UtilFinish", m["util_finish"]) +
        # Percent-scaled utilizations for plotting only
        line("UtilPaintPct", 100.0 * m["util_paint"]) +
        line("UtilFinishPct", 100.0 * m["util_finish"]) 
    )
    with open(TEX_PATH, "w") as f:
        f.write(content)
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
