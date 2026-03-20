"""
Generate a publication-quality comparison table from JSON results.

Reads JSON outputs from compute_population_metrics.py and produces:
  1. A formatted table printed to terminal
  2. A PNG image of the table (for slides)
  3. A LaTeX table (for papers)
  4. A CSV file (for spreadsheets)

Usage:
    # Single dataset
    python generate_comparison_table.py \
        --inputs results_stock.json \
        --outdir ./tables

    # Multiple datasets
    python generate_comparison_table.py \
        --inputs results_stock.json results_energy.json results_sine.json \
        --outdir ./tables
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "discriminative": "Discriminative ↓",
    "predictive": "Predictive ↓",
    "context_fid": "Context-FID ↓",
    "cross_correlation": "Cross-Corr ↓",
    "vds": "VDS ↓",
    "fdds": "FDDS ↓",
}

METRIC_ORDER = ["discriminative", "predictive", "context_fid", "cross_correlation", "vds", "fdds"]


def load_results(paths):
    """Load and merge JSON result files."""
    all_data = []
    for p in paths:
        with open(p) as f:
            all_data.append(json.load(f))
    return all_data


def extract_scores(result, method_key):
    """Extract mean ± std for each metric from a result dict."""
    method = result.get(method_key, {})
    scores = {}
    for m in METRIC_ORDER:
        entry = method.get(m)
        if entry is None:
            scores[m] = ("—", None)
        elif isinstance(entry, dict) and "scores" in entry:
            arr = np.array(entry["scores"], dtype=float)
            scores[m] = (f"{arr.mean():.4f} ± {arr.std():.4f}", arr.mean())
        elif isinstance(entry, list):
            arr = np.array(entry, dtype=float)
            scores[m] = (f"{arr.mean():.4f} ± {arr.std():.4f}", arr.mean())
        else:
            scores[m] = ("—", None)
    return scores


def generate_png(datasets, outdir):
    """Render a table as a PNG image suitable for presentation slides."""
    # Build table data
    col_labels = ["Dataset", "Method"] + [METRIC_LABELS[m] for m in METRIC_ORDER]
    rows = []

    for data in datasets:
        ds_name = data.get("dataset", "?").upper()
        for method_key, method_label in [("diffusion_ts", "Diffusion-TS"), ("padts", "PaD-TS")]:
            if method_key not in data:
                continue
            scores = extract_scores(data, method_key)
            row = [ds_name if method_label == "Diffusion-TS" else "", method_label]
            row += [scores[m][0] for m in METRIC_ORDER]
            rows.append(row)

    if not rows:
        print("No data to render.")
        return

    fig, ax = plt.subplots(figsize=(18, 1.5 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors and bold best scores
    for i in range(len(rows)):
        color = "#f8f9fa" if i % 4 < 2 else "#e9ecef"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    # Bold the best (lowest) score per metric per dataset
    for ds_idx in range(0, len(rows), 2):
        if ds_idx + 1 >= len(rows):
            break
        for m_idx, m in enumerate(METRIC_ORDER):
            col = m_idx + 2
            val1 = extract_scores(datasets[ds_idx // 2], "diffusion_ts").get(m, ("—", None))[1]
            val2 = extract_scores(datasets[ds_idx // 2], "padts").get(m, ("—", None))[1]
            if val1 is not None and val2 is not None:
                if val1 <= val2:
                    table[ds_idx + 1, col].set_text_props(fontweight="bold")
                else:
                    table[ds_idx + 2, col].set_text_props(fontweight="bold")

    plt.title("Diffusion-TS vs PaD-TS: Quantitative Comparison", fontsize=14,
              fontweight="bold", pad=20)
    path = os.path.join(outdir, "comparison_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  PNG: {path}")


def generate_latex(datasets, outdir):
    """Generate a LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison of Diffusion-TS and PaD-TS.}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    ncols = len(METRIC_ORDER) + 2
    lines.append(r"\begin{tabular}{ll" + "c" * len(METRIC_ORDER) + "}")
    lines.append(r"\toprule")

    header = "Dataset & Method & " + " & ".join(
        METRIC_LABELS[m].replace("↓", r"$\downarrow$") for m in METRIC_ORDER
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for data in datasets:
        ds_name = data.get("dataset", "?")
        first = True
        for method_key, method_label in [("diffusion_ts", "Diffusion-TS"), ("padts", "PaD-TS")]:
            if method_key not in data:
                continue
            scores = extract_scores(data, method_key)
            ds_col = ds_name.capitalize() if first else ""
            first = False
            vals = []
            for m in METRIC_ORDER:
                txt, _ = scores[m]
                if txt == "—":
                    vals.append("—")
                else:
                    vals.append(f"${txt.replace('±', r'\pm')}$")
            line = f"{ds_col} & {method_label} & " + " & ".join(vals) + r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    path = os.path.join(outdir, "comparison_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX: {path}")


def generate_csv(datasets, outdir):
    """Generate a CSV file."""
    lines = ["Dataset,Method," + ",".join(METRIC_LABELS[m] for m in METRIC_ORDER)]
    for data in datasets:
        ds_name = data.get("dataset", "?")
        for method_key, method_label in [("diffusion_ts", "Diffusion-TS"), ("padts", "PaD-TS")]:
            if method_key not in data:
                continue
            scores = extract_scores(data, method_key)
            vals = [scores[m][0] for m in METRIC_ORDER]
            lines.append(f"{ds_name},{method_label}," + ",".join(vals))

    path = os.path.join(outdir, "comparison_table.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  CSV: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables from metric JSON results")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON result files from compute_population_metrics.py")
    parser.add_argument("--outdir", default="./tables", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    datasets = load_results(args.inputs)

    print("Generating comparison outputs...")
    generate_png(datasets, args.outdir)
    generate_latex(datasets, args.outdir)
    generate_csv(datasets, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
