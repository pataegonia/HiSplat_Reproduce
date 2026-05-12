"""Extract training-set metrics (last N train steps averaged) from Phase2 logs.

Outputs plots/train_metrics.csv and plots/train_metrics_table.png comparing
refine_out vs raw_gaussians on the TRAINING dataset.
"""
import math
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "MSH_log"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

LAMBDAS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
LAST_N = 500  # average last N training steps

# File selection: for each (target, lmd), pick a log file.
def find_refine_log(lmd):
    # earliest 5 lambdas use legacy naming "phase2_lmd{lmd}_*"
    candidates = sorted(LOG_DIR.glob(f"refine_out_phase2_lmd{lmd}_*.out"))
    if not candidates:
        candidates = sorted(LOG_DIR.glob(f"phase2_lmd{lmd}_*.out"))
        if not candidates:
            legacy = LOG_DIR / f"phase2_lmd{lmd}.out"
            if legacy.exists():
                return legacy
    return candidates[-1] if candidates else None  # latest jobid


def find_raw_log(lmd):
    candidates = sorted(LOG_DIR.glob(f"raw_gaussians_phase2_lmd{lmd}_*.out"))
    return candidates[-1] if candidates else None


KV_RE = re.compile(r"(\w+)=([0-9.eE+\-]+)")


def parse_last_n(path: Path, n: int):
    """Return dict of averaged metric_key -> value for last n train steps."""
    if path is None or not path.exists():
        return None
    lines = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("train step"):
                lines.append(line)
    if not lines:
        return None
    tail = lines[-n:]
    sums = defaultdict(float)
    counts = defaultdict(int)
    for ln in tail:
        for k, v in KV_RE.findall(ln):
            if k in ("mse_0", "mse_1", "mse_2",
                     "lpips_0", "lpips_1", "lpips_2",
                     "bpp_0", "bpp_1", "bpp_2"):
                try:
                    sums[k] += float(v)
                    counts[k] += 1
                except ValueError:
                    pass
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def mse_to_psnr(mse):
    if mse <= 0 or mse != mse:
        return float("nan")
    return -10.0 * math.log10(mse)


def collect_all():
    rows = []  # (target, lmd, stage, psnr, lpips, bpp)
    for lmd in LAMBDAS:
        for target, finder in [("refine_out", find_refine_log),
                               ("raw_gaussians", find_raw_log)]:
            path = finder(lmd)
            metrics = parse_last_n(path, LAST_N)
            if metrics is None:
                continue
            for stage in (0, 1, 2):
                mse = metrics.get(f"mse_{stage}")
                lp = metrics.get(f"lpips_{stage}")
                bpp = metrics.get(f"bpp_{stage}")
                if mse is None:
                    continue
                rows.append({
                    "target": target, "lambda": lmd, "stage": stage,
                    "train_psnr": mse_to_psnr(mse),
                    "train_lpips": lp if lp is not None else float("nan"),
                    "train_bpp": bpp if bpp is not None else float("nan"),
                    "log": path.name,
                })
    return rows


def save_csv(rows):
    import csv
    p = OUT_DIR / "train_metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "lambda", "stage", "train_psnr",
                    "train_lpips", "train_bpp", "log"])
        for r in rows:
            w.writerow([r["target"], f"{r['lambda']:g}", f"stage{r['stage']}",
                        f"{r['train_psnr']:.3f}",
                        f"{r['train_lpips']:.5f}",
                        f"{r['train_bpp']:.5f}",
                        r["log"]])
    print(f"Saved {p}")


def save_table_image(rows):
    """Three subtables (stage0/1/2). Each row: λ | refine PSNR | refine LPIPS | refine BPP |
       raw PSNR | raw LPIPS | raw BPP"""
    by_stage = {0: {}, 1: {}, 2: {}}
    for r in rows:
        by_stage[r["stage"]].setdefault(r["lambda"], {})[r["target"]] = r

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    for ax, stage in zip(axes, (0, 1, 2)):
        ax.axis("off")
        ax.set_title(f"stage{stage}  (train set, avg of last {LAST_N} steps)",
                     loc="left", fontsize=11, pad=8)
        header = ["λ",
                  "refine PSNR", "refine LPIPS", "refine BPP",
                  "raw PSNR", "raw LPIPS", "raw BPP"]
        cell_text = []
        cell_colors = []
        for lmd in LAMBDAS:
            entry = by_stage[stage].get(lmd, {})
            rf = entry.get("refine_out")
            rw = entry.get("raw_gaussians")
            row = [f"{lmd:g}"]
            for r in (rf, rw):
                if r is None:
                    row += ["-", "-", "-"]
                else:
                    row += [f"{r['train_psnr']:.3f}",
                            f"{r['train_lpips']:.4f}",
                            f"{r['train_bpp']:.4f}"]
            cell_text.append(row)
            cell_colors.append(["#ffffff",
                                "#d7e7ff", "#d7e7ff", "#d7e7ff",
                                "#ffe3c5", "#ffe3c5", "#ffe3c5"])
        tbl = ax.table(cellText=cell_text, colLabels=header,
                       cellColours=cell_colors,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)

    fig.suptitle(
        "Training-set metrics: refine_out vs raw_gaussians "
        f"(Phase2, last {LAST_N} train steps averaged)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT_DIR / "train_metrics_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)


def save_rd_plot(rows):
    """Train-set PSNR vs BPP, per-stage subplots."""
    by_stage = {0: {"refine_out": [], "raw_gaussians": []},
                1: {"refine_out": [], "raw_gaussians": []},
                2: {"refine_out": [], "raw_gaussians": []}}
    for r in rows:
        by_stage[r["stage"]][r["target"]].append(
            (r["lambda"], r["train_bpp"], r["train_psnr"]))
    for s in by_stage:
        for t in by_stage[s]:
            by_stage[s][t].sort(key=lambda x: x[1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, stage in zip(axes, (0, 1, 2)):
        for target, color, marker, label in [
            ("refine_out",    "tab:blue",   "o", "MSH refine_out"),
            ("raw_gaussians", "tab:orange", "s", "MSH raw_gaussians"),
        ]:
            pts = by_stage[stage][target]
            if not pts:
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, color=color, marker=marker,
                    linewidth=2.0, label=label)
            for lmd, x, y in pts:
                ax.annotate(f"λ={lmd:g}", xy=(x, y), xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=7, color=color, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("BPP (train, last 500 steps)")
        if stage == 0:
            ax.set_ylabel("PSNR (dB, derived from train MSE)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"stage{stage}")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Training-set RD: PSNR vs BPP  (refine_out vs raw_gaussians)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT_DIR / "train_rd_psnr_vs_bpp.png"
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)


def main():
    rows = collect_all()
    if not rows:
        print("No rows extracted.")
        return
    save_csv(rows)
    save_table_image(rows)
    save_rd_plot(rows)


if __name__ == "__main__":
    main()
