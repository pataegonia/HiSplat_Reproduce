"""Per-stage RD-Curve: HiSplat+MSH (refine_out / raw_gaussians) vs HiSplat+FCGS vs Original HiSplat.

Each metric produces a figure with 3 subplots, one per stage (stage0/1/2).
Lambda values are annotated on every point.
Original HiSplat baseline is drawn as a dashed horizontal line per stage.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "MSH_log"
FCGS_CSV = PROJECT_ROOT / "eval_rdcurve.csv"
OUT_DIR = PROJECT_ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

# Per-stage Original HiSplat baseline (no compression).
# Source: slurm-326800.out (full Re10K evaluation set).
BASELINE_PER_STAGE = {
    "stage0": {"psnr": 26.137455289290944, "ssim": 0.8582867190016248, "lpips": 0.14100903031066092},
    "stage1": {"psnr": 26.990322282441720, "ssim": 0.8785342715351457, "lpips": 0.12036917450704580},
    "stage2": {"psnr": 27.194047111031594, "ssim": 0.8817399117196825, "lpips": 0.11695834854114219},
}

STAGES = ["stage0", "stage1", "stage2"]
METRIC_KEYS = ["psnr", "ssim", "lpips"]

LAMBDAS_LOW = [0.0001, 0.0005, 0.001, 0.005, 0.01]
LAMBDAS_HIGH = [0.05, 0.1, 0.5, 1.0]
LAMBDAS = LAMBDAS_LOW + LAMBDAS_HIGH

FILE_RE = re.compile(r"test_(refine_out|raw_gaussians)_lmd([0-9.]+)_(\d+)\.out")


def parse_scores(path: Path) -> dict:
    """Parse all per-stage + total metrics from an MSH test log."""
    scores = {}
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            key = parts[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            scores[key] = val
    return scores


def collect_msh():
    """Return {target: {lambda: scores}}. scores keys include
    actual_kb_stage{i}, psnr_stage{i}, ssim_stage{i}, lpips_stage{i}."""
    data = {"refine_out": {}, "raw_gaussians": {}}
    for p in sorted(LOG_DIR.glob("test_*.out")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        target, lmd_str, jobid = m.group(1), m.group(2), int(m.group(3))
        try:
            lmd = float(lmd_str)
        except ValueError:
            continue
        if lmd not in LAMBDAS:
            continue
        s = parse_scores(p)
        if "actual_kb_stage0" not in s or "psnr_stage0" not in s:
            continue
        existing = data[target].get(lmd)
        if existing is None or existing["jobid"] < jobid:
            s["jobid"] = jobid
            data[target][lmd] = s
    return data


def collect_fcgs():
    """Return {stage: [(lambda, kb_avg, psnr, ssim, lpips), ...]} from the FCGS csv."""
    if not FCGS_CSV.exists():
        print(f"[warn] FCGS csv not found at {FCGS_CSV}")
        return {s: [] for s in STAGES}

    # per (stage, lambda, sample)
    agg = defaultdict(lambda: defaultdict(dict))  # [stage][lmd][sample] -> metrics
    with FCGS_CSV.open() as f:
        for row in csv.DictReader(f):
            stage = row["stage"]
            if stage not in STAGES:
                continue
            lmd = float(row["lambda"])
            sample = row["sample"]
            agg[stage][lmd][sample] = {
                "kb": float(row["comp_mb"]) * 1024.0,
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
            }

    # Per-stage average (non-cumulative) KB + quality.
    per_stage = {s: {} for s in STAGES}  # {stage: {lmd: (kb, psnr, ssim, lpips)}}
    for stage in STAGES:
        for lmd in sorted(agg[stage].keys()):
            samples = agg[stage][lmd]
            n = len(samples)
            if n == 0:
                continue
            kb = sum(v["kb"] for v in samples.values()) / n
            psnr = sum(v["psnr"] for v in samples.values()) / n
            ssim = sum(v["ssim"] for v in samples.values()) / n
            lpips = sum(v["lpips"] for v in samples.values()) / n
            per_stage[stage][lmd] = (kb, psnr, ssim, lpips)

    # Cumulative KB: stage_i = sum of stage0..i KB, quality stays per-stage.
    out = {s: [] for s in STAGES}
    for i, stage in enumerate(STAGES):
        for lmd, (_, psnr, ssim, lpips) in per_stage[stage].items():
            kb_sum = 0.0
            ok = True
            for k in range(i + 1):
                entry = per_stage[STAGES[k]].get(lmd)
                if entry is None:
                    ok = False
                    break
                kb_sum += entry[0]
            if not ok:
                continue
            out[stage].append((lmd, kb_sum, psnr, ssim, lpips))
        out[stage].sort(key=lambda t: t[1])
    return out


def msh_stage_points(rows, stage_idx, metric):
    """rows: sorted [(lambda, scores), ...]. Returns list of (lambda, kb, metric_val).
    KB is CUMULATIVE: stage_i KB = sum(actual_kb_stage0..stage_i), since at inference
    time all three stages' bitstreams are required to reach stage_i's quality."""
    pts = []
    for lmd, s in rows:
        m_key = f"{metric}_stage{stage_idx}"
        if m_key not in s:
            continue
        kb_sum = 0.0
        missing = False
        for k in range(stage_idx + 1):
            v = s.get(f"actual_kb_stage{k}")
            if v is None:
                missing = True
                break
            kb_sum += v
        if missing:
            continue
        val = s[m_key]
        if val != val:  # NaN (e.g. ssim_stage2)
            continue
        pts.append((lmd, kb_sum, val))
    pts.sort(key=lambda t: t[1])
    return pts


def annotate(ax, lmd, x, y, color, dx=4, dy=4):
    ax.annotate(f"λ={lmd:g}", xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=7, color=color, alpha=0.85)


STAGE_STYLE = {
    0: {"linestyle": ":",  "marker": "o", "alpha": 0.55, "lw": 1.4},
    1: {"linestyle": "--", "marker": "s", "alpha": 0.75, "lw": 1.6},
    2: {"linestyle": "-",  "marker": "D", "alpha": 1.00, "lw": 2.0},
}
BASELINE_COLORS = {"stage0": "#f4a3a3", "stage1": "#e26969", "stage2": "#b30000"}


def draw(metric, ylabel, out_name, lower_better=False):
    data = collect_msh()
    refine = sorted(data["refine_out"].items())
    raw = sorted(data["raw_gaussians"].items())
    fcgs = collect_fcgs()

    # Two panels sharing Y with a visual break between them.
    # Left panel: MSH region (~0.02 – 150 KB).
    # Right panel: FCGS region (~1500 – 5000 KB).
    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True, figsize=(13, 7),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.04},
    )

    def plot_msh(ax, rows, color, method_label, annotate_lmd):
        for i in range(3):
            pts = msh_stage_points(rows, i, metric)
            if not pts:
                continue
            style = STAGE_STYLE[i]
            xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
            ax.plot(xs, ys, color=color,
                    linestyle=style["linestyle"], marker=style["marker"],
                    alpha=style["alpha"], linewidth=style["lw"],
                    label=f"{method_label} (stage{i})")
            if i == 2 and annotate_lmd:
                for lmd, x, y in pts:
                    annotate(ax, lmd, x, y, color)

    # MSH on left panel only.
    plot_msh(axL, refine, "tab:blue",   "MSH refine_out",    annotate_lmd=True)
    plot_msh(axL, raw,    "tab:orange", "MSH raw_gaussians", annotate_lmd=True)

    # FCGS stage2 on right panel only.
    fpts = fcgs.get("stage2", [])
    m_idx = {"psnr": 2, "ssim": 3, "lpips": 4}[metric]
    fpts_clean = [(p[0], p[1], p[m_idx]) for p in fpts if p[m_idx] == p[m_idx]]
    fpts_clean.sort(key=lambda t: t[1])
    if fpts_clean:
        xs = [p[1] for p in fpts_clean]; ys = [p[2] for p in fpts_clean]
        axR.plot(xs, ys, color="tab:green", linestyle="-", marker="^",
                 linewidth=2.0, label="FCGS (stage2)")
        for lmd, x, y in fpts_clean:
            annotate(axR, lmd, x, y, "tab:green", dx=4, dy=-10)

    # Baselines span both panels.
    for ax in (axL, axR):
        for stage in STAGES:
            base = BASELINE_PER_STAGE[stage][metric]
            ax.axhline(base, linestyle="--", color=BASELINE_COLORS[stage],
                       linewidth=1.2,
                       label=(f"Original HiSplat {stage} = {base:.3f}"
                              if ax is axL else None))

    # Tighten the MSH panel so points are more uniformly spaced.
    # Use evenly-spaced log ticks (each step ~×3.16).
    axL.set_xscale("log")
    axL.set_xlim(0.02, 200)
    left_ticks = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    axL.xaxis.set_major_locator(FixedLocator(left_ticks))
    axL.xaxis.set_minor_locator(NullLocator())
    axL.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: f"{v:g}"))

    axR.set_xscale("log")
    axR.set_xlim(1500, 5200)
    right_ticks = [2000, 3000, 4000, 5000]
    axR.xaxis.set_major_locator(FixedLocator(right_ticks))
    axR.xaxis.set_minor_locator(NullLocator())
    axR.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: f"{int(v)}"))

    # Hide the inner spines so the break is visible.
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Diagonal break marks on the inner edges.
    d = 0.012
    kw = dict(transform=axL.transAxes, color="k", clip_on=False, linewidth=1)
    axL.plot((1 - d, 1 + d), (-d, +d), **kw)
    axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw.update(transform=axR.transAxes)
    axR.plot((-d * 4, +d * 4), (-d, +d), **kw)
    axR.plot((-d * 4, +d * 4), (1 - d, 1 + d), **kw)

    axL.grid(True, which="both", alpha=0.3)
    axR.grid(True, which="both", alpha=0.3)
    axL.set_ylabel(ylabel)
    fig.supxlabel("Bitstream size (KB)", y=0.02, fontsize=11)

    arrow = "v lower is better" if lower_better else "^ higher is better"
    fig.suptitle(f"RD Curve -- {ylabel} ({arrow})", fontsize=13)

    # Single combined legend on the left panel.
    handles, labels = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    for h, l in zip(hR, lR):
        if l and l not in labels:
            handles.append(h); labels.append(l)
    axL.legend(handles, labels,
               loc="lower right" if not lower_better else "upper right",
               fontsize=7, ncol=2)

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = OUT_DIR / out_name
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def save_table_image(rows_by_stage, out_name, title):
    """Render a 3-subtable image (one per stage) using matplotlib."""
    stage_titles = {"stage0": "stage0",
                    "stage1": "stage1 (cumulative stage0+1)",
                    "stage2": "stage2 (cumulative stage0+1+2 = real usage)"}
    n_rows_total = sum(len(v) for v in rows_by_stage.values()) + 3 * 2  # header + title
    fig_h = max(4, 0.28 * n_rows_total + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, fig_h))
    for ax, stage in zip(axes, STAGES):
        ax.axis("off")
        header = ["method", "λ", "KB (cumulative)", "PSNR", "SSIM", "LPIPS"]
        rows = rows_by_stage[stage]
        if not rows:
            ax.set_title(stage_titles[stage], loc="left", fontsize=10)
            continue
        cell_text = []
        cell_colors = []
        color_map = {
            "MSH refine_out": "#d7e7ff",
            "MSH raw_gaussians": "#ffe3c5",
            "FCGS": "#d4f0d4",
            "Original HiSplat": "#ffd6d6",
        }
        for r in rows:
            method, lmd, kb, psnr, ssim, lpips = r
            cell_text.append([
                method,
                f"{lmd}" if lmd == "-" else f"{float(lmd):g}",
                "-" if kb == "-" else f"{float(kb):.3f}",
                "-" if psnr == "-" else f"{float(psnr):.3f}",
                "-" if ssim == "-" or ssim != ssim else (ssim if isinstance(ssim, str) else f"{ssim:.4f}"),
                "-" if lpips == "-" else (lpips if isinstance(lpips, str) else f"{float(lpips):.4f}"),
            ])
            cell_colors.append([color_map.get(method, "#ffffff")] * 6)
        tbl = ax.table(cellText=cell_text, colLabels=header,
                       cellColours=cell_colors, loc="center",
                       cellLoc="center", colLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.25)
        ax.set_title(stage_titles[stage], loc="left", fontsize=10, pad=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT_DIR / out_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def build_table_rows():
    """Return {stage: [(method, lambda, kb_cum, psnr, ssim, lpips), ...]} in plot order."""
    data = collect_msh()
    refine = sorted(data["refine_out"].items())
    raw = sorted(data["raw_gaussians"].items())
    fcgs = collect_fcgs()  # already cumulative

    out = {s: [] for s in STAGES}
    for i, stage in enumerate(STAGES):
        # MSH refine_out
        for lmd, s in refine:
            kb_cum = 0.0
            ok = True
            for k in range(i + 1):
                v = s.get(f"actual_kb_stage{k}")
                if v is None:
                    ok = False; break
                kb_cum += v
            if not ok:
                continue
            psnr = s.get(f"psnr_stage{i}")
            ssim = s.get(f"ssim_stage{i}", float("nan"))
            lpips = s.get(f"lpips_stage{i}", float("nan"))
            if psnr is None:
                continue
            out[stage].append(("MSH refine_out", lmd, kb_cum, psnr,
                               ssim if ssim == ssim else float("nan"),
                               lpips if lpips == lpips else float("nan")))
        # MSH raw_gaussians
        for lmd, s in raw:
            kb_cum = 0.0
            ok = True
            for k in range(i + 1):
                v = s.get(f"actual_kb_stage{k}")
                if v is None:
                    ok = False; break
                kb_cum += v
            if not ok:
                continue
            psnr = s.get(f"psnr_stage{i}")
            ssim = s.get(f"ssim_stage{i}", float("nan"))
            lpips = s.get(f"lpips_stage{i}", float("nan"))
            if psnr is None:
                continue
            out[stage].append(("MSH raw_gaussians", lmd, kb_cum, psnr,
                               ssim if ssim == ssim else float("nan"),
                               lpips if lpips == lpips else float("nan")))
        # FCGS
        for lmd, kb, psnr, ssim, lpips in fcgs.get(stage, []):
            out[stage].append(("FCGS", lmd, kb, psnr, ssim, lpips))
        # Original HiSplat baseline row
        b = BASELINE_PER_STAGE[stage]
        out[stage].append(("Original HiSplat", "-", "-", b["psnr"], b["ssim"], b["lpips"]))
    return out


def main():
    draw("psnr", "PSNR (dB)", "rd_curve_psnr_combined.png", lower_better=False)
    draw("ssim", "SSIM", "rd_curve_ssim_combined.png", lower_better=False)
    draw("lpips", "LPIPS", "rd_curve_lpips_combined.png", lower_better=True)

    rows_by_stage = build_table_rows()
    save_table_image(rows_by_stage, "rd_table.png",
                     "RD Results -- per stage (KB = cumulative, real-usage rate)")

    # Text tables per stage
    data = collect_msh()
    refine = sorted(data["refine_out"].items())
    raw = sorted(data["raw_gaussians"].items())
    fcgs = collect_fcgs()

    # CSV export: every (method, lambda, stage) row with KB + metrics.
    csv_path = OUT_DIR / "rd_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "lambda", "stage", "kb_cumulative", "psnr", "ssim", "lpips"])
        for name, rows in [("MSH_refine_out", refine), ("MSH_raw_gaussians", raw)]:
            for lmd, s in rows:
                kb_cum = 0.0
                for i in range(3):
                    v = s.get(f"actual_kb_stage{i}")
                    if v is None:
                        break
                    kb_cum += v
                    psnr = s.get(f"psnr_stage{i}")
                    ssim = s.get(f"ssim_stage{i}")
                    lpips = s.get(f"lpips_stage{i}")
                    if psnr is None:
                        continue
                    w.writerow([name, f"{lmd:g}", f"stage{i}",
                                f"{kb_cum:.6f}", f"{psnr:.6f}",
                                f"{ssim:.6f}" if ssim == ssim else "nan",
                                f"{lpips:.6f}" if lpips == lpips else "nan"])
            # total (all-stage) row per lambda
            for lmd, s in rows:
                kb = s.get("actual_kb_total")
                psnr = s.get("psnr")
                ssim = s.get("ssim")
                lpips = s.get("lpips")
                if kb is None or psnr is None:
                    continue
                w.writerow([name, f"{lmd:g}", "total",
                            f"{kb:.6f}", f"{psnr:.6f}",
                            f"{ssim:.6f}" if ssim == ssim else "nan",
                            f"{lpips:.6f}" if lpips == lpips else "nan"])
        for stage in STAGES:
            for lmd, kb, psnr, ssim, lpips in fcgs.get(stage, []):
                w.writerow(["FCGS", f"{lmd:g}", stage,
                            f"{kb:.6f}", f"{psnr:.6f}",
                            f"{ssim:.6f}", f"{lpips:.6f}"])
        for stage, b in BASELINE_PER_STAGE.items():
            w.writerow(["Original_HiSplat", "-", stage,
                        "-", f"{b['psnr']:.6f}",
                        f"{b['ssim']:.6f}", f"{b['lpips']:.6f}"])
    print(f"Saved {csv_path}")

    for i, stage in enumerate(STAGES):
        print(f"\n=== {stage} ===")
        print(f"{'method':<28}{'lambda':>10}{'KB':>12}{'PSNR':>10}{'LPIPS':>10}")
        for name, rows in [("MSH refine_out", refine), ("MSH raw_gaussians", raw)]:
            for lmd, s in rows:
                kb = s.get(f"actual_kb_stage{i}")
                psnr = s.get(f"psnr_stage{i}")
                lpips = s.get(f"lpips_stage{i}")
                if kb is None or psnr is None:
                    continue
                print(f"{name:<28}{lmd:>10g}{kb:>12.3f}{psnr:>10.3f}{lpips if lpips==lpips else float('nan'):>10.4f}")
        for lmd, kb, psnr, ssim, lpips in fcgs.get(stage, []):
            print(f"{'FCGS':<28}{lmd:>10g}{kb:>12.3f}{psnr:>10.3f}{lpips:>10.4f}")
        b = BASELINE_PER_STAGE[stage]
        print(f"{'Original HiSplat':<28}{'-':>10}{'-':>12}{b['psnr']:>10.3f}{b['lpips']:>10.4f}")


if __name__ == "__main__":
    main()
