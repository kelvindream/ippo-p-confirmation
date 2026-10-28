#!/usr/bin/env python3
"""Generate Figure 2 (confirmatory seed-level effects) and Figure 3
(exploratory 3x3 boundary results) for the empirical paper.

All plotted effect values are read at runtime from the result JSONs;
no effect value is hardcoded here. Run from anywhere:

    python3 paper/figures/make_figures.py
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "paper" / "figures"

# Okabe-Ito colorblind-safe palette (grayscale-distinguishable marker shapes added)
BLUE = "#0072B2"    # confirmatory seed points / H1HIGH
ORANGE = "#D55E00"  # H1SAT (negative)
GREEN = "#009E73"   # H1PROBE700
BLACK = "#000000"   # aggregate / reference markers

CONF_JSON = RESULTS / "ippo_prior_confirmation_v1" / "confirmation_summary.json"
SCALEUP_V1_JSON = RESULTS / "ippo_prior_scaleup_v1" / "scaleup_summary.json"
SCALEUP_V2_JSON = RESULTS / "ippo_prior_scaleup_v2" / "scaleup_v2_summary.json"
PROBE_DECISION_JSON = RESULTS / "ippo_prior_probe_700" / "decision_tree_locked.json"


def to_pct(log_ratio):
    """Percent transform used by the frozen analysis scripts:
    improvement_pct = (1 - exp(-log_ratio)) * 100 (U is lower-is-better)."""
    return (1.0 - np.exp(-np.asarray(log_ratio, dtype=np.float64))) * 100.0


def load_json(path):
    return json.loads(path.read_text())


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    })


def make_figure2(summary):
    """Horizontal dot/strip plot of the 10 per-training-seed effects plus the
    aggregate mean with its bootstrap 95% CI. No smoothing, no model fit."""
    primary = summary["primary"]
    seed_log_ratios = np.asarray(primary["seed_log_ratios"], dtype=np.float64)
    n_seeds = primary["n_seeds"]
    seed_ids = list(range(47, 47 + n_seeds))  # CONF_SEEDS = 47..56 (ordered as in JSON)
    seed_pct = to_pct(seed_log_ratios)
    mean_pct = float(primary["improvement_pct"])
    ci_lo, ci_hi = (float(v) for v in primary["bootstrap_ci95_pct"])
    p_value = float(primary["exact_two_sided_sign_flip_p"])
    seed_wins = int(primary["seed_wins"])

    assert len(seed_ids) == len(seed_pct) == n_seeds
    assert np.all(seed_pct > 0), "confirmatory seed effects must be positive"
    assert seed_wins == int(np.sum(seed_log_ratios > 0))

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ys = np.arange(n_seeds)[::-1]          # seed 47 at the top
    y_mean = n_seeds                        # aggregate row above the seeds

    ax.axvline(0.0, color=BLACK, lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(seed_pct, ys, "o", ms=4.2, color=BLUE, mec="none", zorder=3,
            clip_on=False)
    ax.errorbar([mean_pct], [y_mean],
                xerr=[[mean_pct - ci_lo], [ci_hi - mean_pct]],
                fmt="D", ms=4.6, color=BLACK, ecolor=BLACK,
                elinewidth=1.1, capsize=2.8, capthick=1.1, zorder=4,
                clip_on=False)

    ax.set_yticks([y_mean] + list(ys))
    ax.set_yticklabels(["Mean (95% CI)"] + [str(s) for s in seed_ids])
    ax.set_ylim(-0.7, y_mean + 0.7)

    span = max(ci_hi, seed_pct.max()) - min(0.0, seed_pct.min())
    ax.set_xlim(min(0.0, seed_pct.min()) - 0.06 * span,
                max(ci_hi, seed_pct.max()) + 0.08 * span)

    ax.set_xlabel("Improvement in demand service time, %\n"
                  "(U is lower-is-better; positive = U reduction)")
    ax.set_ylabel("Training seed")
    ax.tick_params(length=2.5, width=0.8)

    ax.set_title(f"n = {n_seeds} training seeds, {seed_wins}/{n_seeds} positive\n"
                 f"exact two-sided sign-flip p = {p_value:.6f}",
                 loc="right", fontsize=6.5, color="0.35", pad=4,
                 linespacing=1.4)

    fig.tight_layout(pad=0.4)
    out = {}
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"figure2_seed_effects.{ext}"
        fig.savefig(path, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)

    print("== Figure 2 (confirmatory, source:", CONF_JSON.relative_to(REPO_ROOT), ")")
    for sid, lp, pp in zip(seed_ids, seed_log_ratios, seed_pct):
        print(f"  seed {sid}: log_ratio {lp:+.6f} -> {pp:+.3f}%")
    print(f"  mean effect: {mean_pct:+.4f}%  bootstrap 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]%"
          f"  (resamples={summary['bootstrap_resamples']}, seed={summary['bootstrap_seed']})")
    print(f"  seed wins {seed_wins}/{n_seeds}, exact p = {p_value:.6f}")
    return out


def make_figure3(scaleup_v1, scaleup_v2, probe):
    """Discrete scatter of exploratory 3x3 boundary results: H1HIGH (n=5),
    H1SAT (n=5), H1PROBE700 (n=2 per-seed points). Group means only where
    meaningful; no smooth curve, no trend line, no threshold line."""
    probe_cfg = probe["probe"]
    scenario_key = probe_cfg["scenario"].split(":")[0].strip()   # H1PROBE700
    probe_root = REPO_ROOT / probe_cfg["result_root"]
    probe_seeds = probe_cfg["seeds"]
    methods = probe_cfg["methods"]                               # IPPO, IPPO-P

    # per-seed probe effects from raw evaluation metrics (mean U over 5 eval seeds)
    probe_log = []
    for seed in probe_seeds:
        u = {}
        for method in methods:
            m = load_json(probe_root / f"{scenario_key}_{method}_seed{seed}"
                          / "evaluation_metrics.json")
            u[method] = float(np.mean(np.asarray(m["demand_service_times"],
                                                 dtype=np.float64)))
        probe_log.append(np.log(u["IPPO"] / u["IPPO-P"]))
    probe_pct = to_pct(probe_log)

    groups = []
    for name, js in (("H1HIGH", scaleup_v1), ("H1SAT", scaleup_v2)):
        per_seed = js["per_seed"]
        seeds = sorted(per_seed, key=int)
        log_ratios = np.asarray([per_seed[s]["log_ratio"] for s in seeds],
                                dtype=np.float64)
        demand = re.search(r"=\s*(\d+)\s*veh/h", js["scenario"]).group(1)
        groups.append({
            "name": name,
            "demand": demand,
            "seeds": [int(s) for s in seeds],
            "pct": to_pct(log_ratios),
            "mean_pct": float(js["primary"]["improvement_pct"]),
            "wins": int(js["primary"]["seed_wins"]),
        })
    probe_demand = re.search(r"=\s*(\d+)\s*veh/h", probe_cfg["scenario"]).group(1)
    groups.append({
        "name": scenario_key,
        "demand": probe_demand,
        "seeds": list(probe_seeds),
        "pct": probe_pct,
        "mean_pct": None,   # n = 2: per-seed points only, no mean marker
        "wins": int(np.sum(np.asarray(probe_log) > 0)),
    })

    # consistency checks against the summary files
    for g in groups[:2]:
        assert g["wins"] == int(np.sum(g["pct"] > 0))
    assert groups[0]["wins"] == 5 and groups[1]["wins"] == 0
    assert groups[2]["wins"] == len(probe_seeds)
    assert np.all(groups[1]["pct"] < 0), "H1SAT must plot as the negative group"

    styles = [
        {"color": BLUE, "marker": "o"},
        {"color": ORANGE, "marker": "s"},
        {"color": GREEN, "marker": "^"},
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    ax.axhline(0.0, color=BLACK, lw=0.8, zorder=1)
    ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    rng = np.random.default_rng(0)  # deterministic small jitter only
    y_all = []
    for i, (g, st) in enumerate(zip(groups, styles)):
        n = len(g["pct"])
        offsets = np.linspace(-0.09, 0.09, n) if n > 2 else np.array([-0.06, 0.06])
        x = i + offsets
        ax.plot(x, g["pct"], linestyle="none", ms=4.2, mew=0.6,
                mec="white", zorder=3, **st)
        y_all.extend(g["pct"].tolist())
        if g["mean_pct"] is not None:
            ax.plot([i], [g["mean_pct"]], "D", ms=4.6, color=BLACK,
                    mec="none", zorder=4)
        txt_color = st["color"]
        if g["wins"] == 0:
            y_ann, va = min(g["pct"]) - 0.22, "top"
        else:
            y_ann, va = max(g["pct"]) + 0.22, "bottom"
        ax.text(i, y_ann, f"{g['wins']}/{n} positive", ha="center", va=va,
                fontsize=7, color=txt_color, fontweight="bold")
        y_all.extend([y_ann])

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [f"{g['name']}\n{g['demand']} veh/h\n(n = {len(g['seeds'])})" for g in groups],
        fontsize=7.5)
    for tick, st in zip(ax.get_xticklabels(), styles):
        tick.set_color(st["color"])
        tick.set_fontweight("bold")
    ax.set_xlim(-0.55, len(groups) - 0.45)
    lo, hi = min(y_all), max(y_all)
    ax.set_ylim(lo - 0.46 * (hi - lo), hi + 0.18 * (hi - lo))

    ax.set_ylabel("Improvement in demand service time, %")
    ax.tick_params(length=2.5, width=0.8)

    ax.set_title("Exploratory, not confirmatory",
                 loc="left", fontsize=7, style="italic", color="0.35",
                 pad=3)

    handles = [
        Line2D([], [], linestyle="none", marker="o", ms=4.2, color=BLUE,
               label="H1HIGH seeds (n = 5)"),
        Line2D([], [], linestyle="none", marker="s", ms=4.2, color=ORANGE,
               label="H1SAT seeds (n = 5)"),
        Line2D([], [], linestyle="none", marker="^", ms=4.2, color=GREEN,
               label="H1PROBE700 seeds (n = 2, no mean marker)"),
        Line2D([], [], linestyle="none", marker="D", ms=4.2, color=BLACK,
               label="Group mean (n = 5)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.0,
              frameon=False, handletextpad=0.3, borderaxespad=0.2,
              labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    out = {}
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"figure3_exploratory_boundary.{ext}"
        fig.savefig(path, bbox_inches="tight")
        out[ext] = path
    plt.close(fig)

    print("== Figure 3 (exploratory)")
    for g in groups:
        vals = ", ".join(f"{v:+.3f}" for v in g["pct"])
        print(f"  {g['name']} ({g['demand']} veh/h, n={len(g['seeds'])}, "
              f"seeds {g['seeds']}): {vals}  wins {g['wins']}/{len(g['seeds'])}"
              + (f"  group mean {g['mean_pct']:+.4f}%" if g["mean_pct"] is not None
                 else "  (per-seed points only)"))
    print("  sources:",
          SCALEUP_V1_JSON.relative_to(REPO_ROOT), ",",
          SCALEUP_V2_JSON.relative_to(REPO_ROOT), ",",
          PROBE_DECISION_JSON.relative_to(REPO_ROOT), "+ per-task evaluation_metrics.json")
    return out


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_json(CONF_JSON)
    scaleup_v1 = load_json(SCALEUP_V1_JSON)
    scaleup_v2 = load_json(SCALEUP_V2_JSON)
    probe = load_json(PROBE_DECISION_JSON)

    fig2_files = make_figure2(summary)
    fig3_files = make_figure3(scaleup_v1, scaleup_v2, probe)

    print("== Saved:")
    for path in [*fig2_files.values(), *fig3_files.values()]:
        print("  ", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
