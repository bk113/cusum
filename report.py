"""
CUSUM Active-Manager Monitor — PDF Report Generator
====================================================
Generates a multi-page PDF report covering:
  1. What CUSUM is and why it matters
  2. Fund universe, benchmarks, and thresholds
  3. Per-fund results with plots and statistics

Run:
    python report.py
Output:
    cusum_report.pdf
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from datetime import datetime

from cusum import (
    CUSUMMonitor, load_from_yfinance, monitor_fund,
    PRESET_PAIRS, THRESHOLD_TABLE
)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = "#1E2761"
RED    = "#C0392B"
GOLD   = "#D4AC0D"
GREY   = "#BDC3C7"
LGREY  = "#F4F6F7"
WHITE  = "#FFFFFF"
GREEN  = "#1E8449"

# ── Helpers ───────────────────────────────────────────────────────────────────

def page_footer(fig, page_num, total):
    fig.text(0.5, 0.02, f"CUSUM Active-Manager Monitor  •  page {page_num} of {total}",
             ha="center", va="bottom", fontsize=8, color="grey")
    fig.text(0.92, 0.02, datetime.today().strftime("%d %b %Y"),
             ha="right", va="bottom", fontsize=8, color="grey")


def divider(ax):
    ax.axhline(0, color=NAVY, lw=2)
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Cover + What is CUSUM?
# ═══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(NAVY)

    # Title block
    fig.text(0.5, 0.82, "CUSUM Active-Manager Monitor",
             ha="center", va="center", fontsize=28, fontweight="bold",
             color=WHITE, wrap=True)
    fig.text(0.5, 0.74, "Statistical Process Control for Portfolio Manager Oversight",
             ha="center", va="center", fontsize=14, color=GOLD)
    fig.text(0.5, 0.68, f"Report generated: {datetime.today().strftime('%d %B %Y')}",
             ha="center", va="center", fontsize=10, color=GREY)

    # Horizontal rule
    line = plt.Line2D([0.08, 0.92], [0.63, 0.63], transform=fig.transFigure,
                      color=GOLD, lw=1.5)
    fig.add_artist(line)

    # What is CUSUM section
    fig.text(0.08, 0.59, "What is CUSUM?", fontsize=15, fontweight="bold",
             color=GOLD, transform=fig.transFigure)

    cusum_text = (
        "CUSUM (Cumulative Sum) is a sequential change-point detection method originally developed "
        "for industrial quality control. In the context of active portfolio management, it was adapted "
        "by Philips, Yashchin & Stein (2003) to answer a critical question:\n\n"
        "   Has a fund manager lost their skill — and if so, when?"
    )
    fig.text(0.08, 0.50, cusum_text, fontsize=10, color=WHITE,
             transform=fig.transFigure, wrap=True,
             va="top", linespacing=1.6)

    # Why it matters
    fig.text(0.08, 0.35, "Why Does It Matter?", fontsize=15, fontweight="bold",
             color=GOLD, transform=fig.transFigure)

    reasons = [
        ("Objective & Systematic",
         "Removes emotional bias from manager review. Raises alarms based on statistical evidence, "
         "not recent headlines or short-term underperformance."),
        ("Early Warning",
         "Designed to detect persistent skill loss as quickly as possible while controlling the rate "
         "of false alarms. Default threshold ~1 false alarm per 5 years."),
        ("Grounded in Theory",
         "Uses the Lindley recursion (Page's CUSUM) on Information Ratio estimates. "
         "Accounts for time-varying tracking error via an EWMA estimator."),
        ("Actionable",
         "A triggered alarm is a signal to investigate — not necessarily to fire the manager — "
         "but to ask: Is the alpha source still intact?"),
    ]

    y = 0.29
    for title, body in reasons:
        fig.text(0.10, y, f"▸  {title}:", fontsize=10, fontweight="bold",
                 color=GOLD, transform=fig.transFigure)
        fig.text(0.10, y - 0.035, body, fontsize=9, color=GREY,
                 transform=fig.transFigure, wrap=True, va="top")
        y -= 0.085

    # Reference
    fig.text(0.5, 0.04,
             "Reference: Philips, T., Yashchin, E. & Stein, D. (2003). "
             "\"Using Statistical Process Control to Monitor Active Managers.\" "
             "Journal of Portfolio Management, 30(1), 86–94.",
             ha="center", fontsize=8, color=GREY, style="italic",
             transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — How CUSUM Works (Methodology)
# ═══════════════════════════════════════════════════════════════════════════════

def page_methodology(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.95, "How CUSUM Works", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.915, "Step-by-step mechanics of the Philips-Yashchin-Stein procedure",
             ha="center", fontsize=11, color="grey")

    # Steps
    steps = [
        ("Step 1 — Log Excess Return",
         r"Each month compute:   eₜ = ln[(1 + r_fund) / (1 + r_benchmark)]",
         "Log returns are used for additive properties and symmetry."),

        ("Step 2 — EWMA Tracking-Error Estimate",
         r"σ²ₜ = γ · σ²ₜ₋₁ + (1−γ) · ½(eₜ − eₜ₋₁)²       [γ = 0.90]",
         "Von Neumann adjacent-difference estimator: robust to drift in the mean, "
         "adapts as the manager's risk budget changes over time."),

        ("Step 3 — Monthly Information Ratio Estimate",
         r"IR̂ₜ = (eₜ · √12) / σₜ₋₁",
         "Annualised IR using the *lagged* sigma to keep the estimator approximately unbiased."),

        ("Step 4 — Lindley Recursion (Page's CUSUM)",
         r"Lₜ = max(0,  Lₜ₋₁ + drift)       drift = ½(IR_good + IR_bad) − IR̂ₜ",
         "Default: IR_good = 0.5, IR_bad = 0.0  →  drift = 0.25 − IR̂ₜ\n"
         "L accumulates when the manager underperforms (IR̂ < 0.25), resets to 0 when they outperform."),

        ("Step 5 — Alarm Rule",
         r"ALARM if Lₜ ≥ h       [default h = 19.81]",
         "Threshold h is chosen from Table 2 of the paper to control the false-alarm rate. "
         "h = 19.81 gives ~1 false alarm per 60 months (5 years) when the true IR = +0.5."),
    ]

    y = 0.87
    for i, (title, formula, explanation) in enumerate(steps):
        # Step number bubble
        ax_bubble = fig.add_axes([0.055, y - 0.005, 0.03, 0.038])
        circle = plt.Circle((0.5, 0.5), 0.5, color=NAVY)
        ax_bubble.add_patch(circle)
        ax_bubble.text(0.5, 0.5, str(i + 1), ha="center", va="center",
                       fontsize=11, fontweight="bold", color=WHITE)
        ax_bubble.set_xlim(0, 1); ax_bubble.set_ylim(0, 1)
        ax_bubble.axis("off")

        fig.text(0.10, y, title, fontsize=11, fontweight="bold", color=NAVY)
        fig.text(0.10, y - 0.028, formula, fontsize=10, color=RED,
                 fontfamily="monospace")
        fig.text(0.10, y - 0.053, explanation, fontsize=9, color="#555555",
                 va="top", wrap=True)
        y -= 0.155

    # Threshold table
    fig.text(0.08, 0.095, "Alarm Threshold Reference Table (Table 2, Philips et al. 2003)",
             fontsize=10, fontweight="bold", color=NAVY)

    col_labels = ["Threshold (h)", "ARL – Good (IR=0.5)", "ARL – Flat (IR=0)", "ARL – Bad (IR=−0.5)"]
    col_x      = [0.08, 0.28, 0.50, 0.70]
    row_data   = THRESHOLD_TABLE.values.tolist()

    # Header
    for lbl, x in zip(col_labels, col_x):
        fig.text(x, 0.073, lbl, fontsize=8, fontweight="bold", color=WHITE,
                 transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))

    for r, row in enumerate(row_data):
        bg = LGREY if r % 2 == 0 else WHITE
        yrow = 0.055 - r * 0.022
        rect = FancyBboxPatch((0.075, yrow - 0.008), 0.85, 0.020,
                              boxstyle="round,pad=0.002",
                              facecolor=bg, edgecolor="none",
                              transform=fig.transFigure)
        fig.add_artist(rect)
        highlight = (abs(row[0] - 19.81) < 0.01)
        for val, x in zip(row, col_x):
            label = f"{val:.2f}" if isinstance(val, float) else f"{int(val)} months"
            fig.text(x, yrow, label, fontsize=8,
                     color=RED if highlight else "#333333",
                     fontweight="bold" if highlight else "normal")
        if highlight:
            fig.text(0.90, yrow, "← default", fontsize=7, color=RED, style="italic")

    page_footer(fig, 2, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Fund Universe
# ═══════════════════════════════════════════════════════════════════════════════

def page_fund_universe(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.95, "Fund Universe", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.918, "Eight active managers across equity, international, and fixed income",
             ha="center", fontsize=11, color="grey")

    # Table headers
    headers = ["#", "Fund Ticker", "Fund Name", "Benchmark", "Asset Class", "Start Date", "Threshold (h)"]
    col_x   = [0.04, 0.09, 0.20, 0.53, 0.64, 0.77, 0.87]
    widths  = [0.05, 0.10, 0.32, 0.10, 0.12, 0.09, 0.10]

    # Header row background
    header_rect = FancyBboxPatch((0.03, 0.855), 0.94, 0.028,
                                 boxstyle="round,pad=0.003",
                                 facecolor=NAVY, edgecolor="none",
                                 transform=fig.transFigure)
    fig.add_artist(header_rect)
    for hdr, x in zip(headers, col_x):
        fig.text(x, 0.863, hdr, fontsize=9, fontweight="bold",
                 color=WHITE, va="center")

    asset_classes = ["US Large Growth", "US Large Value", "US Large Blend",
                     "US Small Cap", "Intl Developed", "Emerging Markets",
                     "Balanced 60/40", "US Fixed Income"]

    rows = []
    for i, (ticker, (fund, bench, name)) in enumerate(PRESET_PAIRS.items()):
        rows.append([str(i + 1), fund, name, bench, asset_classes[i], "Jan 2005", "19.81"])

    for r, row in enumerate(rows):
        bg = LGREY if r % 2 == 0 else WHITE
        yrow = 0.825 - r * 0.072
        rect = FancyBboxPatch((0.03, yrow - 0.015), 0.94, 0.068,
                              boxstyle="round,pad=0.003",
                              facecolor=bg, edgecolor="none",
                              transform=fig.transFigure)
        fig.add_artist(rect)
        for val, x, w in zip(row, col_x, widths):
            fig.text(x, yrow + 0.018, val, fontsize=9, va="center",
                     color=NAVY if r < 4 else ("#1A5276" if r < 6 else "#145A32"),
                     wrap=True)

    # Legend: asset class colour note
    fig.text(0.04, 0.30, "Asset Class Groups:", fontsize=10, fontweight="bold", color=NAVY)
    groups = [
        (NAVY,     "US Equity (FCNTX, DODGX, AGTHX, OTCFX)"),
        ("#1A5276", "International Equity (OAKIX, TEDMX)"),
        ("#145A32", "Fixed Income / Balanced (VWELX, PTTRX)"),
    ]
    for gy, (col, label) in zip([0.275, 0.252, 0.229], groups):
        patch = mpatches.Patch(color=col)
        fig.text(0.06, gy, f"■  {label}", fontsize=9, color=col)

    # Threshold explanation box
    ax_box = fig.add_axes([0.04, 0.05, 0.92, 0.155])
    ax_box.set_facecolor(LGREY)
    for spine in ax_box.spines.values():
        spine.set_visible(False)
    ax_box.set_xticks([]); ax_box.set_yticks([])

    ax_box.text(0.5, 0.88, "Threshold Choice: h = 19.81 (paper default)",
                ha="center", va="top", fontsize=10, fontweight="bold", color=NAVY,
                transform=ax_box.transAxes)
    ax_box.text(0.5, 0.65,
                "All funds are monitored using the paper's recommended default threshold of h = 19.81.\n"
                "At this level, the expected time between false alarms is ~60 months (5 years) when the "
                "true manager IR = +0.5.\n"
                "A higher h (e.g. 23.59) would reduce false alarms at the cost of slower detection. "
                "A lower h (e.g. 11.81) detects skill loss faster but flags more false positives.\n"
                "The threshold can be adjusted via the `threshold` parameter in `monitor_fund()`.",
                ha="center", va="top", fontsize=9, color="#444444",
                transform=ax_box.transAxes, linespacing=1.7)

    page_footer(fig, 3, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Summary Results Table
# ═══════════════════════════════════════════════════════════════════════════════

def page_summary(pdf, results):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.95, "Summary Results", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.918, "CUSUM alarm status and key statistics for each manager (Jan 2005 – present)",
             ha="center", fontsize=11, color="grey")

    headers = ["Fund", "Benchmark", "Months", "Ann. TE", "Realised IR", "Cum. Excess", "Alarm?", "Alarm Date"]
    col_x   = [0.04, 0.13, 0.22, 0.31, 0.42, 0.55, 0.68, 0.79]

    header_rect = FancyBboxPatch((0.03, 0.855), 0.94, 0.028,
                                 boxstyle="round,pad=0.003",
                                 facecolor=NAVY, edgecolor="none",
                                 transform=fig.transFigure)
    fig.add_artist(header_rect)
    for hdr, x in zip(headers, col_x):
        fig.text(x, 0.863, hdr, fontsize=9, fontweight="bold", color=WHITE, va="center")

    for r, res in enumerate(results):
        bg = LGREY if r % 2 == 0 else WHITE
        yrow = 0.815 - r * 0.072
        rect = FancyBboxPatch((0.03, yrow - 0.015), 0.94, 0.060,
                              boxstyle="round,pad=0.003",
                              facecolor=bg, edgecolor="none",
                              transform=fig.transFigure)
        fig.add_artist(rect)

        alarm_col  = RED   if res["alarm"] else GREEN
        alarm_text = "YES" if res["alarm"] else "NO"
        ir_col = GREEN if res["ir"] > 0.3 else (RED if res["ir"] < 0.1 else NAVY)

        vals = [res["fund"], res["bench"], str(res["months"]),
                f"{res['te']:.1%}", f"{res['ir']:+.2f}",
                f"{res['cum_excess']:+.1%}", alarm_text, res["alarm_date"]]
        colors = [NAVY, NAVY, NAVY, NAVY, ir_col, NAVY, alarm_col, alarm_col]

        for val, x, col in zip(vals, col_x, colors):
            fw = "bold" if col != NAVY else "normal"
            fig.text(x, yrow + 0.010, val, fontsize=9, va="center",
                     color=col, fontweight=fw)

    # Legend
    fig.text(0.04, 0.24, "Key:", fontsize=10, fontweight="bold", color=NAVY)
    fig.text(0.04, 0.215,
             f"■  IR (green): IR > 0.30  indicates strong outperformance  |  "
             f"IR (red): IR < 0.10  indicates weak/no skill\n"
             f"■  Alarm YES (red): CUSUM statistic breached h = 19.81 threshold\n"
             f"■  Alarm NO (green): Manager remained on-process for full observation period",
             fontsize=9, color="#444444", linespacing=1.7)

    # Observations box
    ax_obs = fig.add_axes([0.04, 0.04, 0.92, 0.155])
    ax_obs.set_facecolor("#EAF2FF")
    for spine in ax_obs.spines.values():
        spine.set_visible(False)
    ax_obs.set_xticks([]); ax_obs.set_yticks([])

    ax_obs.text(0.5, 0.90, "Key Observations", ha="center", va="top",
                fontsize=10, fontweight="bold", color=NAVY,
                transform=ax_obs.transAxes)

    alarm_funds = [r for r in results if r["alarm"]]
    clean_funds = [r for r in results if not r["alarm"]]
    earliest    = min(alarm_funds, key=lambda x: x["alarm_date"])
    best_ir     = max(results, key=lambda x: x["ir"])

    obs_text = (
        f"• {len(alarm_funds)} of {len(results)} managers triggered a CUSUM alarm — "
        f"most clustering around the 2007–2009 Global Financial Crisis.\n"
        f"• Earliest alarm: {earliest['fund']} ({earliest['alarm_date']}) — "
        f"persistent underperformance vs {earliest['bench']} detected within 2 years.\n"
        f"• Clean record: {', '.join(r['fund'] for r in clean_funds)} — "
        f"no alarm fired across the full observation period.\n"
        f"• Strongest manager: {best_ir['fund']} with realised IR = {best_ir['ir']:+.2f} "
        f"and cumulative excess of {best_ir['cum_excess']:+.1%}."
    )
    ax_obs.text(0.03, 0.68, obs_text, va="top", fontsize=9, color="#333333",
                transform=ax_obs.transAxes, linespacing=1.7)

    page_footer(fig, 4, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Individual Fund Plots (2×4 grid)
# ═══════════════════════════════════════════════════════════════════════════════

def page_fund_plots(pdf, all_rets, all_monitors, all_results):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)
    fig.text(0.5, 0.97, "Individual Fund CUSUM Charts", ha="center",
             fontsize=16, fontweight="bold", color=NAVY)

    outer = gridspec.GridSpec(4, 2, figure=fig,
                              left=0.06, right=0.97,
                              top=0.93, bottom=0.06,
                              hspace=0.55, wspace=0.3)

    for idx, (res, rets, mon) in enumerate(zip(all_results, all_rets, all_monitors)):
        row, col = divmod(idx, 2)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                                 hspace=0.0, height_ratios=[1, 1.2])
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)

        df   = mon.history_df
        x    = df["t"].values
        dates = rets.index

        # Panel 1: cumulative excess return
        ax1.plot(x, df["excess_return"].cumsum(), lw=1.2, color=NAVY)
        ax1.axhline(0, color=GREY, lw=0.5)
        ax1.set_facecolor(LGREY)
        ax1.tick_params(labelbottom=False, labelsize=6)
        ax1.yaxis.set_tick_params(labelsize=6)
        ax1.set_title(f"{res['fund']} vs {res['bench']}", fontsize=8,
                      fontweight="bold", color=NAVY, pad=2)
        ax1.grid(alpha=0.3, lw=0.4)

        # Panel 2: CUSUM statistic
        ax2.plot(x, df["L"], lw=1.2, color=NAVY)
        ax2.axhline(mon.threshold, color=RED, ls="--", lw=0.8)
        ax2.invert_yaxis()
        ax2.set_facecolor(LGREY)
        ax2.grid(alpha=0.3, lw=0.4)

        if mon.alarm:
            ax2.axvline(mon.alarm_t, color=RED, lw=0.8, alpha=0.7)
            alarm_date_str = dates[mon.alarm_t - 1].strftime("%b'%y")
            ax2.text(mon.alarm_t + len(x) * 0.02,
                     mon.threshold * 0.55,
                     f"▲{alarm_date_str}", color=RED,
                     fontsize=6, fontweight="bold")

        # x-axis: year labels
        tick_locs = [t for t in ax2.get_xticks() if 0 < int(t) <= len(dates)]
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(
            [dates[int(t) - 1].strftime("%Y") for t in tick_locs],
            fontsize=5, rotation=45)

        # Stat annotation
        status = f"ALARM {res['alarm_date']}" if res["alarm"] else "No Alarm"
        status_col = RED if res["alarm"] else GREEN
        ax1.text(0.98, 0.95,
                 f"IR={res['ir']:+.2f}  TE={res['te']:.1%}\n{status}",
                 transform=ax1.transAxes, fontsize=5.5, ha="right", va="top",
                 color=status_col, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE,
                           edgecolor=status_col, alpha=0.85))

    page_footer(fig, 5, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

class MonitorResult:
    """Thin wrapper so we can attach history_df to a CUSUMMonitor."""
    def __init__(self, monitor, history_df):
        for k, v in vars(monitor).items():
            setattr(self, k, v)
        self.history_df = history_df


def collect_data():
    all_rets, all_monitors, all_results = [], [], []
    print("Fetching data from Yahoo Finance…")
    for ticker, (fund, bench, name) in PRESET_PAIRS.items():
        print(f"  {ticker}…", end=" ", flush=True)
        try:
            rets = load_from_yfinance(fund, bench)

            # Estimate initial TE
            seed = rets.iloc[: min(24, max(6, len(rets) // 4))]
            excess_seed = seed["portfolio"] - seed["benchmark"]
            init_te = excess_seed.std() * np.sqrt(12)

            monitor = CUSUMMonitor(initial_te=init_te, threshold=19.81)
            history_df = monitor.run(rets)

            excess = rets["portfolio"] - rets["benchmark"]
            ir    = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
            te    = excess.std() * np.sqrt(12)
            cum   = (1 + rets["portfolio"]).prod() / (1 + rets["benchmark"]).prod() - 1

            alarm_date = (rets.index[monitor.alarm_t - 1].strftime("%b %Y")
                          if monitor.alarm else "—")

            all_rets.append(rets)
            mr = MonitorResult(monitor, history_df)
            all_monitors.append(mr)
            all_results.append({
                "fund": fund, "bench": bench, "name": name,
                "months": len(rets), "ir": ir, "te": te,
                "cum_excess": cum, "alarm": monitor.alarm,
                "alarm_date": alarm_date,
            })
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    return all_rets, all_monitors, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    output = "cusum_report.pdf"
    all_rets, all_monitors, all_results = collect_data()

    print(f"\nBuilding report -> {output}")
    with PdfPages(output) as pdf:
        page_cover(pdf)
        page_methodology(pdf)
        page_fund_universe(pdf)
        page_summary(pdf, all_results)
        page_fund_plots(pdf, all_rets, all_monitors, all_results)

    print(f"Done. Report saved to: {output}")


if __name__ == "__main__":
    main()
