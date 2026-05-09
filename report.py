"""
CUSUM Active-Manager Monitor — PDF & HTML Report Generator
===========================================================
Generates a 7-page PDF report (and optional self-contained HTML):
  1. Cover + CIO case for CUSUM
  2. Methodology — step-by-step formula walkthrough
  3. Fund universe — 14 funds, benchmarks, thresholds
  4. Summary results — IR, TE, CI, alarm status
  5. Post-alarm analysis — returns after each alarm
  6. Waterfall chart — cumulative excess ranked
  7. Individual CUSUM charts

Run:
    python report.py                 # PDF only
    python report.py --html          # PDF + HTML
    python report.py --output my.pdf # custom output path
"""

import base64
import io
import argparse
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
    PRESET_PAIRS, THRESHOLD_TABLE,
    bootstrap_ir_ci, post_alarm_returns,
)

TOTAL_PAGES = 7

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY  = "#1E2761"
RED   = "#C0392B"
GOLD  = "#D4AC0D"
GREY  = "#BDC3C7"
LGREY = "#F4F6F7"
WHITE = "#FFFFFF"
GREEN = "#1E8449"


# ── Helpers ───────────────────────────────────────────────────────────────────

def page_footer(fig, page_num):
    fig.text(0.5, 0.02,
             f"CUSUM Active-Manager Monitor  •  page {page_num} of {TOTAL_PAGES}",
             ha="center", va="bottom", fontsize=8, color="grey")
    fig.text(0.92, 0.02, datetime.today().strftime("%d %b %Y"),
             ha="right", va="bottom", fontsize=8, color="grey")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Cover + CIO Case
# ═══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(NAVY)

    # Title
    fig.text(0.5, 0.91, "CUSUM Active-Manager Monitor",
             ha="center", fontsize=26, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.85, "Statistical Process Control for Portfolio Manager Oversight",
             ha="center", fontsize=13, color=GOLD)
    fig.text(0.5, 0.80, f"Report generated: {datetime.today().strftime('%d %B %Y')}",
             ha="center", fontsize=10, color=GREY)

    line = plt.Line2D([0.08, 0.92], [0.77, 0.77], transform=fig.transFigure,
                      color=GOLD, lw=1.2)
    fig.add_artist(line)

    # What is CUSUM
    fig.text(0.08, 0.74, "What is CUSUM?", fontsize=13, fontweight="bold",
             color=GOLD)
    fig.text(0.08, 0.69,
             "CUSUM (Cumulative Sum) is a sequential change-point detection method from "
             "statistical process control, adapted for investment management by Philips, "
             "Yashchin & Stein (2003). It answers one critical question:",
             fontsize=9, color=WHITE, va="top", linespacing=1.5)
    fig.text(0.10, 0.625,
             "Has a fund manager lost their investment skill — and if so, exactly when?",
             fontsize=10, color=GOLD, fontstyle="italic", va="top")

    # Why it matters — CIO case
    fig.text(0.08, 0.585, "The Case for CUSUM — Why CIOs Should Care",
             fontsize=13, fontweight="bold", color=GOLD)

    reasons = [
        ("Cost of Inaction",
         "Retaining an underperforming manager for 3 extra years on a 500M mandate "
         "at IR=-0.3, TE=6% costs ~9M in excess losses. Early, systematic detection pays."),
        ("Faster than Rolling IR",
         "A trailing 3-year IR review needs 10+ years of data to distinguish skill from "
         "luck at 95% confidence. CUSUM detects persistent CHANGES in IR within 2-3 years."),
        ("Rigorous False-Alarm Control",
         "Threshold h is calibrated from the paper's ARL table: h=19.81 gives ~1 false "
         "alarm per 60 months (5 years) when the true IR=+0.5. Fewer spurious terminations."),
        ("Two-Sided Monitoring",
         "Detects skill LOSS (lower CUSUM) and unexpected skill GAIN (upper CUSUM), "
         "enabling both firing decisions and early-stage manager promotions."),
        ("Rolling Restart",
         "After each alarm L resets to zero — monitoring never stops. Multiple alarms "
         "build a richer signal; post-alarm returns can be tracked to validate each flag."),
        ("Industry Precedent",
         "SPC methods are standard in manufacturing, clinical trials, and nuclear operations. "
         "Mercer and WTW publish SPC-adjacent monitoring frameworks for manager oversight."),
        ("Actionable, Not Automatic",
         "An alarm triggers a structured review — not automatic termination. It forces "
         "the investment committee to ask: Is the alpha source still intact?"),
    ]

    y = 0.545
    for title, body in reasons:
        fig.text(0.10, y, f"  {title}:", fontsize=8.5, fontweight="bold",
                 color=GOLD)
        fig.text(0.10, y - 0.028, f"  {body}", fontsize=7.8, color=GREY,
                 va="top", linespacing=1.4)
        y -= 0.068

    fig.text(0.5, 0.025,
             "Reference: Philips, T., Yashchin, E. & Stein, D. (2003). "
             "\"Using Statistical Process Control to Monitor Active Managers.\" "
             "Journal of Portfolio Management, 30(1), 86-94.",
             ha="center", fontsize=7.5, color=GREY, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════

def page_methodology(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.96, "How CUSUM Works", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.932, "Step-by-step mechanics of the Philips-Yashchin-Stein procedure",
             ha="center", fontsize=11, color="grey")

    steps = [
        ("Step 1 — Log Excess Return",
         r"Each month:   et = ln[(1 + r_fund) / (1 + r_benchmark)]",
         "Log returns are additive and symmetric — they handle compounding correctly."),

        ("Step 2 — EWMA Tracking-Error Estimate",
         r"sigma^2_t = gamma * sigma^2_{t-1} + (1-gamma) * 0.5*(et - e_{t-1})^2   [gamma=0.90]",
         "Von Neumann adjacent-difference estimator: robust to drift in the mean, "
         "adapts as the manager's risk budget changes."),

        ("Step 3 — Monthly Information Ratio Estimate",
         r"IR_hat_t = (et * sqrt(12)) / sigma_{t-1}",
         "Annualised IR using lagged sigma to keep the estimator approximately unbiased."),

        ("Step 4 — Lindley Recursion (Page's CUSUM)",
         r"L_t = max(0, L_{t-1} + drift)    drift = 0.5*(IR_good + IR_bad) - IR_hat_t",
         "Default: IR_good=0.5, IR_bad=0.0 -> drift = 0.25 - IR_hat. "
         "L grows when manager underperforms, resets to 0 when they outperform. "
         "Upper CUSUM L_up uses drift_up = IR_hat - 0.5*(IR_exceptional + IR_good)."),

        ("Step 5 — Rolling Alarm Rule",
         r"ALARM if L_t >= h   [default h = 19.81]   then reset L_t = 0 and continue",
         "After each alarm L resets to zero — monitoring never stops. "
         "All alarm times are recorded; post-alarm returns validate each signal."),
    ]

    y = 0.885
    for i, (title, formula, explanation) in enumerate(steps):
        ax_b = fig.add_axes([0.055, y - 0.004, 0.028, 0.035])
        circle = plt.Circle((0.5, 0.5), 0.5, color=NAVY)
        ax_b.add_patch(circle)
        ax_b.text(0.5, 0.5, str(i + 1), ha="center", va="center",
                  fontsize=10, fontweight="bold", color=WHITE)
        ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")

        fig.text(0.10, y, title, fontsize=10.5, fontweight="bold", color=NAVY)
        fig.text(0.10, y - 0.027, formula, fontsize=9, color=RED,
                 fontfamily="monospace")
        fig.text(0.10, y - 0.050, explanation, fontsize=8.5, color="#555555",
                 va="top", linespacing=1.4)
        y -= 0.148

    # Threshold table
    fig.text(0.08, 0.112, "Alarm Threshold Reference Table (Table 2, Philips et al. 2003)",
             fontsize=10, fontweight="bold", color=NAVY)

    col_labels = ["Threshold (h)", "ARL — Good (IR=0.5)", "ARL — Flat (IR=0)",
                  "ARL — Bad (IR=-0.5)"]
    col_x = [0.08, 0.28, 0.50, 0.70]

    for lbl, x in zip(col_labels, col_x):
        fig.text(x, 0.093, lbl, fontsize=8, fontweight="bold", color=WHITE,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))

    for r, row in enumerate(THRESHOLD_TABLE.values.tolist()):
        bg  = LGREY if r % 2 == 0 else WHITE
        yrow = 0.076 - r * 0.021
        rect = FancyBboxPatch((0.075, yrow - 0.007), 0.85, 0.019,
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
            fig.text(0.90, yrow, "<- default", fontsize=7, color=RED, style="italic")

    page_footer(fig, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Fund Universe
# ═══════════════════════════════════════════════════════════════════════════════

def page_fund_universe(pdf):
    """Renders fund universe across as many pages as needed (20 rows per page)."""
    asset_classes = [
        "US Large Growth", "US Large Value",  "US Large Blend",  "US Small Cap",
        "Intl Developed",  "Emerging Markets","Balanced 60/40",  "US Fixed Income",
        "Balanced 60/40",  "US Large Blend",  "US Large Blend",  "US Small/Mid Cap",
        "Intl Value",      "US Fixed Income",
        # EM funds (27)
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets","Emerging Markets",
        "Emerging Markets","Emerging Markets","Emerging Markets",
    ]
    color_map = {
        "US Large Growth": NAVY, "US Large Value": NAVY, "US Large Blend": NAVY,
        "US Small Cap": NAVY, "US Small/Mid Cap": NAVY,
        "Intl Developed": "#1A5276", "Intl Value": "#1A5276", "Emerging Markets": "#1A5276",
        "Balanced 60/40": "#145A32", "US Fixed Income": "#145A32",
    }

    all_rows = []
    for i, (ticker, (fund, bench, name)) in enumerate(PRESET_PAIRS.items()):
        all_rows.append([str(i + 1), fund, name, bench,
                         asset_classes[i] if i < len(asset_classes) else "Other",
                         "Jan 2005", "19.81"])

    rows_per_page = 20
    total_fund_pages = int(np.ceil(len(all_rows) / rows_per_page))
    headers = ["#", "Ticker", "Fund Name", "Benchmark", "Asset Class", "Start", "h"]
    col_x   = [0.035, 0.075, 0.175, 0.545, 0.640, 0.760, 0.855]

    for fp in range(total_fund_pages):
        chunk = all_rows[fp * rows_per_page:(fp + 1) * rows_per_page]
        fig   = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        subtitle = (f"{len(all_rows)} active managers — page {fp+1} of {total_fund_pages}"
                    if total_fund_pages > 1 else
                    f"{len(all_rows)} active managers across equity, international, fixed income & EM")
        fig.text(0.5, 0.96, "Fund Universe", ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.928, subtitle, ha="center", fontsize=11, color="grey")

        header_rect = FancyBboxPatch((0.03, 0.878), 0.94, 0.026,
                                     boxstyle="round,pad=0.003",
                                     facecolor=NAVY, edgecolor="none",
                                     transform=fig.transFigure)
        fig.add_artist(header_rect)
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=8.5, fontweight="bold",
                     color=WHITE, va="center")

        row_h   = min(0.040, 0.82 / max(len(chunk), 1))
        y_start = 0.862

        for r, row in enumerate(chunk):
            bg   = LGREY if r % 2 == 0 else WHITE
            yrow = y_start - (r + 1) * row_h
            rect = FancyBboxPatch((0.03, yrow - 0.005), 0.94, row_h - 0.003,
                                  boxstyle="round,pad=0.002",
                                  facecolor=bg, edgecolor="none",
                                  transform=fig.transFigure)
            fig.add_artist(rect)
            ac_col = color_map.get(row[4], NAVY)
            for j, (val, x) in enumerate(zip(row, col_x)):
                col = ac_col if j == 4 else NAVY
                fig.text(x, yrow + row_h * 0.25, val,
                         fontsize=7.5, va="center", color=col)

        # Legend (first page only)
        if fp == 0:
            legend_y = max(0.04, y_start - (len(chunk) + 1) * row_h - 0.01)
            legend_items = [
                (NAVY,     "US Equity (FCNTX, DODGX, AGTHX, OTCFX, FMAGX, SEQUX, FLPSX)"),
                ("#1A5276", "Intl & Emerging Markets (OAKIX, TEDMX, DODWX, + 27 EM funds)"),
                ("#145A32", "Fixed Income / Balanced (VWELX, PTTRX, PRWCX, LSBRX)"),
            ]
            for li, (col, label) in enumerate(legend_items):
                fig.text(0.06, legend_y - li * 0.020, f"  {label}",
                         fontsize=8, color=col)

        page_footer(fig, 3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Summary Results
# ═══════════════════════════════════════════════════════════════════════════════

def page_summary(pdf, all_results):
    """Renders summary results across as many pages as needed (18 rows per page)."""
    headers = ["Fund", "Bench", "Months", "Ann TE", "IR (95% CI)", "Cum Excess",
               "# Alarms", "Alarm?", "First Alarm"]
    col_x   = [0.035, 0.098, 0.163, 0.228, 0.300, 0.445, 0.555, 0.630, 0.715]

    rows_per_page  = 18
    total_sum_pages = int(np.ceil(len(all_results) / rows_per_page))

    alarm_funds = [r for r in all_results if r["alarm"]]
    clean_funds = [r for r in all_results if not r["alarm"]]
    best_ir     = max(all_results, key=lambda x: x["ir"])
    most_alarms = max(all_results, key=lambda x: x.get("n_alarms", 0))

    for sp in range(total_sum_pages):
        chunk = all_results[sp * rows_per_page:(sp + 1) * rows_per_page]
        fig   = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        subtitle = (f"CUSUM alarm status and key statistics (Jan 2005 - present) "
                    f"— page {sp+1} of {total_sum_pages}"
                    if total_sum_pages > 1 else
                    "CUSUM alarm status and key statistics (Jan 2005 - present)")
        fig.text(0.5, 0.96, "Summary Results", ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.928, subtitle, ha="center", fontsize=10, color="grey")

        header_rect = FancyBboxPatch((0.03, 0.878), 0.94, 0.026,
                                     boxstyle="round,pad=0.003",
                                     facecolor=NAVY, edgecolor="none",
                                     transform=fig.transFigure)
        fig.add_artist(header_rect)
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=8, fontweight="bold", color=WHITE, va="center")

        row_h   = min(0.042, 0.75 / max(len(chunk), 1))
        y_start = 0.862

        for r, res in enumerate(chunk):
            bg        = LGREY if r % 2 == 0 else WHITE
            yrow      = y_start - (r + 1) * row_h
            rect      = FancyBboxPatch((0.03, yrow - 0.004), 0.94, row_h - 0.003,
                                       boxstyle="round,pad=0.002",
                                       facecolor=bg, edgecolor="none",
                                       transform=fig.transFigure)
            fig.add_artist(rect)

            alarm_col  = RED   if res["alarm"] else GREEN
            alarm_text = "YES" if res["alarm"] else "NO"
            ir_col     = (GREEN if res["ir"] > 0.3 else (RED if res["ir"] < 0.1 else NAVY))
            ci_str     = (f"{res['ir']:+.2f} [{res['ir_lo']:+.2f},{res['ir_hi']:+.2f}]"
                          if "ir_lo" in res else f"{res['ir']:+.2f}")

            vals   = [res["fund"], res["bench"], str(res["months"]),
                      f"{res['te']:.1%}", ci_str, f"{res['cum_excess']:+.1%}",
                      str(res.get("n_alarms", "-")), alarm_text, res["alarm_date"]]
            colors = [NAVY, NAVY, NAVY, NAVY, ir_col, NAVY, NAVY, alarm_col, alarm_col]

            for val, x, col in zip(vals, col_x, colors):
                fig.text(x, yrow + row_h * 0.3, val, fontsize=7.5, va="center",
                         color=col, fontweight="bold" if col != NAVY else "normal")

        # Key observations box on last page only
        if sp == total_sum_pages - 1:
            obs_y = max(0.03, y_start - (len(chunk) + 1) * row_h - 0.01)
            ax_obs = fig.add_axes([0.04, obs_y, 0.92, min(0.13, obs_y + 0.10)])
            ax_obs.set_facecolor("#EAF2FF")
            for s in ax_obs.spines.values():
                s.set_visible(False)
            ax_obs.set_xticks([]); ax_obs.set_yticks([])
            ax_obs.text(0.5, 0.95, "Key Observations", ha="center", va="top",
                        fontsize=9, fontweight="bold", color=NAVY,
                        transform=ax_obs.transAxes)
            obs = (
                f"  {len(alarm_funds)} of {len(all_results)} managers triggered an alarm. "
                f"Clean record: {', '.join(f['fund'] for f in clean_funds[:5])}{'...' if len(clean_funds) > 5 else ''}.\n"
                f"  Strongest IR: {best_ir['fund']} ({best_ir['ir']:+.2f}), "
                f"cum excess {best_ir['cum_excess']:+.1%}.  "
                f"Most re-alarms: {most_alarms['fund']} ({most_alarms.get('n_alarms', 0)}x)."
            )
            ax_obs.text(0.02, 0.62, obs, va="top", fontsize=8, color="#333333",
                        transform=ax_obs.transAxes, linespacing=1.6)

        page_footer(fig, 4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Post-Alarm Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def page_post_alarm(pdf, all_results, all_post_alarm):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.96, "Post-Alarm Analysis", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.928,
             "Cumulative excess return vs benchmark in the months following the FIRST alarm",
             ha="center", fontsize=11, color="grey")

    headers = ["Fund", "First Alarm", "+12M Fund", "+12M Bench", "+12M Excess",
               "+24M Excess", "+36M Excess", "12M Signal"]
    col_x   = [0.035, 0.115, 0.220, 0.310, 0.400, 0.500, 0.595, 0.690]

    header_rect = FancyBboxPatch((0.03, 0.878), 0.94, 0.026,
                                 boxstyle="round,pad=0.003",
                                 facecolor=NAVY, edgecolor="none",
                                 transform=fig.transFigure)
    fig.add_artist(header_rect)
    for hdr, x in zip(headers, col_x):
        fig.text(x, 0.884, hdr, fontsize=8, fontweight="bold", color=WHITE, va="center")

    def _fmt(v):
        return f"{v:+.1%}" if not (v != v) else "n/a"  # nan check

    r = 0
    for res, pa_df in zip(all_results, all_post_alarm):
        if not res["alarm"] or pa_df is None or pa_df.empty:
            continue

        # Use only the first alarm row for each horizon
        first_alarm_rows = pa_df[pa_df["alarm_t"] == pa_df["alarm_t"].min()]
        row12 = first_alarm_rows[first_alarm_rows["horizon"] == 12]
        row24 = first_alarm_rows[first_alarm_rows["horizon"] == 24]
        row36 = first_alarm_rows[first_alarm_rows["horizon"] == 36]

        if row12.empty:
            continue

        d12 = row12.iloc[0]
        exc12 = d12["excess_cum"]
        exc24 = row24.iloc[0]["excess_cum"] if not row24.empty else np.nan
        exc36 = row36.iloc[0]["excess_cum"] if not row36.empty else np.nan
        correct = d12["correct_alarm"]

        bg   = LGREY if r % 2 == 0 else WHITE
        yrow = 0.845 - r * 0.048
        rect = FancyBboxPatch((0.03, yrow - 0.013), 0.94, 0.043,
                              boxstyle="round,pad=0.002",
                              facecolor=bg, edgecolor="none",
                              transform=fig.transFigure)
        fig.add_artist(rect)

        signal_col  = GREEN if correct else RED
        signal_text = "CORRECT" if correct else "FALSE"
        exc12_col   = RED if not (exc12 != exc12) and exc12 < 0 else GREEN

        vals   = [res["fund"], res["alarm_date"],
                  _fmt(d12["fund_cum"]), _fmt(d12["bench_cum"]),
                  _fmt(exc12), _fmt(exc24), _fmt(exc36), signal_text]
        colors = [NAVY, NAVY, NAVY, NAVY, exc12_col,
                  (RED if not (exc24 != exc24) and exc24 < 0 else GREEN),
                  (RED if not (exc36 != exc36) and exc36 < 0 else GREEN),
                  signal_col]

        for val, x, col in zip(vals, col_x, colors):
            fig.text(x, yrow + 0.008, val, fontsize=8, va="center",
                     color=col, fontweight="bold" if col != NAVY else "normal")
        r += 1

    # No alarm funds
    no_alarm = [res["fund"] for res in all_results if not res["alarm"]]
    if no_alarm:
        yrow = 0.845 - r * 0.048 - 0.02
        fig.text(0.035, yrow,
                 f"No alarm (no post-alarm analysis): {', '.join(no_alarm)}",
                 fontsize=8.5, color=GREEN, fontstyle="italic")

    # Explanation box
    ax_exp = fig.add_axes([0.04, 0.03, 0.92, 0.130])
    ax_exp.set_facecolor(LGREY)
    for sp in ax_exp.spines.values():
        sp.set_visible(False)
    ax_exp.set_xticks([]); ax_exp.set_yticks([])

    ax_exp.text(0.5, 0.93, "How to Read This Table", ha="center", va="top",
                fontsize=10, fontweight="bold", color=NAVY,
                transform=ax_exp.transAxes)
    ax_exp.text(0.03, 0.68,
                "  +12M / +24M / +36M Excess: cumulative return of the fund minus the benchmark "
                "in the 12, 24, and 36 months immediately AFTER the first alarm.\n"
                "  CORRECT: The fund underperformed its benchmark after the alarm (negative excess) "
                "— the CUSUM signal was validated by subsequent returns.\n"
                "  FALSE: The fund outperformed after the alarm — the signal was a false positive. "
                "At h=19.81, theory predicts ~1 false alarm per 60 months.",
                va="top", fontsize=8.5, color="#333333",
                transform=ax_exp.transAxes, linespacing=1.7)

    page_footer(fig, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Waterfall Chart
# ═══════════════════════════════════════════════════════════════════════════════

def page_waterfall(pdf, all_results):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.96, "Cumulative Excess Returns — Ranked", ha="center",
             fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.928,
             "Total return of each fund vs its benchmark (Jan 2005 - present)",
             ha="center", fontsize=11, color="grey")

    ax = fig.add_axes([0.22, 0.10, 0.70, 0.80])

    sorted_res = sorted(all_results, key=lambda x: x["cum_excess"])
    labels = [f"{r['fund']} vs {r['bench']}" for r in sorted_res]
    values = [r["cum_excess"] for r in sorted_res]
    colors = [GREEN if v >= 0 else RED for v in values]
    alarm  = [r["alarm"] for r in sorted_res]

    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="white", lw=0.5)

    # Value labels
    for bar, val, alm in zip(bars, values, alarm):
        x_pos = val + 0.003 if val >= 0 else val - 0.003
        ha    = "left" if val >= 0 else "right"
        label = f"{val:+.1%}"
        if alm:
            label += " *"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=9,
                color=GREEN if val >= 0 else RED, fontweight="bold")

    ax.axvline(0, color=NAVY, lw=1.2)
    ax.set_xlabel("Cumulative excess return vs benchmark", fontsize=10, color=NAVY)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_facecolor(LGREY)
    ax.grid(axis="x", alpha=0.4, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=GREEN, label="Outperformed benchmark"),
        mpatches.Patch(facecolor=RED,   label="Underperformed benchmark"),
        plt.Line2D([0], [0], color="none", label="* = CUSUM alarm triggered"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    page_footer(fig, 6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7+ — Individual Fund Charts (6 per page, 3 rows x 2 cols)
# ═══════════════════════════════════════════════════════════════════════════════

FUNDS_PER_PLOT_PAGE = 6   # 3 rows × 2 cols — large enough to read clearly

def _draw_fund_chart(fig, outer, slot, res, rets, mon):
    """Draw one fund's two-panel chart into a GridSpec slot."""
    row, col = divmod(slot, 2)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, col],
        hspace=0.0, height_ratios=[1, 1.3])
    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1], sharex=ax1)

    df    = mon.history_df
    x     = df["t"].values
    dates = rets.index

    # Panel 1: cumulative excess return
    ax1.plot(x, df["excess_return"].cumsum(), lw=1.2, color=NAVY)
    ax1.axhline(0, color=GREY, lw=0.6)
    ax1.set_facecolor(LGREY)
    ax1.tick_params(labelbottom=False, labelsize=6)
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.set_title(f"{res['fund']} vs {res['bench']}", fontsize=8.5,
                  fontweight="bold", color=NAVY, pad=3)
    ax1.grid(alpha=0.3, lw=0.5)

    # Panel 2: lower CUSUM (L) and upper CUSUM (L_up)
    ax2.plot(x, df["L"],    lw=1.2, color=NAVY, label="L (loss)")
    ax2.plot(x, df["L_up"], lw=0.9, color=GREEN, ls="--", label="L_up (gain)")
    ax2.axhline(mon.threshold, color=RED, ls="--", lw=0.8,
                label=f"h={mon.threshold}")
    ax2.invert_yaxis()
    ax2.set_facecolor(LGREY)
    ax2.grid(alpha=0.3, lw=0.5)

    # Mark all skill-loss alarm lines
    for at in mon.all_alarms:
        ax2.axvline(at, color=RED, lw=0.7, alpha=0.5)
    if mon.alarm_t and dates is not None and mon.alarm_t <= len(dates):
        d_str = dates[mon.alarm_t - 1].strftime("%b'%y")
        ax2.text(mon.alarm_t + len(x) * 0.02,
                 mon.threshold * 0.55, f"^{d_str}",
                 color=RED, fontsize=6, fontweight="bold")

    # x-axis date labels
    if dates is not None and len(dates) >= len(x):
        tick_locs = [t for t in ax2.get_xticks() if 0 < int(t) <= len(dates)]
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(
            [dates[int(t) - 1].strftime("%Y") for t in tick_locs],
            fontsize=6, rotation=45)

    # Stats annotation
    n_al   = len(mon.all_alarms)
    status = f"ALARM {res['alarm_date']} ({n_al}x)" if res["alarm"] else "No Alarm"
    s_col  = RED if res["alarm"] else GREEN
    ax1.text(0.98, 0.95,
             f"IR={res['ir']:+.2f}  TE={res['te']:.1%}\n{status}",
             transform=ax1.transAxes, fontsize=6.5, ha="right", va="top",
             color=s_col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                       edgecolor=s_col, alpha=0.9))


def page_fund_plots(pdf, all_rets, all_monitors, all_results):
    """Renders individual fund CUSUM charts, 6 per page."""
    n_funds      = len(all_results)
    n_plot_pages = int(np.ceil(n_funds / FUNDS_PER_PLOT_PAGE))

    for pp in range(n_plot_pages):
        chunk_res  = all_results[pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]
        chunk_rets = all_rets   [pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]
        chunk_mon  = all_monitors[pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]

        n_in_chunk = len(chunk_res)
        n_rows     = int(np.ceil(n_in_chunk / 2))

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)
        fig.text(0.5, 0.980,
                 f"Individual Fund CUSUM Charts — page {pp+1} of {n_plot_pages}",
                 ha="center", fontsize=12, fontweight="bold", color=NAVY)

        outer = gridspec.GridSpec(n_rows, 2, figure=fig,
                                  left=0.06, right=0.97,
                                  top=0.960, bottom=0.05,
                                  hspace=0.70, wspace=0.30)

        for slot, (res, rets, mon) in enumerate(zip(chunk_res, chunk_rets, chunk_mon)):
            _draw_fund_chart(fig, outer, slot, res, rets, mon)

        page_footer(fig, 7)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_html(all_results, all_post_alarm, output_path="cusum_report.html"):
    """Generate a self-contained HTML report with embedded base64 images."""

    def tbl_row(cells, header=False, colors=None):
        tag   = "th" if header else "td"
        cols  = colors or [""] * len(cells)
        cells_html = "".join(
            f'<{tag} style="color:{c};font-weight:{"bold" if c else "normal"}">'
            f'{v}</{tag}>'
            for v, c in zip(cells, cols)
        )
        return f"<tr>{cells_html}</tr>"

    # Summary table HTML
    summary_rows = tbl_row(
        ["Fund", "Bench", "Months", "Ann TE", "IR", "Cum Excess", "#Alarms", "Alarm?", "First Alarm"],
        header=True)
    for res in all_results:
        alarm_col = "#C0392B" if res["alarm"] else "#1E8449"
        ir_col    = "#1E8449" if res["ir"] > 0.3 else ("#C0392B" if res["ir"] < 0.1 else "")
        summary_rows += tbl_row(
            [res["fund"], res["bench"], res["months"],
             f"{res['te']:.1%}", f"{res['ir']:+.2f}",
             f"{res['cum_excess']:+.1%}", res.get("n_alarms", "-"),
             "YES" if res["alarm"] else "NO", res["alarm_date"]],
            colors=["", "", "", "", ir_col, "", "", alarm_col, alarm_col])

    # Post-alarm table HTML
    pa_rows = tbl_row(
        ["Fund", "Alarm Date", "+12M Fund", "+12M Bench", "+12M Excess",
         "+24M Excess", "+36M Excess", "Signal"],
        header=True)

    def _fmt(v):
        return f"{v:+.1%}" if v == v else "n/a"

    for res, pa_df in zip(all_results, all_post_alarm):
        if not res["alarm"] or pa_df is None or pa_df.empty:
            continue
        first_t = pa_df["alarm_t"].min()
        fa      = pa_df[pa_df["alarm_t"] == first_t]
        r12 = fa[fa["horizon"] == 12].iloc[0] if not fa[fa["horizon"] == 12].empty else None
        r24 = fa[fa["horizon"] == 24].iloc[0] if not fa[fa["horizon"] == 24].empty else None
        r36 = fa[fa["horizon"] == 36].iloc[0] if not fa[fa["horizon"] == 36].empty else None
        if r12 is None:
            continue
        sig_col = "#1E8449" if r12["correct_alarm"] else "#C0392B"
        sig_txt = "CORRECT" if r12["correct_alarm"] else "FALSE"
        def ec(v): return "#C0392B" if v == v and v < 0 else "#1E8449"
        pa_rows += tbl_row(
            [res["fund"], res["alarm_date"],
             _fmt(r12["fund_cum"]), _fmt(r12["bench_cum"]),
             _fmt(r12["excess_cum"]),
             _fmt(r24["excess_cum"] if r24 is not None else np.nan),
             _fmt(r36["excess_cum"] if r36 is not None else np.nan),
             sig_txt],
            colors=["", "", "", "", ec(r12["excess_cum"]),
                    ec(r24["excess_cum"] if r24 is not None else np.nan),
                    ec(r36["excess_cum"] if r36 is not None else np.nan),
                    sig_col])

    css = """
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto;
           color: #333; background: #f9f9f9; }
    h1   { color: #1E2761; border-bottom: 3px solid #D4AC0D; padding-bottom: 8px; }
    h2   { color: #1E2761; margin-top: 40px; }
    table{ border-collapse: collapse; width: 100%; margin-bottom: 30px;
           background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    th   { background: #1E2761; color: white; padding: 8px 10px;
           text-align: left; font-size: 13px; }
    td   { padding: 7px 10px; font-size: 12px; border-bottom: 1px solid #eee; }
    tr:nth-child(even) td { background: #f4f6f7; }
    img  { max-width: 100%; margin: 20px 0; border: 1px solid #ddd;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .footer { color: #999; font-size: 11px; margin-top: 40px; text-align: center; }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CUSUM Active-Manager Monitor</title>
  <style>{css}</style>
</head>
<body>
  <h1>CUSUM Active-Manager Monitor</h1>
  <p><em>Generated: {datetime.today().strftime('%d %B %Y')}</em></p>

  <h2>Summary Results</h2>
  <table>{summary_rows}</table>

  <h2>Post-Alarm Analysis</h2>
  <table>{pa_rows}</table>

  <p class="footer">
    Reference: Philips, T., Yashchin, E. &amp; Stein, D. (2003).
    <em>Using Statistical Process Control to Monitor Active Managers.</em>
    Journal of Portfolio Management, 30(1), 86-94.
  </p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

class MonitorResult:
    """Thin wrapper attaching history_df to a CUSUMMonitor."""
    def __init__(self, monitor, history_df):
        for k, v in vars(monitor).items():
            setattr(self, k, v)
        self.history_df = history_df


def collect_data():
    all_rets, all_monitors, all_results, all_post_alarm = [], [], [], []
    print("Fetching data from Yahoo Finance...")
    for ticker, (fund, bench, name) in PRESET_PAIRS.items():
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            rets = load_from_yfinance(fund, bench)

            seed       = rets.iloc[: min(24, max(6, len(rets) // 4))]
            excess_s   = seed["portfolio"] - seed["benchmark"]
            init_te    = excess_s.std() * np.sqrt(12)

            monitor    = CUSUMMonitor(initial_te=init_te, threshold=19.81)
            history_df = monitor.run(rets)

            excess     = rets["portfolio"] - rets["benchmark"]
            ir         = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
            te         = excess.std() * np.sqrt(12)
            cum        = (1 + rets["portfolio"]).prod() / (1 + rets["benchmark"]).prod() - 1
            ir_pt, ir_lo, ir_hi = bootstrap_ir_ci(rets)

            alarm_date = (rets.index[monitor.alarm_t - 1].strftime("%b %Y")
                          if monitor.alarm else "-")

            pa_df = post_alarm_returns(rets, monitor)

            all_rets.append(rets)
            mr = MonitorResult(monitor, history_df)
            all_monitors.append(mr)
            all_results.append({
                "fund": fund, "bench": bench, "name": name,
                "months":    len(rets),
                "ir":        ir,  "ir_lo": ir_lo, "ir_hi": ir_hi,
                "te":        te,
                "cum_excess": cum,
                "n_alarms":  len(monitor.all_alarms),
                "alarm":     monitor.alarm,
                "alarm_date": alarm_date,
            })
            all_post_alarm.append(pa_df)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    return all_rets, all_monitors, all_results, all_post_alarm


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CUSUM Report Generator")
    parser.add_argument("--output", default="cusum_report.pdf",
                        help="Output PDF path (default: cusum_report.pdf)")
    parser.add_argument("--html",   action="store_true",
                        help="Also generate self-contained HTML export")
    args = parser.parse_args()

    all_rets, all_monitors, all_results, all_post_alarm = collect_data()

    print(f"\nBuilding PDF -> {args.output}")
    with PdfPages(args.output) as pdf:
        page_cover(pdf)                                        # 1
        page_methodology(pdf)                                  # 2
        page_fund_universe(pdf)                                # 3
        page_summary(pdf, all_results)                         # 4
        page_post_alarm(pdf, all_results, all_post_alarm)      # 5
        page_waterfall(pdf, all_results)                       # 6
        page_fund_plots(pdf, all_rets, all_monitors, all_results)  # 7

    print(f"  PDF saved: {args.output}")

    if args.html:
        export_html(all_results, all_post_alarm)

    print("Done.")


if __name__ == "__main__":
    main()
