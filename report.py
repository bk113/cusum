"""
CUSUM Active-Manager Monitor — PDF Report Generator
=====================================================
Page structure:
  1.    Cover
  2.    Executive Summary             (KPIs · asset-class table · key findings)
  3.    Methodology                   (step-by-step CUSUM walkthrough)
  4.    Model Assumptions & Parameters
  5–N.  Fund Universe                 (grouped by asset class, 20 items/page)
  N+.   Summary Results               (grouped by asset class, 18 items/page)
  M+.   Post-Alarm Analysis           (14 rows/page)
  P.    Waterfall Chart
  P+1+. Individual CUSUM Charts       (6/page)
  Last. Conclusions & Action Items    (REVIEW · WATCH · REGIME · CLEAN)

Run:
    python report.py
    python report.py --html
    python report.py --output my.pdf
"""

import base64
import io
import argparse
import math
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
    PRESET_PAIRS, THRESHOLD_TABLE, ASSET_CLASS_PRESETS,
    bootstrap_ir_ci, post_alarm_returns,
    compute_extra_stats, alarm_regime, signal_quality,
    MIN_MONITORING_MONTHS,
)

# ── Layout constants ───────────────────────────────────────────────────────────
FUNDS_PER_PLOT_PAGE = 6
UNIVERSE_PER_PAGE   = 20
SUMMARY_PER_PAGE    = 18
POST_ALARM_PER_PAGE = 14
CONCLUSION_PER_PAGE = 14

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY  = "#1E2761"
RED   = "#C0392B"
GOLD  = "#D4AC0D"
GREY  = "#BDC3C7"
LGREY = "#F4F6F7"
WHITE = "#FFFFFF"
GREEN = "#1E8449"
AMBER = "#E67E22"

# ── Asset class lookup ─────────────────────────────────────────────────────────
BENCH_TO_ASSET_CLASS = {
    "IWF":  "US Large Growth",
    "IWD":  "US Large Value",
    "SPY":  "US Large Blend",
    "IWM":  "US Small Cap",
    "EFA":  "Intl Developed",
    "EEM":  "Emerging Markets",
    "AOR":  "Balanced 60/40",
    "AGG":  "US Fixed Income",
    "EMB":  "EM Hard CCY Debt",
    "EMLC": "EM Local CCY Debt",
    "ACWI": "Global Equity",
    "URTH": "Global Equity",
}
TICKER_AC_OVERRIDES = {
    "FLPSX": "US Small/Mid Cap",
    "DODWX": "Intl Value",
}
ASSET_CLASS_COLORS = {
    "US Large Blend":    NAVY,
    "US Large Growth":   NAVY,
    "US Large Value":    NAVY,
    "US Small Cap":      NAVY,
    "US Small/Mid Cap":  NAVY,
    "Intl Developed":    "#1A5276",
    "Intl Value":        "#1A5276",
    "Emerging Markets":  "#1A5276",
    "Balanced 60/40":    "#145A32",
    "US Fixed Income":   "#145A32",
    "EM Hard CCY Debt":  "#6E2F8A",
    "EM Local CCY Debt": "#6E2F8A",
    "Global Equity":     "#7D6608",
}
ASSET_CLASS_ORDER = [
    "US Large Blend",
    "US Large Growth",
    "US Large Value",
    "US Small Cap",
    "US Small/Mid Cap",
    "Balanced 60/40",
    "US Fixed Income",
    "Intl Developed",
    "Intl Value",
    "Emerging Markets",
    "EM Hard CCY Debt",
    "EM Local CCY Debt",
    "Global Equity",
]
CLS_COLOR = {"REVIEW": RED, "WATCH": AMBER, "REGIME": GOLD, "CLEAN": GREEN}

# Maps each report asset class to the corresponding ASSET_CLASS_PRESETS key.
# All equity-like classes → "equity"; bond classes → "fixed_income"; mixed → "balanced".
AC_TO_PRESET: dict = {
    "US Large Blend":    "equity",
    "US Large Growth":   "equity",
    "US Large Value":    "equity",
    "US Small Cap":      "equity",
    "US Small/Mid Cap":  "equity",
    "Intl Developed":    "equity",
    "Intl Value":        "equity",
    "Emerging Markets":  "equity",
    "Global Equity":     "equity",
    "Balanced 60/40":    "balanced",
    "US Fixed Income":   "fixed_income",
    "EM Hard CCY Debt":  "fixed_income",
    "EM Local CCY Debt": "fixed_income",
}


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _asset_class(fund_ticker, bench):
    return TICKER_AC_OVERRIDES.get(fund_ticker,
           BENCH_TO_ASSET_CLASS.get(bench, "Other"))


def _ac_sort_key(ac):
    try:
        return ASSET_CLASS_ORDER.index(ac)
    except ValueError:
        return len(ASSET_CLASS_ORDER)


def _classify_alarm(res, pa_df):
    """Return 'REVIEW', 'WATCH', 'REGIME', or 'CLEAN'."""
    if not res["alarm"]:
        return "CLEAN"
    if res.get("regime") in ("GFC", "COVID"):
        return "REGIME"
    if pa_df is not None and not pa_df.empty:
        first_t = pa_df["alarm_t"].min()
        fa = pa_df[pa_df["alarm_t"] == first_t]
        row12 = fa[fa["horizon"] == 12]
        if not row12.empty:
            exc12 = row12.iloc[0]["excess_cum"]
            if exc12 == exc12:          # not NaN
                return "REVIEW" if exc12 < 0 else "WATCH"
    return "WATCH"                      # alarm raised; no post-alarm data yet


def _get_post12(pa_df):
    """Return cumulative excess at +12M for the first alarm, or NaN."""
    if pa_df is None or pa_df.empty:
        return float("nan")
    first_t = pa_df["alarm_t"].min()
    fa = pa_df[pa_df["alarm_t"] == first_t]
    row12 = fa[fa["horizon"] == 12]
    if row12.empty:
        return float("nan")
    return row12.iloc[0]["excess_cum"]


def _build_grouped_rows_universe():
    """Flat list of ('header', ac_str) | ('data', row_list) for the universe table."""
    all_funds = []
    for key, (fund, bench, name) in PRESET_PAIRS.items():
        ac = _asset_class(fund, bench)
        all_funds.append((ac, fund, name, bench))
    all_funds.sort(key=lambda x: (_ac_sort_key(x[0]), x[1]))

    rows, prev_ac, seq = [], None, 1
    for (ac, fund, name, bench) in all_funds:
        if ac != prev_ac:
            rows.append(("header", ac))
            prev_ac = ac
        rows.append(("data", [str(seq), fund, name, bench, ac, "19.81"]))
        seq += 1
    return rows


def _build_grouped_rows_summary(all_results):
    """Flat list of ('header', ac_str) | ('data', res_dict) for the summary table."""
    sorted_res = sorted(all_results,
                        key=lambda r: (_ac_sort_key(_asset_class(r["fund"], r["bench"])),
                                       r["fund"]))
    rows, prev_ac = [], None
    for r in sorted_res:
        ac = _asset_class(r["fund"], r["bench"])
        if ac != prev_ac:
            rows.append(("header", ac))
            prev_ac = ac
        rows.append(("data", r))
    return rows


def _paginate_grouped_rows(rows, per_page):
    """Split grouped rows into pages; prevents orphan headers at page bottoms."""
    pages, current, used = [], [], 0
    for kind, data in rows:
        if used >= per_page:
            pages.append(current); current = []; used = 0
        # Orphan prevention: don't let a header be the very last slot on a page
        if kind == "header" and used == per_page - 1:
            pages.append(current); current = []; used = 0
        current.append((kind, data)); used += 1
    if current:
        pages.append(current)
    return pages or [[]]


def _count_grouped_pages(rows, per_page):
    return max(1, len(_paginate_grouped_rows(rows, per_page)))


# ── Page-count helper ──────────────────────────────────────────────────────────

def compute_total_pages(all_results):
    n_loaded  = len(all_results)
    n_alarm   = sum(1 for r in all_results if r["alarm"])
    univ_rows = _build_grouped_rows_universe()
    summ_rows = _build_grouped_rows_summary(all_results)
    n_concl   = max(1, math.ceil(n_alarm / CONCLUSION_PER_PAGE))
    return (
        1                                                                    # cover
        + 1                                                                  # exec summary
        + 1                                                                  # methodology
        + 1                                                                  # assumptions
        + _count_grouped_pages(univ_rows, UNIVERSE_PER_PAGE)                # fund universe
        + _count_grouped_pages(summ_rows, SUMMARY_PER_PAGE)                 # summary
        + max(1, math.ceil(n_alarm   / POST_ALARM_PER_PAGE))                # post-alarm
        + 1                                                                  # waterfall
        + max(1, math.ceil(n_loaded  / FUNDS_PER_PLOT_PAGE))                # charts
        + n_concl                                                            # conclusion
    )


# ── Footer / utility ───────────────────────────────────────────────────────────

def page_footer(fig, page_num, total_pages):
    fig.text(0.5, 0.022,
             f"CUSUM Active-Manager Monitor  •  page {page_num} of {total_pages}",
             ha="center", va="bottom", fontsize=8, color="grey")
    fig.text(0.93, 0.022, datetime.today().strftime("%d %b %Y"),
             ha="right", va="bottom", fontsize=8, color="grey")
    fig.text(0.07, 0.022,
             "CONFIDENTIAL — For internal use only",
             ha="left", va="bottom", fontsize=7.5, color="grey", style="italic")


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Cover
# ═══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf, pnum, total_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(NAVY)

    fig.text(0.5, 0.91, "CUSUM Active-Manager Monitor",
             ha="center", fontsize=26, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.85, "Statistical Process Control for Portfolio Manager Oversight",
             ha="center", fontsize=13, color=GOLD)
    fig.text(0.5, 0.80, f"Report generated: {datetime.today().strftime('%d %B %Y')}",
             ha="center", fontsize=10, color=GREY)

    fig.add_artist(plt.Line2D([0.08, 0.92], [0.77, 0.77],
                              transform=fig.transFigure, color=GOLD, lw=1.2))

    fig.text(0.08, 0.74, "What is CUSUM?", fontsize=13, fontweight="bold", color=GOLD)
    fig.text(0.08, 0.69,
             "CUSUM (Cumulative Sum) is a sequential change-point detection method from "
             "statistical process control, adapted for investment management by Philips, "
             "Yashchin & Stein (2003). It answers one critical question:",
             fontsize=9, color=WHITE, va="top", linespacing=1.5)
    fig.text(0.10, 0.625,
             "Has a fund manager lost their investment skill — and if so, exactly when?",
             fontsize=10, color=GOLD, fontstyle="italic", va="top")

    fig.text(0.08, 0.585, "The Case for CUSUM — Why CIOs Should Care",
             fontsize=13, fontweight="bold", color=GOLD)

    reasons = [
        ("Cost of Inaction",
         "Retaining an underperforming manager for 3 extra years on a 500M mandate "
         "at IR=−0.3, TE=6% costs ~9M in excess losses. Early, systematic detection pays."),
        ("Faster than Rolling IR",
         "A trailing 3-year IR review needs 10+ years of data to distinguish skill from "
         "luck at 95% confidence. CUSUM detects persistent changes in IR within 2–3 years."),
        ("Rigorous False-Alarm Control",
         "Threshold h is calibrated from the paper's ARL table: h=19.81 gives ~1 false "
         "alarm per 60 months (5 years) when the true IR=+0.5. Fewer spurious terminations."),
        ("Two-Sided Monitoring",
         "Detects skill loss (lower CUSUM) and unexpected skill gain (upper CUSUM), "
         "enabling both termination decisions and early-stage manager promotions."),
        ("Rolling Restart",
         "After each alarm L resets to zero — monitoring never stops. Multiple alarms "
         "build a richer signal; post-alarm returns validate each flag."),
        ("Actionable, Not Automatic",
         "An alarm triggers a structured review — not automatic termination. It forces "
         "the investment committee to ask: Is the alpha source still intact?"),
    ]
    y = 0.545
    for title, body in reasons:
        fig.text(0.10, y, f"  {title}:", fontsize=8.5, fontweight="bold", color=GOLD)
        fig.text(0.10, y - 0.028, f"  {body}", fontsize=7.8, color=GREY,
                 va="top", linespacing=1.4)
        y -= 0.068

    fig.text(0.5, 0.025,
             "Reference: Philips, T., Yashchin, E. & Stein, D. (2003). "
             "\"Using Statistical Process Control to Monitor Active Managers.\" "
             "Journal of Portfolio Management, 30(1), 86–94.",
             ha="center", fontsize=7.5, color=GREY, style="italic")

    page_footer(fig, pnum[0], total_pages)
    pnum[0] += 1
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Executive Summary
# ═══════════════════════════════════════════════════════════════════════════════

def page_executive_summary(pdf, all_results, all_post_alarm, pnum, total_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.962, "Executive Summary", ha="center",
             fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.932, f"CUSUM Active-Manager Monitor  •  {datetime.today().strftime('%d %B %Y')}",
             ha="center", fontsize=11, color="grey")

    # ── Compute headline stats ────────────────────────────────────────────────
    n_total = len(all_results)
    n_alarm = sum(1 for r in all_results if r["alarm"])
    n_clean = n_total - n_alarm
    gfc_alm = sum(1 for r in all_results if r.get("regime") == "GFC")
    covid_alm = sum(1 for r in all_results if r.get("regime") == "COVID")
    exp_fa = round(n_total * 12 / 60, 1)

    classifications = {(r["fund"], r["bench"]): _classify_alarm(r, pa)
                       for r, pa in zip(all_results, all_post_alarm)}
    n_review = sum(1 for v in classifications.values() if v == "REVIEW")
    n_regime = sum(1 for v in classifications.values() if v == "REGIME")

    best_ir   = max(all_results, key=lambda r: r["ir"])
    worst_ir  = min(all_results, key=lambda r: r["ir"])
    most_alm  = max(all_results, key=lambda r: r.get("n_alarms", 0))

    # ── KPI boxes ─────────────────────────────────────────────────────────────
    kpis = [
        (str(n_total),  "Funds\nMonitored",         NAVY),
        (str(n_alarm),  "CUSUM\nAlarms",             RED   if n_alarm  > 0 else GREEN),
        (str(n_review), "Require\nReview",           AMBER if n_review > 0 else GREEN),
        (str(n_clean),  "No Alarm\n(Clean)",         GREEN),
    ]
    box_xs  = [0.030, 0.280, 0.530, 0.775]
    box_w   = 0.215
    box_top = 0.895
    box_bot = 0.790

    for (val, label, col), bx in zip(kpis, box_xs):
        fig.add_artist(FancyBboxPatch((bx, box_bot), box_w, box_top - box_bot,
                                      boxstyle="round,pad=0.008",
                                      facecolor=col, edgecolor="none",
                                      transform=fig.transFigure, alpha=0.10))
        fig.add_artist(FancyBboxPatch((bx, box_bot), box_w, box_top - box_bot,
                                      boxstyle="round,pad=0.008",
                                      facecolor="none", edgecolor=col, lw=2,
                                      transform=fig.transFigure))
        mid = bx + box_w / 2
        fig.text(mid, box_bot + (box_top - box_bot) * 0.62, val,
                 ha="center", va="center", fontsize=26, fontweight="bold", color=col)
        fig.text(mid, box_bot + (box_top - box_bot) * 0.22, label,
                 ha="center", va="center", fontsize=8.5, color="#555555", linespacing=1.4)

    # ── Asset class summary table ──────────────────────────────────────────────
    tbl_top = 0.765
    fig.text(0.030, tbl_top, "Performance Summary by Asset Class",
             fontsize=10, fontweight="bold", color=NAVY)

    ac_hdrs  = ["Asset Class", "Funds", "Alarmed", "Avg IR", "Hit%", "Best Fund", "Signal"]
    ac_col_x = [0.030, 0.300, 0.370, 0.440, 0.510, 0.575, 0.790]

    hdr_y = tbl_top - 0.027
    fig.add_artist(FancyBboxPatch((0.025, hdr_y - 0.007), 0.95, 0.022,
                                   boxstyle="round,pad=0.002", facecolor=NAVY,
                                   edgecolor="none", transform=fig.transFigure))
    for hdr, x in zip(ac_hdrs, ac_col_x):
        fig.text(x, hdr_y, hdr, fontsize=7.5, fontweight="bold", color=WHITE, va="center")

    ac_groups = {}
    for r in all_results:
        ac_groups.setdefault(_asset_class(r["fund"], r["bench"]), []).append(r)

    row_h = 0.026
    y = hdr_y - 0.010
    for i, ac in enumerate(ASSET_CLASS_ORDER):
        grp = ac_groups.get(ac)
        if not grp:
            continue
        n_g  = len(grp)
        n_ga = sum(1 for r in grp if r["alarm"])
        avg_ir  = float(np.mean([r["ir"] for r in grp]))
        avg_hit = float(np.mean([r.get("hit_rate") or 0.5 for r in grp]))
        best_g  = max(grp, key=lambda r: r["ir"])

        grp_cls_vals = [classifications.get((r["fund"], r["bench"]), "CLEAN") for r in grp]
        grp_cls = ("REVIEW" if "REVIEW" in grp_cls_vals else
                   ("WATCH" if "WATCH" in grp_cls_vals else
                    ("REGIME" if "REGIME" in grp_cls_vals else "CLEAN")))

        y -= row_h
        bg = LGREY if i % 2 == 0 else WHITE
        fig.add_artist(FancyBboxPatch((0.025, y - 0.004), 0.95, row_h - 0.002,
                                      boxstyle="round,pad=0.002", facecolor=bg,
                                      edgecolor="none", transform=fig.transFigure))

        ac_col  = ASSET_CLASS_COLORS.get(ac, NAVY)
        alm_col = RED if n_ga > 0 else GREEN
        ir_col  = GREEN if avg_ir > 0 else RED
        cls_col = CLS_COLOR.get(grp_cls, NAVY)

        row_vals  = [ac, str(n_g), str(n_ga) if n_ga else "—",
                     f"{avg_ir:+.2f}", f"{avg_hit:.0%}",
                     f"{best_g['fund']} ({best_g['ir']:+.2f})", grp_cls]
        row_colors= [ac_col, NAVY, alm_col, ir_col, NAVY, NAVY, cls_col]

        for val, x, col in zip(row_vals, ac_col_x, row_colors):
            fw = "bold" if col != NAVY else "normal"
            fig.text(x, y + row_h * 0.22, val, fontsize=7.5, va="center",
                     color=col, fontweight=fw)

    # ── Key Findings box ────────────────────────────────────────────────────────
    box_bot_y = y - row_h - 0.008
    avail_h   = max(0.06, box_bot_y - 0.048)

    ax_find = fig.add_axes([0.025, 0.045, 0.95, avail_h])
    ax_find.set_facecolor("#EBF5FB")
    for s in ax_find.spines.values():
        s.set_visible(False)
    ax_find.set_xticks([]); ax_find.set_yticks([])
    ax_find.text(0.5, 0.97, "Key Findings", ha="center", va="top",
                 fontsize=9, fontweight="bold", color=NAVY, transform=ax_find.transAxes)

    findings = [
        f"{n_alarm} of {n_total} funds ({n_alarm/n_total:.0%}) triggered a CUSUM alarm between Jan 2005 and today.",
        f"Regime alarms — GFC (Oct'07–Mar'09): {gfc_alm}  |  COVID (Feb–Apr'20): {covid_alm}  — these reflect market stress, not skill loss.",
        f"{n_review} fund(s) classified REVIEW (confirmed by negative post-alarm return at +12M); {n_alarm - n_review - n_regime} classified WATCH.",
        f"Best risk-adjusted manager: {best_ir['fund']} (IR {best_ir['ir']:+.2f}, t={best_ir.get('t_stat', 0):+.1f}).  "
        f"Weakest: {worst_ir['fund']} (IR {worst_ir['ir']:+.2f}).",
        f"Multiple testing: at ARL=60m per fund, expect ~{exp_fa} false alarms/year across {n_total} funds — see Summary section for detail.",
    ]
    fy = 0.82
    dy = min(0.155, 0.82 / max(len(findings), 1))
    for f in findings:
        ax_find.text(0.015, fy, f"  \u2022  {f}", va="top", fontsize=8.0, color="#333333",
                     transform=ax_find.transAxes, linespacing=1.35)
        fy -= dy

    page_footer(fig, pnum[0], total_pages)
    pnum[0] += 1
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════

def page_methodology(pdf, pnum, total_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)
    fig.text(0.5, 0.96, "How CUSUM Works", ha="center", fontsize=20,
             fontweight="bold", color=NAVY)
    fig.text(0.5, 0.932, "Step-by-step mechanics of the Philips-Yashchin-Stein procedure",
             ha="center", fontsize=11, color="grey")

    steps = [
        ("Step 1 — Log Excess Return",
         "et = ln[(1 + r_fund) / (1 + r_benchmark)]",
         "Log returns are additive and symmetric — they handle compounding correctly."),
        ("Step 2 — EWMA Tracking-Error Estimate",
         "sigma²t = gamma * sigma²(t-1) + (1-gamma) * 0.5*(et - e(t-1))²   [gamma=0.90]",
         "Von Neumann adjacent-difference estimator: robust to drift in the mean, "
         "adapts as the manager's risk budget changes."),
        ("Step 3 — Monthly IR Estimate",
         "IR_hat_t = et / sigma(t-1)   [monthly scale, std \u2248 1 under H\u2080; no clipping]",
         "Monthly IR using lagged sigma. If |IR_hat| > regime_threshold (default 1.0, \u223c3\u20134\u03c3), "
         "the month is a regime outlier: L and L_up are frozen (not accumulated) and the "
         "index is recorded. Cleaner than winsorisation — the raw IR is preserved."),
        ("Step 4 — 24-Month Calibration Gate",
         "First 24 months: sigma EWMA runs; L frozen at 0.",
         "Prevents in-sample contamination: the same data used to initialise sigma "
         "is not also tested against it."),
        ("Step 5 — Lindley Recursion (Page's CUSUM)",
         "L_t = max(0, L(t-1) + 0.25 - IR_hat_t)     [drift = 0.5*(IR_good+IR_bad) - IR_hat]",
         "L grows when the manager underperforms, resets to 0 when outperforming. "
         "Upper CUSUM L_up uses drift_up = IR_hat - 0.5*(IR_exceptional + IR_good)."),
        ("Step 6 — Rolling Alarm Rule",
         "ALARM if L_t >= h = 19.81   then reset L_t = 0 and continue",
         "After each alarm L resets to zero — monitoring never stops. "
         "All alarm times are recorded; post-alarm returns validate each signal."),
    ]

    y = 0.885
    for i, (title, formula, explanation) in enumerate(steps):
        ax_b = fig.add_axes([0.055, y - 0.004, 0.027, 0.034])
        ax_b.add_patch(plt.Circle((0.5, 0.5), 0.5, color=NAVY))
        ax_b.text(0.5, 0.5, str(i + 1), ha="center", va="center",
                  fontsize=10, fontweight="bold", color=WHITE)
        ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")
        fig.text(0.095, y, title, fontsize=10, fontweight="bold", color=NAVY)
        fig.text(0.095, y - 0.026, formula, fontsize=8.5, color=RED, fontfamily="monospace")
        fig.text(0.095, y - 0.048, explanation, fontsize=8.2, color="#555555",
                 va="top", linespacing=1.4)
        y -= 0.132

    fig.text(0.08, 0.118, "Alarm Threshold Reference Table (Table 2, Philips et al. 2003)",
             fontsize=10, fontweight="bold", color=NAVY)
    col_labels = ["Threshold (h)", "ARL — Good (IR=0.5)", "ARL — Flat (IR=0)", "ARL — Bad (IR=−0.5)"]
    col_x = [0.08, 0.28, 0.50, 0.70]
    for lbl, x in zip(col_labels, col_x):
        fig.text(x, 0.099, lbl, fontsize=8, fontweight="bold", color=WHITE,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, edgecolor="none"))
    for r, row in enumerate(THRESHOLD_TABLE.values.tolist()):
        bg  = LGREY if r % 2 == 0 else WHITE
        yrow = 0.082 - r * 0.021
        fig.add_artist(FancyBboxPatch((0.075, yrow - 0.007), 0.85, 0.019,
                                      boxstyle="round,pad=0.002",
                                      facecolor=bg, edgecolor="none",
                                      transform=fig.transFigure))
        hl = (abs(row[0] - 19.81) < 0.01)
        for val, x in zip(row, col_x):
            label = f"{val:.2f}" if isinstance(val, float) else f"{int(val)} months"
            fig.text(x, yrow, label, fontsize=8,
                     color=RED if hl else "#333333",
                     fontweight="bold" if hl else "normal")
        if hl:
            fig.text(0.90, yrow, "<- default", fontsize=7, color=RED, style="italic")

    page_footer(fig, pnum[0], total_pages)
    pnum[0] += 1
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Assumptions & Parameters
# ═══════════════════════════════════════════════════════════════════════════════

def page_assumptions(pdf, pnum, total_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    fig.text(0.5, 0.962, "Model Assumptions & Parameters", ha="center",
             fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.932, "Philips-Yashchin-Stein (2003) — implementation reference",
             ha="center", fontsize=11, color="grey")

    # Vertical divider
    fig.add_artist(plt.Line2D([0.508, 0.508], [0.055, 0.905],
                              transform=fig.transFigure, color=LGREY, lw=1.5))

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    fig.text(0.040, 0.898, "Algorithm Parameters",
             fontsize=11, fontweight="bold", color=NAVY)

    params = [
        ("Threshold (h)",       "19.81",      "ARL ≈ 60 months under H₀ at IR = +0.5"),
        ("EWMA decay (γ)",      "0.90",        "Von Neumann estimator; ~10-month half-life"),
        ("μ_good  (IR floor)",  "+0.50",       "Signifies 'acceptable' manager skill"),
        ("μ_bad   (IR floor)",  " 0.00",       "Signifies 'underperforming' manager"),
        ("μ_exceptional",       "+1.00",       "Upper-CUSUM trigger for skill gain"),
        ("Calibration window",  "24 months",   "Sigma warmup only; L frozen at zero"),
        ("Regime threshold",    "±1.0",        "If |IR_hat_monthly|>1: freeze L, record outlier month (~3\u20134\u03c3)"),
        ("Bootstrap blocks",    "n^(1/3)",     "Circular block (Politis-White 2004)"),
        ("Min. monitoring",     "36 months",   "Below this, results flagged * (low conf.)"),
    ]

    p_col_x = [0.040, 0.185, 0.260]
    hdr_y   = 0.866
    fig.add_artist(FancyBboxPatch((0.035, hdr_y - 0.007), 0.458, 0.021,
                                   boxstyle="round,pad=0.002", facecolor=NAVY,
                                   edgecolor="none", transform=fig.transFigure))
    for hdr, x in zip(["Parameter", "Value", "Rationale"], p_col_x):
        fig.text(x, hdr_y, hdr, fontsize=8, fontweight="bold", color=WHITE, va="center")

    ROW_P = 0.052
    py = hdr_y - 0.012
    for i, (param, val, rationale) in enumerate(params):
        py -= ROW_P
        bg = LGREY if i % 2 == 0 else WHITE
        fig.add_artist(FancyBboxPatch((0.035, py - 0.008), 0.458, ROW_P - 0.004,
                                      boxstyle="round,pad=0.002", facecolor=bg,
                                      edgecolor="none", transform=fig.transFigure))
        fig.text(p_col_x[0], py + ROW_P * 0.38, param,    fontsize=7.5, va="center", color=NAVY)
        fig.text(p_col_x[1], py + ROW_P * 0.38, val,      fontsize=7.5, va="center",
                 color=RED, fontweight="bold", fontfamily="monospace")
        fig.text(p_col_x[2], py + ROW_P * 0.20, rationale, fontsize=7.0, va="center",
                 color="#555555", linespacing=1.3)

    # Alarm classification guide
    cls_y = py - 0.042
    fig.text(0.040, cls_y, "Alarm Classification Guide",
             fontsize=10, fontweight="bold", color=NAVY)
    cls_items = [
        (RED,   "REVIEW",  "Alarm confirmed — fund underperformed at +12M. Schedule IC review."),
        (AMBER, "WATCH",   "Alarm raised; inconclusive post-alarm data or fund recovered."),
        (GOLD,  "REGIME",  "Alarm in GFC or COVID window. Likely market-driven, not skill loss."),
        (GREEN, "CLEAN",   "No alarm. Monitor at standard quarterly cadence."),
    ]
    cy = cls_y - 0.030
    for col, cls, desc in cls_items:
        fig.add_artist(FancyBboxPatch((0.038, cy - 0.005), 0.015, 0.016,
                                      boxstyle="round,pad=0.001", facecolor=col,
                                      edgecolor="none", transform=fig.transFigure))
        fig.text(0.060, cy + 0.003, f"{cls}:", fontsize=8, fontweight="bold", color=col, va="center")
        fig.text(0.115, cy + 0.003, desc, fontsize=7.5, color="#333333", va="center")
        cy -= 0.034

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    fig.text(0.522, 0.898, "Data Source & Coverage",
             fontsize=11, fontweight="bold", color=NAVY)

    data_items = [
        ("Source",         "Yahoo Finance (yfinance) — no API key required"),
        ("Prices",         "Total-return adjusted close (dividends reinvested)"),
        ("Frequency",      "Daily prices → month-end to month-end returns"),
        ("Universe start", "January 2005 (common start for all funds)"),
        ("Returns used",   "Log excess: eₜ = ln[(1+r_fund) / (1+r_bench)]"),
        ("Benchmarks",     "SPY · IWF · IWD · IWM · EEM · AGG · EMB · EMLC · ACWI · URTH · AOR"),
    ]
    dy_step = 0.037
    ry = 0.856
    for label, val in data_items:
        fig.text(0.522, ry, f"{label}:", fontsize=8.5, color=NAVY, fontweight="bold", va="top")
        fig.text(0.648, ry, val, fontsize=8.2, color="#333333", va="top", linespacing=1.3)
        ry -= dy_step

    # What an alarm means
    wam_y = ry - 0.018
    fig.text(0.522, wam_y, "What an Alarm Means",
             fontsize=11, fontweight="bold", color=NAVY)
    fig.text(0.522, wam_y - 0.028,
             "A CUSUM alarm (L \u2265 h) indicates that accumulated evidence of "
             "underperformance is statistically significant. It should trigger:\n\n"
             "  \u2022  A structured investment-committee review\n"
             "  \u2022  A mandate compliance check with the portfolio manager\n"
             "  \u2022  NOT automatic termination\n\n"
             "L resets to 0 after each alarm (rolling restart). Monitoring never stops.",
             fontsize=8.5, color="#333333", va="top", linespacing=1.5)

    # Known limitations
    lim_y = wam_y - 0.190
    fig.text(0.522, lim_y, "Known Limitations",
             fontsize=11, fontweight="bold", color=NAVY)
    limitations = [
        "Multiple testing: 67 funds × ARL=60m → ~13 false alarms/year across universe.",
        "Yahoo Finance adjusted prices may differ from official NAV series.",
        "Singapore AIAIM funds use Morningstar proxy IDs (0P...) with limited history.",
        "Regime outlier months (|IR_hat_monthly|>1.0, e.g. March 2020): L frozen rather than clipped — no distortion.",
        "Calibration sigma uses first 24m of actual fund data (mild look-ahead).",
        "Single uniform threshold h=19.81; optimal h varies by mandate TE.",
    ]
    ly = lim_y - 0.030
    for i, lim in enumerate(limitations, 1):
        fig.text(0.522, ly, f"  {i}.  {lim}", fontsize=7.8, color="#333333",
                 va="top", linespacing=1.35)
        ly -= 0.036

    page_footer(fig, pnum[0], total_pages)
    pnum[0] += 1
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES — Fund Universe (grouped by asset class)
# ═══════════════════════════════════════════════════════════════════════════════

def page_fund_universe(pdf, pnum, total_pages):
    rows  = _build_grouped_rows_universe()
    pages = _paginate_grouped_rows(rows, UNIVERSE_PER_PAGE)
    total_univ_pages = len(pages)

    headers = ["#", "Ticker", "Fund Name", "Benchmark", "Asset Class", "h"]
    col_x   = [0.030, 0.068, 0.152, 0.590, 0.682, 0.862]

    DATA_ROW_H = 0.036
    HDR_ROW_H  = 0.024

    for fp, chunk in enumerate(pages):
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        subtitle = (f"{len(PRESET_PAIRS)} active managers grouped by asset class"
                    + (f"  —  page {fp+1} of {total_univ_pages}" if total_univ_pages > 1 else ""))
        fig.text(0.5, 0.962, "Fund Universe", ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.930, subtitle, ha="center", fontsize=11, color="grey")

        fig.add_artist(FancyBboxPatch((0.025, 0.878), 0.95, 0.026,
                                      boxstyle="round,pad=0.003", facecolor=NAVY,
                                      edgecolor="none", transform=fig.transFigure))
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=8.5, fontweight="bold",
                     color=WHITE, va="center")

        y = 0.862
        data_idx = 0
        for kind, data in chunk:
            if kind == "header":
                ac     = data
                ac_col = ASSET_CLASS_COLORS.get(ac, NAVY)
                fig.add_artist(FancyBboxPatch((0.025, y - HDR_ROW_H + 0.003),
                                              0.95, HDR_ROW_H - 0.004,
                                              boxstyle="round,pad=0.002",
                                              facecolor=ac_col, edgecolor="none",
                                              transform=fig.transFigure, alpha=0.15))
                fig.text(0.032, y - HDR_ROW_H * 0.45,
                         f"  \u25aa  {ac.upper()}",
                         fontsize=8, fontweight="bold", color=ac_col, va="center")
                y -= HDR_ROW_H
            else:
                row    = data   # [seq, fund, name, bench, ac, h]
                ac     = row[4]
                ac_col = ASSET_CLASS_COLORS.get(ac, NAVY)
                bg     = LGREY if data_idx % 2 == 0 else WHITE
                y -= DATA_ROW_H
                fig.add_artist(FancyBboxPatch((0.025, y - 0.005), 0.95, DATA_ROW_H - 0.003,
                                              boxstyle="round,pad=0.002", facecolor=bg,
                                              edgecolor="none", transform=fig.transFigure))
                disp   = [row[0], row[1], row[2], row[3], row[4], row[5]]
                d_cols = [NAVY, NAVY, NAVY, NAVY, ac_col, NAVY]
                for val, x, col in zip(disp, col_x, d_cols):
                    fig.text(x, y + DATA_ROW_H * 0.25, val,
                             fontsize=7.5, va="center", color=col)
                data_idx += 1

        page_footer(fig, pnum[0], total_pages)
        pnum[0] += 1
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES — Summary Results (grouped by asset class)
# ═══════════════════════════════════════════════════════════════════════════════

def page_summary(pdf, all_results, all_post_alarm, pnum, total_pages):
    # Build classification lookup
    cls_map = {(r["fund"], r["bench"]): _classify_alarm(r, pa)
               for r, pa in zip(all_results, all_post_alarm)}

    rows  = _build_grouped_rows_summary(all_results)
    pages = _paginate_grouped_rows(rows, SUMMARY_PER_PAGE)
    total_sum_pages = len(pages)

    # Stats for callout box
    alarm_funds  = [r for r in all_results if r["alarm"]]
    clean_funds  = [r for r in all_results if not r["alarm"]]
    n_funds      = len(all_results)
    exp_fa_yr    = round(n_funds * 12 / 60, 1)
    gfc_alarms   = sum(1 for r in alarm_funds if r.get("regime") == "GFC")
    covid_alarms = sum(1 for r in alarm_funds if r.get("regime") == "COVID")
    best_ir      = max(all_results, key=lambda x: x["ir"])
    worst_ir     = min(all_results, key=lambda x: x["ir"])

    headers = ["Fund", "Bench", "IR (95% CI)", "t-stat", "Hit%",
               "Up/Dn Cap", "Cum Exc", "First Alarm", "Signal"]
    col_x   = [0.030, 0.090, 0.148, 0.318, 0.378, 0.432, 0.535, 0.638, 0.750]

    DATA_ROW_H = 0.033
    HDR_ROW_H  = 0.022

    for sp, chunk in enumerate(pages):
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        subtitle = ("Risk-adjusted statistics — grouped by asset class, 24m calibration excluded"
                    + (f"  —  page {sp+1} of {total_sum_pages}" if total_sum_pages > 1 else ""))
        fig.text(0.5, 0.962, "Summary Results", ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.930, subtitle, ha="center", fontsize=9, color="grey")

        fig.add_artist(FancyBboxPatch((0.025, 0.878), 0.95, 0.026,
                                      boxstyle="round,pad=0.003", facecolor=NAVY,
                                      edgecolor="none", transform=fig.transFigure))
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=7, fontweight="bold", color=WHITE, va="center")

        y = 0.862
        data_idx = 0
        for kind, data in chunk:
            if kind == "header":
                ac     = data
                ac_col = ASSET_CLASS_COLORS.get(ac, NAVY)
                fig.add_artist(FancyBboxPatch((0.025, y - HDR_ROW_H + 0.003),
                                              0.95, HDR_ROW_H - 0.004,
                                              boxstyle="round,pad=0.002",
                                              facecolor=ac_col, edgecolor="none",
                                              transform=fig.transFigure, alpha=0.15))
                fig.text(0.032, y - HDR_ROW_H * 0.45,
                         f"  \u25aa  {ac.upper()}",
                         fontsize=7.5, fontweight="bold", color=ac_col, va="center")
                y -= HDR_ROW_H
            else:
                res    = data
                cls    = cls_map.get((res["fund"], res["bench"]), "CLEAN")
                cls_col= CLS_COLOR.get(cls, NAVY)
                bg     = LGREY if data_idx % 2 == 0 else WHITE
                y -= DATA_ROW_H
                fig.add_artist(FancyBboxPatch((0.025, y - 0.004), 0.95, DATA_ROW_H - 0.003,
                                              boxstyle="round,pad=0.002", facecolor=bg,
                                              edgecolor="none", transform=fig.transFigure))

                t   = res.get("t_stat") or 0
                ir_col = GREEN if t > 1.65 else (RED if res["ir"] < 0 else NAVY)

                ci_str  = (f"{res['ir']:+.2f} [{res.get('ir_lo', 0):+.2f},{res.get('ir_hi', 0):+.2f}]"
                           if "ir_lo" in res else f"{res['ir']:+.2f}")
                t_str   = f"{t:+.2f}" if res.get("t_stat") is not None else "-"
                hit_str = f"{res.get('hit_rate', 0):.0%}" if res.get("hit_rate") is not None else "-"

                uc = res.get("up_cap"); dc = res.get("dn_cap")
                cap_str = (f"{uc:.0%}/{dc:.0%}"
                           if uc is not None and dc is not None and uc == uc and dc == dc else "-")
                cap_col = (GREEN if uc is not None and dc is not None
                           and uc == uc and dc == dc and uc > 1.0 and dc < 1.0 else NAVY)

                fund_lbl = res["fund"] + ("*" if res.get("short_hist") else "")

                vals   = [fund_lbl, res["bench"], ci_str, t_str, hit_str,
                          cap_str, f"{res['cum_excess']:+.1%}",
                          res["alarm_date"], cls]
                colors = [NAVY if not res.get("short_hist") else "#7D6608",
                          NAVY, ir_col, ir_col, NAVY,
                          cap_col, NAVY, NAVY, cls_col]

                for val, x, col in zip(vals, col_x, colors):
                    fig.text(x, y + DATA_ROW_H * 0.25, val, fontsize=6.8, va="center",
                             color=col, fontweight="bold" if col != NAVY else "normal")
                data_idx += 1

        # Multiple-testing callout on last summary page
        if sp == total_sum_pages - 1:
            obs_y = max(0.035, y - 0.008)
            box_h = min(0.150, obs_y - 0.035)
            if box_h > 0.04:
                ax_obs = fig.add_axes([0.025, 0.035, 0.95, box_h])
                ax_obs.set_facecolor("#FDF2E9")
                for s in ax_obs.spines.values():
                    s.set_visible(False)
                ax_obs.set_xticks([]); ax_obs.set_yticks([])
                ax_obs.text(0.5, 0.97,
                            "Multiple-Testing Caution  |  Key Observations",
                            ha="center", va="top", fontsize=8.5, fontweight="bold",
                            color=RED, transform=ax_obs.transAxes)
                clean_tickers = ", ".join(r["fund"] for r in clean_funds[:6])
                extra = " ..." if len(clean_funds) > 6 else ""
                obs = (
                    f"  MULTIPLE TESTING: With {n_funds} funds at h=19.81 (ARL=60m/fund), "
                    f"expected FALSE alarms = {exp_fa_yr:.1f}/year across the universe. "
                    f"GFC: {gfc_alarms} alarm(s)  |  COVID: {covid_alarms} alarm(s) "
                    f"— treat as regime signals, not skill-loss signals.\n"
                    f"  * = Short monitoring history (<{MIN_MONITORING_MONTHS}m live CUSUM; treat with caution).  "
                    f"Best IR: {best_ir['fund']} ({best_ir['ir']:+.2f}, t={best_ir.get('t_stat', 0):+.1f}).  "
                    f"Weakest: {worst_ir['fund']} ({worst_ir['ir']:+.2f}).\n"
                    f"  Clean (no alarm): {clean_tickers}{extra}"
                )
                ax_obs.text(0.01, 0.72, obs, va="top", fontsize=7.5, color="#333333",
                            transform=ax_obs.transAxes, linespacing=1.55)

        page_footer(fig, pnum[0], total_pages)
        pnum[0] += 1
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES — Post-Alarm Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def page_post_alarm(pdf, all_results, all_post_alarm, pnum, total_pages):
    headers = ["Fund", "First Alarm", "+12M Fund", "+12M Bench", "+12M Exc",
               "+24M Exc", "+36M Exc", "1st Sig", "#Al", "Sig%(all)"]
    col_x   = [0.030, 0.105, 0.200, 0.285, 0.368, 0.455, 0.540, 0.625, 0.710, 0.775]

    def _fmt(v):
        return f"{v:+.1%}" if v == v else "n/a"

    display_rows = []
    for res, pa_df in zip(all_results, all_post_alarm):
        if not res["alarm"] or pa_df is None or pa_df.empty:
            continue
        first_t = pa_df["alarm_t"].min()
        fa      = pa_df[pa_df["alarm_t"] == first_t]
        row12   = fa[fa["horizon"] == 12]
        row24   = fa[fa["horizon"] == 24]
        row36   = fa[fa["horizon"] == 36]
        if row12.empty:
            continue
        d12   = row12.iloc[0]
        exc24 = row24.iloc[0]["excess_cum"] if not row24.empty else float("nan")
        exc36 = row36.iloc[0]["excess_cum"] if not row36.empty else float("nan")
        display_rows.append((res, d12, d12["excess_cum"], exc24, exc36, d12["correct_alarm"]))

    no_alarm   = [r["fund"] for r in all_results if not r["alarm"]]
    n_pa_pages = max(1, math.ceil(len(display_rows) / POST_ALARM_PER_PAGE))

    for pp in range(n_pa_pages):
        chunk = display_rows[pp * POST_ALARM_PER_PAGE:(pp + 1) * POST_ALARM_PER_PAGE]
        fig   = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        subtitle = ("Cumulative excess return in months following the FIRST alarm"
                    + (f"  —  page {pp+1} of {n_pa_pages}" if n_pa_pages > 1 else ""))
        fig.text(0.5, 0.962, "Post-Alarm Analysis", ha="center", fontsize=20,
                 fontweight="bold", color=NAVY)
        fig.text(0.5, 0.930, subtitle, ha="center", fontsize=10, color="grey")

        fig.add_artist(FancyBboxPatch((0.030, 0.878), 0.94, 0.026,
                                      boxstyle="round,pad=0.003", facecolor=NAVY,
                                      edgecolor="none", transform=fig.transFigure))
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=8, fontweight="bold", color=WHITE, va="center")

        row_h   = 0.048
        y_start = 0.845
        for r, (res, d12, exc12, exc24, exc36, correct) in enumerate(chunk):
            bg   = LGREY if r % 2 == 0 else WHITE
            yrow = y_start - r * row_h
            fig.add_artist(FancyBboxPatch((0.030, yrow - 0.013), 0.94, 0.043,
                                          boxstyle="round,pad=0.002", facecolor=bg,
                                          edgecolor="none", transform=fig.transFigure))

            sig_col  = GREEN if correct else RED
            sig_text = "CORRECT" if correct else "FALSE"
            e12_col  = RED   if exc12 == exc12 and exc12 < 0 else GREEN
            e24_col  = RED   if exc24 == exc24 and exc24 < 0 else GREEN
            e36_col  = RED   if exc36 == exc36 and exc36 < 0 else GREEN

            sq     = res.get("sig_quality")
            sq_str = f"{sq:.0%}" if sq is not None else "n/a"
            sq_col = (GREEN if sq is not None and sq > 0.6 else
                      (RED if sq is not None and sq < 0.4 else NAVY))

            vals   = [res["fund"], res["alarm_date"],
                      _fmt(d12["fund_cum"]), _fmt(d12["bench_cum"]),
                      _fmt(exc12), _fmt(exc24), _fmt(exc36),
                      sig_text, str(res.get("n_alarms", "-")), sq_str]
            colors = [NAVY, NAVY, NAVY, NAVY,
                      e12_col, e24_col, e36_col,
                      sig_col, NAVY, sq_col]

            for val, x, col in zip(vals, col_x, colors):
                fig.text(x, yrow + 0.008, val, fontsize=8, va="center",
                         color=col, fontweight="bold" if col != NAVY else "normal")

        if pp == n_pa_pages - 1:
            last_y = y_start - (len(chunk) - 1) * row_h
            note_y = max(0.175, last_y - 0.055)
            if no_alarm:
                fig.text(0.035, note_y,
                         f"No alarm: {', '.join(no_alarm)}",
                         fontsize=8.5, color=GREEN, fontstyle="italic")
            ax_exp = fig.add_axes([0.04, 0.035, 0.92, 0.130])
            ax_exp.set_facecolor(LGREY)
            for s in ax_exp.spines.values():
                s.set_visible(False)
            ax_exp.set_xticks([]); ax_exp.set_yticks([])
            ax_exp.text(0.5, 0.93, "How to Read This Table",
                        ha="center", va="top", fontsize=10, fontweight="bold",
                        color=NAVY, transform=ax_exp.transAxes)
            ax_exp.text(0.03, 0.68,
                        "  +12M/+24M/+36M: cumulative excess return (fund minus benchmark) after the FIRST alarm.\n"
                        "  1st Sig: CORRECT if fund underperformed after first alarm (signal validated). FALSE = false positive.\n"
                        "  Sig%(all): % of ALL alarms where fund underperformed at +12M. "
                        ">60% = reliable signal. 50% = coin flip.\n"
                        "  At h=19.81, false-alarm rate = 1/60m per fund. Across a 67-fund universe, expect ~13 false alarms/year.",
                        va="top", fontsize=8, color="#333333",
                        transform=ax_exp.transAxes, linespacing=1.65)

        page_footer(fig, pnum[0], total_pages)
        pnum[0] += 1
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Waterfall Chart
# ═══════════════════════════════════════════════════════════════════════════════

def page_waterfall(pdf, all_results, pnum, total_pages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)
    fig.text(0.5, 0.962, "Cumulative Excess Returns — Ranked", ha="center",
             fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.930, "Total return of each fund vs its benchmark (Jan 2005 – present)",
             ha="center", fontsize=11, color="grey")

    ax = fig.add_axes([0.24, 0.05, 0.69, 0.86])
    sorted_res = sorted(all_results, key=lambda x: x["cum_excess"])
    labels = [f"{r['fund']} vs {r['bench']}" for r in sorted_res]
    values = [r["cum_excess"] for r in sorted_res]
    colors = [GREEN if v >= 0 else RED for v in values]
    alarm  = [r["alarm"] for r in sorted_res]

    bar_h = max(0.3, min(0.7, 20 / max(len(sorted_res), 1)))
    bars  = ax.barh(labels, values, color=colors, height=bar_h,
                    edgecolor="white", lw=0.4)

    fs = max(5.5, min(8, 200 / max(len(sorted_res), 1)))
    for bar, val, alm in zip(bars, values, alarm):
        x_pos = val + 0.003 if val >= 0 else val - 0.003
        ha    = "left" if val >= 0 else "right"
        label = f"{val:+.1%}" + (" *" if alm else "")
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=fs,
                color=GREEN if val >= 0 else RED, fontweight="bold")

    ax.axvline(0, color=NAVY, lw=1.2)
    ax.set_xlabel("Cumulative excess return vs benchmark", fontsize=10, color=NAVY)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="y", labelsize=max(5, int(fs)))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_facecolor(LGREY)
    ax.grid(axis="x", alpha=0.4, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=GREEN, label="Outperformed benchmark"),
        mpatches.Patch(facecolor=RED,   label="Underperformed benchmark"),
        plt.Line2D([0], [0], color="none", label="* = CUSUM alarm triggered"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    page_footer(fig, pnum[0], total_pages)
    pnum[0] += 1
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES — Individual Fund Charts (6 per page, 3×2 grid)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_fund_chart(fig, outer, slot, res, rets, mon):
    row, col = divmod(slot, 2)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, col], hspace=0.0, height_ratios=[1, 1.3])
    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1], sharex=ax1)

    df    = mon.history_df
    x     = df["t"].values
    dates = rets.index

    ax1.plot(x, df["excess_return"].cumsum(), lw=1.2, color=NAVY)
    ax1.axhline(0, color=GREY, lw=0.6)
    ax1.set_facecolor(LGREY)
    ax1.tick_params(labelbottom=False, labelsize=6)
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.set_title(f"{res['fund']} vs {res['bench']}", fontsize=8.5,
                  fontweight="bold", color=NAVY, pad=3)
    ax1.grid(alpha=0.3, lw=0.5)

    ax2.plot(x, df["L"],    lw=1.2, color=NAVY,  label="L (loss)")
    ax2.plot(x, df["L_up"], lw=0.9, color=GREEN, ls="--", label="L_up (gain)")
    ax2.axhline(mon.threshold, color=RED, ls="--", lw=0.8, label=f"h={mon.threshold}")
    ax2.invert_yaxis()
    ax2.set_facecolor(LGREY)
    ax2.grid(alpha=0.3, lw=0.5)

    for at in mon.all_alarms:
        ax2.axvline(at, color=RED, lw=0.7, alpha=0.5)
    if mon.alarm_t and dates is not None and mon.alarm_t <= len(dates):
        d_str = dates[mon.alarm_t - 1].strftime("%b'%y")
        ax2.text(mon.alarm_t + len(x) * 0.02,
                 mon.threshold * 0.55, f"^{d_str}",
                 color=RED, fontsize=6, fontweight="bold")

    if dates is not None and len(dates) >= len(x):
        tick_locs = [t for t in ax2.get_xticks() if 0 < int(t) <= len(dates)]
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(
            [dates[int(t) - 1].strftime("%Y") for t in tick_locs],
            fontsize=6, rotation=45)

    n_al   = len(mon.all_alarms)
    status = f"ALARM {res['alarm_date']} ({n_al}x)" if res["alarm"] else "No Alarm"
    s_col  = RED if res["alarm"] else GREEN
    ax1.text(0.98, 0.95,
             f"IR={res['ir']:+.2f}  TE={res['te']:.1%}\n{status}",
             transform=ax1.transAxes, fontsize=6.5, ha="right", va="top",
             color=s_col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                       edgecolor=s_col, alpha=0.9))


def page_fund_plots(pdf, all_rets, all_monitors, all_results, pnum, total_pages):
    n_funds      = len(all_results)
    n_plot_pages = max(1, math.ceil(n_funds / FUNDS_PER_PLOT_PAGE))

    for pp in range(n_plot_pages):
        chunk_res  = all_results [pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]
        chunk_rets = all_rets    [pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]
        chunk_mon  = all_monitors[pp * FUNDS_PER_PLOT_PAGE:(pp + 1) * FUNDS_PER_PLOT_PAGE]
        n_rows     = math.ceil(len(chunk_res) / 2)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)
        fig.text(0.5, 0.985,
                 f"Appendix — Alarm frequency (informal IR proxy, not a calibrated metric)"
                 f"  [{pp+1} / {n_plot_pages}]",
                 ha="center", fontsize=11, fontweight="bold", color=NAVY)
        fig.text(0.5, 0.966,
                 "Multiple alarm markers visible only when rolling_restart=True. "
                 "Default (paper-faithful) mode produces at most one alarm per fund.",
                 ha="center", fontsize=7.5, color="#555555", style="italic")

        outer = gridspec.GridSpec(n_rows, 2, figure=fig,
                                  left=0.06, right=0.97,
                                  top=0.950, bottom=0.05,
                                  hspace=0.70, wspace=0.30)
        for slot, (res, rets, mon) in enumerate(zip(chunk_res, chunk_rets, chunk_mon)):
            _draw_fund_chart(fig, outer, slot, res, rets, mon)

        page_footer(fig, pnum[0], total_pages)
        pnum[0] += 1
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES — Conclusions & Action Items
# ═══════════════════════════════════════════════════════════════════════════════

def page_conclusion(pdf, all_results, all_post_alarm, pnum, total_pages):
    classified = [(r, pa, _classify_alarm(r, pa))
                  for r, pa in zip(all_results, all_post_alarm)]

    alarmed = [(r, pa, cls) for r, pa, cls in classified if r["alarm"]]
    clean   = [r for r, pa, cls in classified if not r["alarm"]]

    cls_order_map = {"REVIEW": 0, "WATCH": 1, "REGIME": 2}
    alarmed.sort(key=lambda x: (cls_order_map.get(x[2], 3),
                                 _ac_sort_key(_asset_class(x[0]["fund"], x[0]["bench"]))))

    n_concl_pages = max(1, math.ceil(len(alarmed) / CONCLUSION_PER_PAGE))

    headers = ["Fund", "Bench", "Asset Class", "IR", "t-stat",
               "#Alarms", "First Alarm", "Post +12M", "Classification"]
    col_x   = [0.030, 0.092, 0.152, 0.310, 0.368, 0.422, 0.482, 0.565, 0.648]

    n_funds = len(all_results)
    exp_fa  = round(n_funds * 12 / 60, 1)
    gfc_alm = sum(1 for r in all_results if r.get("regime") == "GFC")
    cov_alm = sum(1 for r in all_results if r.get("regime") == "COVID")
    n_rev   = sum(1 for _, _, c in classified if c == "REVIEW")

    for cp in range(n_concl_pages):
        chunk   = alarmed[cp * CONCLUSION_PER_PAGE:(cp + 1) * CONCLUSION_PER_PAGE]
        is_last = (cp == n_concl_pages - 1)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(WHITE)

        pg_lbl = f"  —  page {cp+1} of {n_concl_pages}" if n_concl_pages > 1 else ""
        fig.text(0.5, 0.962, "Conclusions & Action Items",
                 ha="center", fontsize=20, fontweight="bold", color=NAVY)
        fig.text(0.5, 0.930,
                 f"Alarm classification and recommended actions as of "
                 f"{datetime.today().strftime('%d %B %Y')}{pg_lbl}",
                 ha="center", fontsize=10, color="grey")

        fig.add_artist(FancyBboxPatch((0.025, 0.878), 0.95, 0.026,
                                      boxstyle="round,pad=0.003", facecolor=NAVY,
                                      edgecolor="none", transform=fig.transFigure))
        for hdr, x in zip(headers, col_x):
            fig.text(x, 0.884, hdr, fontsize=7, fontweight="bold", color=WHITE, va="center")

        ROW_H = 0.040
        y = 0.858

        for r_idx, (res, pa_df, cls) in enumerate(chunk):
            cls_col = CLS_COLOR.get(cls, NAVY)
            bg = LGREY if r_idx % 2 == 0 else WHITE
            y -= ROW_H
            fig.add_artist(FancyBboxPatch((0.025, y - 0.006), 0.95, ROW_H - 0.004,
                                          boxstyle="round,pad=0.002", facecolor=bg,
                                          edgecolor="none", transform=fig.transFigure))

            post12    = _get_post12(pa_df)
            p12_str   = f"{post12:+.1%}" if post12 == post12 else "n/a"
            p12_col   = RED if post12 == post12 and post12 < 0 else \
                        (GREEN if post12 == post12 else NAVY)
            ac        = _asset_class(res["fund"], res["bench"])
            t         = res.get("t_stat") or 0

            vals   = [res["fund"], res["bench"], ac,
                      f"{res['ir']:+.2f}", f"{t:+.1f}",
                      str(res.get("n_alarms", "-")),
                      res["alarm_date"], p12_str, cls]
            colors = [NAVY, NAVY, ASSET_CLASS_COLORS.get(ac, NAVY),
                      GREEN if res["ir"] > 0 else RED,
                      NAVY, NAVY, NAVY, p12_col, cls_col]

            for val, x, col in zip(vals, col_x, colors):
                fw = "bold" if col != NAVY else "normal"
                fig.text(x, y + ROW_H * 0.20, val, fontsize=7, va="center",
                         color=col, fontweight=fw)

        # Bottom section — only on last conclusion page
        if is_last:
            bottom_y = y - ROW_H * 0.4

            if clean:
                clean_line = ", ".join(r["fund"] for r in clean)
                cln_y = max(0.300, bottom_y - 0.005)
                fig.text(0.030, cln_y,
                         f"No alarm ({len(clean)} funds):  {clean_line}",
                         fontsize=7.5, color=GREEN, fontstyle="italic",
                         wrap=True)
                bottom_y = cln_y - 0.035

            box_y  = 0.048
            box_h  = min(0.155, max(0.080, bottom_y - box_y - 0.005))

            # Caveats (left)
            ax_cav = fig.add_axes([0.030, box_y, 0.455, box_h])
            ax_cav.set_facecolor("#FDEDEC")
            for s in ax_cav.spines.values():
                s.set_visible(False)
            ax_cav.set_xticks([]); ax_cav.set_yticks([])
            ax_cav.text(0.5, 0.96, "Statistical Caveats",
                        ha="center", va="top", fontsize=9, fontweight="bold",
                        color=RED, transform=ax_cav.transAxes)
            caveats = (
                f"  \u2022  Multiple testing: {n_funds} funds \u00d7 ARL=60m "
                f"\u2192 ~{exp_fa} expected false alarms/year.\n"
                f"  \u2022  GFC: {gfc_alm} alarm(s)  |  COVID: {cov_alm} alarm(s) "
                f"\u2014 regime signals, not skill loss.\n"
                f"  \u2022  Funds marked (*) have <{MIN_MONITORING_MONTHS}m of live CUSUM "
                f"\u2014 treat with caution.\n"
                f"  \u2022  An alarm triggers a structured REVIEW, NOT automatic termination."
            )
            ax_cav.text(0.02, 0.74, caveats, va="top", fontsize=7.8,
                        color="#333333", transform=ax_cav.transAxes, linespacing=1.55)

            # Next Steps (right)
            ax_next = fig.add_axes([0.515, box_y, 0.455, box_h])
            ax_next.set_facecolor("#EBF5FB")
            for s in ax_next.spines.values():
                s.set_visible(False)
            ax_next.set_xticks([]); ax_next.set_yticks([])
            ax_next.text(0.5, 0.96, "Recommended Next Steps",
                         ha="center", va="top", fontsize=9, fontweight="bold",
                         color=NAVY, transform=ax_next.transAxes)
            steps = (
                f"  \u2022  Schedule IC review for all {n_rev} REVIEW-classified manager(s).\n"
                "  \u2022  Place WATCH-classified managers on heightened monitoring.\n"
                "  \u2022  Confirm REGIME alarms are attributable to market stress.\n"
                "  \u2022  Verify short-history (*) fund data against official NAV sources.\n"
                "  \u2022  Re-run this report monthly after each month-end NAV update."
            )
            ax_next.text(0.02, 0.74, steps, va="top", fontsize=7.8,
                         color="#333333", transform=ax_next.transAxes, linespacing=1.55)

        page_footer(fig, pnum[0], total_pages)
        pnum[0] += 1
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_html(all_results, all_post_alarm, output_path="cusum_report.html"):
    def tbl_row(cells, header=False, colors=None):
        tag   = "th" if header else "td"
        cols  = colors or [""] * len(cells)
        cells_html = "".join(
            f'<{tag} style="color:{c};font-weight:{"bold" if c else "normal"}">'
            f'{v}</{tag}>'
            for v, c in zip(cells, cols))
        return f"<tr>{cells_html}</tr>"

    summary_rows = tbl_row(
        ["Fund", "Bench", "Months", "Ann TE", "IR", "Cum Excess",
         "#Alarms", "Alarm?", "First Alarm"],
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
             _fmt(r24["excess_cum"] if r24 is not None else float("nan")),
             _fmt(r36["excess_cum"] if r36 is not None else float("nan")),
             sig_txt],
            colors=["", "", "", "", ec(r12["excess_cum"]),
                    ec(r24["excess_cum"] if r24 is not None else float("nan")),
                    ec(r36["excess_cum"] if r36 is not None else float("nan")),
                    sig_col])

    css = """
    body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;
         color:#333;background:#f9f9f9;}
    h1{color:#1E2761;border-bottom:3px solid #D4AC0D;padding-bottom:8px;}
    h2{color:#1E2761;margin-top:40px;}
    table{border-collapse:collapse;width:100%;margin-bottom:30px;
          background:white;box-shadow:0 1px 4px rgba(0,0,0,.1);}
    th{background:#1E2761;color:white;padding:8px 10px;text-align:left;font-size:13px;}
    td{padding:7px 10px;font-size:12px;border-bottom:1px solid #eee;}
    tr:nth-child(even) td{background:#f4f6f7;}
    .footer{color:#999;font-size:11px;margin-top:40px;text-align:center;}
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>CUSUM Active-Manager Monitor</title>
<style>{css}</style></head>
<body>
<h1>CUSUM Active-Manager Monitor</h1>
<p><em>Generated: {datetime.today().strftime('%d %B %Y')}</em></p>
<h2>Summary Results</h2><table>{summary_rows}</table>
<h2>Post-Alarm Analysis</h2><table>{pa_rows}</table>
<p class="footer">Reference: Philips, T., Yashchin, E. &amp; Stein, D. (2003).
<em>Using Statistical Process Control to Monitor Active Managers.</em>
Journal of Portfolio Management, 30(1), 86–94.</p>
</body></html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

class MonitorResult:
    def __init__(self, monitor, history_df):
        for k, v in vars(monitor).items():
            setattr(self, k, v)
        self.history_df = history_df


CALIB_MONTHS = 24


def collect_data():
    all_rets, all_monitors, all_results, all_post_alarm = [], [], [], []
    print("Fetching data from Yahoo Finance...")
    for ticker, (fund, bench, name) in PRESET_PAIRS.items():
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            rets = load_from_yfinance(fund, bench)

            calib    = rets.iloc[:CALIB_MONTHS]
            excess_s = calib["portfolio"] - calib["benchmark"]
            init_te  = max(excess_s.std() * (12 ** 0.5), 0.01)

            ac      = _asset_class(fund, bench)
            preset  = AC_TO_PRESET.get(ac, "equity")   # default to equity if unmapped
            monitor = CUSUMMonitor(initial_te=init_te, preset=preset,
                                   calibration_months=CALIB_MONTHS)
            history_df = monitor.run(rets)

            excess_log = np.log((1 + rets["portfolio"]) / (1 + rets["benchmark"]))
            ir  = (excess_log.mean() * 12) / (excess_log.std() * (12 ** 0.5)) \
                  if excess_log.std() > 0 else 0.0
            te  = excess_log.std() * (12 ** 0.5)
            cum = (1 + rets["portfolio"]).prod() / (1 + rets["benchmark"]).prod() - 1

            ir_pt, ir_lo, ir_hi = bootstrap_ir_ci(rets)
            extra = compute_extra_stats(rets)

            alarm_date = (rets.index[monitor.alarm_t - 1].strftime("%b %Y")
                          if monitor.alarm else "-")
            regime  = alarm_regime(alarm_date)
            pa_df   = post_alarm_returns(rets, monitor)
            sig_q   = signal_quality(pa_df, horizon=12)

            mon_months = len(rets) - CALIB_MONTHS
            short_hist = mon_months < MIN_MONITORING_MONTHS

            all_rets.append(rets)
            mr = MonitorResult(monitor, history_df)
            all_monitors.append(mr)
            all_results.append({
                "fund":       fund,  "bench":  bench,  "name": name,
                "months":     len(rets),
                "mon_months": mon_months,
                "short_hist": short_hist,
                "ir":         ir,   "ir_lo":  ir_lo,  "ir_hi": ir_hi,
                "te":         te,
                "cum_excess": cum,
                "hit_rate":   extra["hit_rate"],
                "t_stat":     extra["t_stat"],
                "up_cap":     extra["up_cap"],
                "dn_cap":     extra["dn_cap"],
                "n_alarms":   len(monitor.all_alarms),
                "alarm":      monitor.alarm,
                "alarm_date": alarm_date,
                "regime":     regime,
                "sig_quality":sig_q,
            })
            all_post_alarm.append(pa_df)
            tag = f"[{regime}] " if regime else ""
            print(f"OK  {tag}({mon_months}m live{'  SHORT' if short_hist else ''})")
        except Exception as e:
            print(f"ERROR: {e}")

    return all_rets, all_monitors, all_results, all_post_alarm


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import os
    parser = argparse.ArgumentParser(description="CUSUM Report Generator")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: output/cusum_report.pdf)")
    parser.add_argument("--html", action="store_true",
                        help="Also generate HTML export")
    args = parser.parse_args()

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path  = args.output or os.path.join(out_dir, "cusum_report.pdf")
    html_path = os.path.join(os.path.dirname(pdf_path), "cusum_report.html")

    all_rets, all_monitors, all_results, all_post_alarm = collect_data()

    total_pages = compute_total_pages(all_results)
    pnum        = [1]

    print(f"\nBuilding PDF -> {pdf_path}  ({total_pages} pages)")
    with PdfPages(pdf_path) as pdf:
        page_cover             (pdf, pnum, total_pages)
        page_executive_summary (pdf, all_results, all_post_alarm, pnum, total_pages)
        page_methodology       (pdf, pnum, total_pages)
        page_assumptions       (pdf, pnum, total_pages)
        page_fund_universe     (pdf, pnum, total_pages)
        page_summary           (pdf, all_results, all_post_alarm, pnum, total_pages)
        page_post_alarm        (pdf, all_results, all_post_alarm, pnum, total_pages)
        page_waterfall         (pdf, all_results, pnum, total_pages)
        page_fund_plots        (pdf, all_rets, all_monitors, all_results, pnum, total_pages)
        page_conclusion        (pdf, all_results, all_post_alarm, pnum, total_pages)

    print(f"  PDF saved: {pdf_path}")
    if args.html:
        export_html(all_results, all_post_alarm, output_path=html_path)
    print("Done.")


if __name__ == "__main__":
    main()
