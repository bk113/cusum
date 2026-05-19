"""
CUSUM Active-Manager Monitor — All-in-One
==========================================

Single-file implementation of the Philips-Yashchin-Stein (2003) statistical
process control procedure for monitoring active portfolios. Includes the
algorithm, three real-data loaders (yfinance / CSV / Excel), an end-to-end
monitoring pipeline, and a self-test demo.

USAGE
-----
1) Pull a real fund from Yahoo Finance and run CUSUM:
       python cusum.py FCNTX
       python cusum.py FCNTX SPY --start-date 2010-01-01

2) Run on your own CSV (cols: date, fund_nav, benchmark_nav):
       python -c "from cusum import load_from_csv, monitor_fund; \\
                  r = load_from_csv('myfund.csv', 'fund_nav', 'benchmark_nav'); \\
                  monitor_fund(r, title='my fund')"

3) Run synthetic self-test (no internet, no data file):
       python cusum.py --demo

4) Override threshold and output path:
       python cusum.py FCNTX --threshold 23.59 --output fcntx_strict.png

DEPENDENCIES
------------
Required: numpy, pandas, matplotlib
Optional: yfinance  (only needed for `load_from_yfinance`)
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# PART 1 — Algorithm
# ===========================================================================

# Threshold table from Philips-Yashchin-Stein (2003), Table 2.
# Each row: (threshold h, ARL_good IR=0.5, ARL_flat IR=0, ARL_bad IR=-0.5)
# ARL = expected months until alarm. Higher h -> fewer false alarms, slower detection.
THRESHOLD_TABLE = pd.DataFrame(
    [
        (11.81, 24, 16, 11),
        (15.00, 36, 22, 15),
        (17.60, 48, 27, 18),
        (19.81, 60, 32, 21),  # default: ~1 false alarm / 5 yrs
        (21.79, 72, 37, 23),
        (23.59, 84, 41, 25),  # paper's preferred: ~1 false alarm / 7 yrs
    ],
    columns=["threshold", "arl_good_IR0.5", "arl_flat_IR0", "arl_bad_IR-0.5"],
)

# ---------------------------------------------------------------------------
# IR reference values — ANNUALISED scale, matching PYS (2003) Table 2.
#
# The CUSUM Lindley recursion accumulates:
#   ir_hat = (e_t / sigma_monthly) * sqrt(12)   (annualised IR, std ≈ sqrt(12) under H₀)
#
# mu parameters are on the same annualised scale. This is the scale on which
# Moustakides-optimality and the ARL table are derived, and allows direct
# citation of Table 2 (h ≈ 19.81 for ARL=60 monitoring months). Monthly-scale
# IR (std ≈ 1) would require h ≈ 5.7 — internally consistent but disconnected
# from the published table, losing report provenance.
#
# Note on regime_threshold (see CUSUMMonitor): with annualised ir_hat having
# std ≈ sqrt(12) ≈ 3.46 under H₀, the default of 14.0 sits at ~3.9σ (upper
# tail) and ~4.2σ (lower tail), giving a freeze rate of ~0.006% under normal
# regimes. This catches genuine crash months (Oct 2008, March 2020) where a
# fund with 4% TE losing 5% excess produces ir_hat ≈ −15, without freezing
# routine 1–2σ months. Do NOT lower this to ≤ 4.0 — that would freeze ~25%
# of months, suppress L accumulation, and pull calibrated h far below 19.81.
# ---------------------------------------------------------------------------
_SQRT12 = np.sqrt(12.0)

# Named ANNUALISED IR reference values — same units as Table 2.
_MU_EQUITY_GOOD    = 0.5    # annualised IR considered "good" for equity
_MU_EQUITY_BAD     = 0.0
_MU_FIXEDINC_GOOD  = 0.3    # annualised IR considered "good" for fixed income
_MU_FIXEDINC_BAD   = 0.0
_MU_BALANCED_GOOD  = 0.4    # annualised IR considered "good" for balanced
_MU_BALANCED_BAD   = 0.0

# Asset-class presets — mu values are ANNUALISED (matching Table 2 scale).
# Thresholds are loaded from thresholds_cache.json at import time and
# recomputed by calibrate_threshold() when the cache is absent.
# Fallback threshold values are placeholders; they are overwritten at import
# by _ensure_presets_calibrated().
ASSET_CLASS_PRESETS: dict = {
    "equity": {
        "mu_good":   _MU_EQUITY_GOOD,    # annualised IR: 0.50
        "mu_bad":    _MU_EQUITY_BAD,
        "threshold": 19.81,              # fallback; overwritten at import
    },
    "fixed_income": {
        "mu_good":   _MU_FIXEDINC_GOOD,  # annualised IR: 0.30
        "mu_bad":    _MU_FIXEDINC_BAD,
        "threshold": 19.81,
    },
    "balanced": {
        "mu_good":   _MU_BALANCED_GOOD,  # annualised IR: 0.40
        "mu_bad":    _MU_BALANCED_BAD,
        "threshold": 19.81,
    },
}


@dataclass
class CUSUMMonitor:
    """
    Sequential change-point detector for active-manager skill (lower CUSUM only).

    Detects skill loss via the Lindley recursion on monthly log-excess returns.
    After the first alarm, accumulation stops by default (rolling_restart=False,
    paper-faithful); set rolling_restart=True to reset L and continue.

    Parameters
    ----------
    initial_te          : annualised tracking-error seed (e.g. 0.04 = 4 %).
    threshold           : alarm trigger. See THRESHOLD_TABLE. Default 19.81.
    gamma               : EWMA decay for tracking-error estimate (0.9 per paper).
    mu_good             : annualised IR considered "good" (default 0.50 for equity,
                          matching Table 2). Must match the scale of
                          ir_hat = (e / sigma_monthly) * sqrt(12) (annualised, std ≈ sqrt(12) under H₀).
    mu_bad              : annualised IR considered "bad" (default 0.0).
    calibration_months  : months used ONLY to warm up sigma; L frozen at 0 during
                          this window so calibration data is not also test data.
    regime_threshold    : if |ir_hat| (annualised) exceeds this value, the month is
                          flagged as a regime outlier — L is frozen (not updated) and
                          the time index recorded in regime_outliers. Unlike
                          winsorisation, the raw IR is preserved and the month is
                          simply skipped, leaving no distortion in L.
                          Default 14.0. With annualised ir_hat (std ≈ sqrt(12) ≈ 3.46
                          under H₀), this sits at ~3.9σ on the upper tail and ~4.2σ
                          on the lower (asymmetric because mean ≈ +0.5 under H₀),
                          giving a freeze rate of ~0.006% under normal regimes.
                          Catches genuine crash months such as Oct 2008 / March 2020
                          where a fund with 4% TE losing 5% excess vs benchmark
                          produces ir_hat ≈ −15.
    rolling_restart     : if True, L resets to 0 after each alarm and accumulation
                          continues — all alarms are recorded in all_alarms.
                          If False (default, paper-faithful), L accumulation stops
                          after the first alarm; all_alarms has at most one entry.
    preset              : if given, must be a key in ASSET_CLASS_PRESETS ("equity",
                          "fixed_income", or "balanced"). Sets mu_good, mu_bad, and
                          threshold to MC-calibrated values. Any of these three can
                          be further overridden by passing them explicitly after the
                          preset is applied.

    Scale note
    ----------
    ir_hat (and ir_hat_raw) are on ANNUALISED scale (std ≈ sqrt(12) ≈ 3.46 under H₀),
    matching the scale on which PYS (2003) Table 2 thresholds and Moustakides-optimality
    are derived. This allows direct citation of Table 2 (h=19.81 ↔ ARL≈60 monitoring
    months). mu_good and mu_bad must be annualised values (e.g. 0.50, 0.30). The
    history dict stores ir_hat_annual = ir_hat for display — no further conversion needed.
    """
    initial_te:         float = 0.04
    threshold:          float = 19.81
    gamma:              float = 0.9
    mu_good:            float = _MU_EQUITY_GOOD   # annualised IR = 0.50 (equity default)
    mu_bad:             float = 0.0
    calibration_months: int   = 24    # L frozen during this window
    regime_threshold:   float = 14.0  # |ir_hat_annual| above this → freeze L (~3.9σ)
    rolling_restart:    bool  = False  # paper-faithful default: stop after first alarm
    preset:             Optional[str] = None  # key into ASSET_CLASS_PRESETS

    # ── lower CUSUM state (skill loss) ──────────────────────────────────────
    t:                int             = 0
    L:                float           = 0.0
    sigma_monthly_sq: float           = field(init=False)
    e_prev:           Optional[float] = None
    alarm:            bool            = False
    alarm_t:          Optional[int]   = None
    all_alarms:       List[int]       = field(default_factory=list)

    # ── regime-outlier tracking ──────────────────────────────────────────────
    regime_outliers:  List[int]       = field(default_factory=list)

    history:          list            = field(default_factory=list)

    def __post_init__(self):
        if self.preset is not None:
            if self.preset not in ASSET_CLASS_PRESETS:
                raise ValueError(
                    f"Unknown preset {self.preset!r}. "
                    f"Valid keys: {list(ASSET_CLASS_PRESETS)}"
                )
            p = ASSET_CLASS_PRESETS[self.preset]
            self.mu_good   = p["mu_good"]
            self.mu_bad    = p["mu_bad"]
            self.threshold = p["threshold"]
        self.sigma_monthly_sq = (self.initial_te ** 2) / 12.0

    def update(self, r_portfolio: float, r_benchmark: float) -> dict:
        """Process one month's return pair."""
        self.t += 1
        in_calibration = (self.t <= self.calibration_months)

        # (1) Log excess return
        e = np.log((1 + r_portfolio) / (1 + r_benchmark))

        # (2) EWMA tracking-error update (Von Neumann adjacent-difference estimator).
        #     Runs during calibration AND monitoring — sigma warms up from day 1.
        if self.e_prev is not None:
            innovation = 0.5 * (e - self.e_prev) ** 2
            self.sigma_monthly_sq = (
                self.gamma * self.sigma_monthly_sq + (1 - self.gamma) * innovation
            )
        sigma_monthly = np.sqrt(self.sigma_monthly_sq)

        # (3) Annualised IR estimate: ir_hat = (e / sigma_monthly) * sqrt(12).
        #     Annualised scale gives std(ir_hat) ≈ sqrt(12) under H₀, matching the
        #     scale on which PYS (2003) Table 2 thresholds and Moustakides-optimality
        #     are derived. mu_good/mu_bad are annualised; threshold h can be read
        #     directly from Table 2 (e.g. h=19.81 ↔ ARL≈60 monitoring months).
        #     No clipping — the raw value is preserved; regime outliers are handled
        #     by freezing L, not by capping ir_hat.
        ir_hat_raw     = (e / sigma_monthly) * _SQRT12 if sigma_monthly > 0 else 0.0
        ir_hat         = ir_hat_raw
        regime_outlier = abs(ir_hat_raw) > self.regime_threshold

        # (4) Lindley recursion — only AFTER calibration period.
        #     Regime-outlier months: L is frozen — skipped entirely so the crash
        #     month leaves no distortion in L.
        #     rolling_restart=False: once the first alarm fires, accumulation halts.
        if not in_calibration:
            if regime_outlier:
                self.regime_outliers.append(self.t)
            else:
                if not self.alarm or self.rolling_restart:
                    drift_down = 0.5 * (self.mu_good + self.mu_bad) - ir_hat
                    self.L = max(0.0, self.L + drift_down)

                    if self.L >= self.threshold:
                        self.all_alarms.append(self.t)
                        if not self.alarm:
                            self.alarm   = True
                            self.alarm_t = self.t
                        if self.rolling_restart:
                            self.L = 0.0   # reset and continue

        self.e_prev = e
        state = {
            "t":              self.t,
            "calibration":    in_calibration,
            "excess_return":  e,
            "te_annual":      np.sqrt(self.sigma_monthly_sq * 12),
            # ir_hat / ir_hat_raw: annualised scale (std ≈ sqrt(12) under H₀),
            # matching Table 2. ir_hat_annual is an alias kept for API compatibility.
            "ir_hat":         ir_hat,
            "ir_hat_raw":     ir_hat_raw,
            "ir_hat_annual":  ir_hat,   # already annualised — no conversion needed
            "regime_outlier": regime_outlier,
            "L":              self.L,
            "alarm":          self.alarm,
        }
        self.history.append(state)
        return state

    def run(self, returns: pd.DataFrame,
            port_col: str = "portfolio", bench_col: str = "benchmark") -> pd.DataFrame:
        """Run the monitor over a full return series."""
        for _, row in returns.iterrows():
            self.update(row[port_col], row[bench_col])
        return pd.DataFrame(self.history)

    def plot(self, dates: Optional[pd.Index] = None,
             title: str = "CUSUM monitor") -> plt.Figure:
        """Two-panel plot: cumulative excess return + CUSUM statistic."""
        df = pd.DataFrame(self.history)
        if df.empty:
            raise RuntimeError("no history — run the monitor first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        x = df["t"].values

        # Panel 1: cumulative log excess return
        ax1.plot(x, df["excess_return"].cumsum(), lw=1.5, color="#1E2761")
        ax1.axhline(0, color="grey", lw=0.6)
        ax1.set_ylabel("cumulative log excess return")
        ax1.set_title(title)
        ax1.grid(alpha=0.3)

        # Panel 2: CUSUM statistic (inverted so threshold is a ceiling)
        ax2.plot(x, df["L"], lw=1.5, color="#1E2761", label="L (skill loss)")
        ax2.axhline(self.threshold, color="#C0392B", ls="--",
                    label=f"threshold = {self.threshold:.2f}")
        ax2.invert_yaxis()
        ax2.set_ylabel("CUSUM statistic (inverted)")
        ax2.grid(alpha=0.3)

        for at in self.all_alarms:
            ax2.axvline(at, color="#C0392B", lw=0.9, alpha=0.6)
        if self.alarm_t:
            label_x = self.alarm_t + len(x) * 0.01
            ax2.text(label_x, self.threshold * 0.55,
                     f"ALARM @ t={self.alarm_t}",
                     color="#C0392B", va="center", fontsize=9, fontweight="bold")

        ax2.legend(loc="lower right", fontsize=8)

        if dates is not None and len(dates) >= len(x):
            for ax in (ax1, ax2):
                tick_locs = [t for t in ax.get_xticks() if 0 < int(t) <= len(dates)]
                ax.set_xticks(tick_locs)
                ax.set_xticklabels([dates[int(t) - 1].strftime("%Y") for t in tick_locs])
                ax.set_xlabel("")
        else:
            ax2.set_xlabel("month")

        plt.tight_layout()
        return fig


# ===========================================================================
# Monte Carlo threshold calibration
# ===========================================================================

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds_cache.json")


def calibrate_threshold(
    mu_good: float,
    mu_bad: float,
    initial_te: float = 0.04,
    target_arl: float = 84.0,
    n_paths: int = 10_000,
    max_months: int = 600,
    seed: int = 42,
    tol: float = 0.5,
    h_lo: float = 1.0,
    h_hi: float = 30.0,
) -> float:
    """
    Find threshold h such that the simulated mean time-to-first-alarm under H₀
    (continuous IR = mu_good, no skill change) equals target_arl months.

    Uses bisection on h. Runs the actual CUSUMMonitor — including its 24-month
    sigma warm-up window, regime-outlier freeze (regime_threshold=14.0), and
    annualised-scale Lindley recursion — not a simplified version.

    Under H₀, monthly log-excess returns are drawn iid from
      N(mu_good * initial_te / 12,  (initial_te / sqrt(12))²)
    which gives E[ir_hat_annual] = E[(e / sigma_m) * sqrt(12)] = mu_good.
    r_benchmark is set to zero; r_portfolio = e_t, valid for small monthly
    returns (|e| < 0.15 typically; log(1+x) ≈ x approximation error < 1 %).

    mu_good and mu_bad must be on MONTHLY scale, matching the convention
    used throughout the code. Use the _MU_* constants.

    Parameters
    ----------
    mu_good    : annualised IR under H₀ (e.g. _MU_EQUITY_GOOD = 0.50, matching Table 2).
    mu_bad     : annualised IR under H₁ (typically 0.0).
    initial_te : annualised TE seed for each CUSUMMonitor path. Default 0.04.
    target_arl : target average run length in months. Default 84.
                 IMPORTANT: alarm_t counts from month 1 including the
                 calibration_months window during which L is frozen at 0.
                 The paper's ARL=60 is measured from the START OF MONITORING
                 (post-calibration). To match: target_arl = 60 + calibration_months
                 = 60 + 24 = 84. Do NOT change this back to 60 — that would aim
                 for only 36 monitoring months (60 − 24), misaligning with Table 2.
    n_paths    : Monte Carlo paths per bisection step. Default 10 000.
                 Increase to 50 000 for production-grade precision at cost of ~5×
                 longer runtime; 10 000 gives threshold accuracy ±0.3 or better.
    max_months : max simulation length per path; censored paths count as max_months.
    seed       : RNG seed for reproducibility.
    tol        : bisection stops when |simulated ARL − target_arl| < tol.
    h_lo/h_hi  : search bracket for h. Default 1–30; must span the expected
                 annualised-scale threshold (~19–23 for equity at ARL=60 monitoring months).

    Returns
    -------
    Calibrated threshold h (float, rounded to 4 d.p.).
    """
    rng    = np.random.default_rng(seed)
    # Monthly log-excess mean and std under H₀ so that E[ir_hat_annual] = mu_good.
    # ir_hat_annual = (e / sigma_m) * sqrt(12)  where sigma_m = initial_te / sqrt(12)
    # => E[e] = mu_good * sigma_m / sqrt(12) = mu_good * initial_te / 12
    # std(e) = sigma_m = initial_te / sqrt(12)  (unchanged)
    e_mean = mu_good * initial_te / 12.0
    e_std  = initial_te / _SQRT12

    # Pre-generate all random draws — shared across bisection steps so that
    # each step gets the same noise realisations (variance reduction).
    draws = rng.normal(e_mean, e_std, size=(n_paths, max_months))

    def mean_ttf(h: float) -> float:
        total = 0.0
        for i in range(n_paths):
            m = CUSUMMonitor(
                initial_te=initial_te,
                threshold=h,
                mu_good=mu_good,
                mu_bad=mu_bad,
                rolling_restart=False,
            )
            alarmed = False
            for j in range(max_months):
                m.update(draws[i, j], 0.0)
                if m.alarm:
                    total  += m.alarm_t
                    alarmed = True
                    break
            if not alarmed:
                total += max_months
        return total / n_paths

    lo, hi = h_lo, h_hi
    for _ in range(25):
        mid     = (lo + hi) / 2.0
        arl_mid = mean_ttf(mid)
        if abs(arl_mid - target_arl) < tol:
            return round(mid, 4)
        if arl_mid < target_arl:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def _ensure_presets_calibrated() -> None:
    """
    Load calibrated thresholds from thresholds_cache.json into ASSET_CLASS_PRESETS.
    If the cache is absent, run calibrate_threshold() for each asset class (one-time,
    ~1–3 min), save the results, then update the in-memory presets.
    """
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            cache = json.load(f)
        for ac, data in cache.items():
            if ac in ASSET_CLASS_PRESETS and "threshold" in data:
                ASSET_CLASS_PRESETS[ac]["threshold"] = data["threshold"]
        return

    # Cache absent — calibrate now (runs once, then cached on disk).
    print("thresholds_cache.json not found - calibrating thresholds "
          "(this runs once and is then cached)...")
    cache = {}
    for ac, spec in ASSET_CLASS_PRESETS.items():
        h = calibrate_threshold(mu_good=spec["mu_good"], mu_bad=spec["mu_bad"])
        ASSET_CLASS_PRESETS[ac]["threshold"] = h
        cache[ac] = {
            "threshold": h,
            "mu_good":   spec["mu_good"],
            "mu_bad":    spec["mu_bad"],
            "target_arl": 84.0,   # 60 monitoring months + 24 calibration months
        }
        print(f"  {ac}: h = {h}")

    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  saved -> {_CACHE_PATH}")


_ensure_presets_calibrated()


# ===========================================================================
# PART 2 — Statistical utilities
# ===========================================================================

# Regime windows used to flag alarms that coincide with market crises
GFC_START   = pd.Timestamp("2007-10-01")
GFC_END     = pd.Timestamp("2009-03-31")
COVID_START = pd.Timestamp("2020-02-01")
COVID_END   = pd.Timestamp("2020-04-30")

MIN_MONITORING_MONTHS = 60   # minimum months of live CUSUM after calibration


def sanity_check_returns(
    returns: pd.DataFrame,
    fund_name: str = "",
    port_col: str = "portfolio",
    bench_col: str = "benchmark",
    regime_threshold: float = 14.0,
) -> List[str]:
    """
    Pre-flight data-quality check on a monthly return series.

    Returns a list of warning strings (empty list = clean data). Designed to
    be called for every fund in monitor_book() before the CUSUM is run so that
    data problems are surfaced rather than silently distorting the statistic.

    Checks
    ------
    1. Annualised tracking error > 25 %  (suggests data error or mis-matched series).
    2. Any single-month |return| > 50 %  (split / dividend artifact).
    3. More than 3 missing months in the date index (gaps in the time series).
    4. More than 5 months where the single-observation monthly IR estimate
       would exceed regime_threshold (persistent extreme readings → TE estimate
       is probably wrong or the series has outlier contamination).
    5. Total observations < 60 months (24 calibration + 36 minimum monitoring).
    6. Runs of ≥ 3 consecutive months with identical fund return (stale NAV).
    7. Months where fund_return == benchmark_return to 6 decimal places (stale data).
    8. Lag-1 autocorrelation of log excess returns |ρ₁| > 0.3 (CUSUM independence
       assumption violated — may indicate return-smoothing or stale pricing).

    Parameters
    ----------
    returns          : DataFrame with portfolio and benchmark columns.
    fund_name        : label used in warning messages.
    regime_threshold : same value used in CUSUMMonitor (default 14.0 annualised,
                       ~3.9–4.2σ with annualised ir_hat std ≈ sqrt(12) ≈ 3.46 under H₀).
    """
    warnings: List[str] = []
    prefix = f"[{fund_name}] " if fund_name else ""

    excess_arith = returns[port_col] - returns[bench_col]
    te_annual    = excess_arith.std() * _SQRT12

    # (1) TE sanity
    if te_annual > 0.25:
        warnings.append(
            f"{prefix}annualised TE = {te_annual:.1%} exceeds 25 % — "
            "possible data error or benchmark mismatch"
        )

    # (2) Single-month extreme return
    for col in (port_col, bench_col):
        big = returns[col].abs() > 0.50
        if big.any():
            dates = returns.index[big].strftime("%Y-%m").tolist()
            warnings.append(
                f"{prefix}column '{col}' has |return| > 50 % in: {dates} — "
                "check for NAV restatement or split"
            )

    # (3) Missing months in the date index
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        expected_months = pd.date_range(
            returns.index[0], returns.index[-1], freq="ME"
        )
        missing = expected_months.difference(returns.index)
        if len(missing) > 3:
            warnings.append(
                f"{prefix}{len(missing)} missing months in return series "
                f"(first few: {missing[:3].strftime('%Y-%m').tolist()}) — "
                "check for download gaps"
            )

    # (4) Persistent regime-threshold breaches
    sigma_m    = te_annual / _SQRT12 if te_annual > 0 else 1e-6
    excess_log = np.log((1 + returns[port_col]) / (1 + returns[bench_col]))
    ir_hat_series = (excess_log / sigma_m) * _SQRT12   # annualised scale, std ≈ sqrt(12) under H₀
    n_outliers = int((ir_hat_series.abs() > regime_threshold).sum())
    if n_outliers > 5:
        warnings.append(
            f"{prefix}{n_outliers} months exceed regime_threshold={regime_threshold} "
            f"— TE estimate may be too low or series has outlier contamination"
        )

    # (5) Insufficient history
    if len(returns) < 60:
        warnings.append(
            f"{prefix}only {len(returns)} months of history — minimum 60 required "
            "(24 calibration + 36 monitoring)"
        )

    # (6) Stale NAV: runs of ≥ 3 consecutive identical fund returns
    fund_vals  = returns[port_col].values
    fund_dates = returns.index
    stale_starts: List[str] = []
    i = 0
    while i < len(fund_vals):
        j = i + 1
        while j < len(fund_vals) and fund_vals[j] == fund_vals[i]:
            j += 1
        if j - i >= 3:
            stale_starts.append(fund_dates[i].strftime("%Y-%m"))
        i = j
    if stale_starts:
        warnings.append(
            f"{prefix}stale NAV: {len(stale_starts)} run(s) of ≥3 consecutive identical "
            f"fund returns — first run starts {stale_starts[0]}"
        )

    # (7) Fund return == benchmark return to 6 decimal places (identical data)
    exact_match = (returns[port_col].round(6) == returns[bench_col].round(6))
    n_exact = int(exact_match.sum())
    if n_exact > 0:
        first_exact = returns.index[exact_match][0].strftime("%Y-%m")
        warnings.append(
            f"{prefix}{n_exact} month(s) with fund_return == benchmark_return to 6dp "
            f"(first: {first_exact}) — possible stale or copy-paste data"
        )

    # (8) Excess return lag-1 autocorrelation (CUSUM independence assumption)
    if len(excess_log) > 12:
        rho1 = float(excess_log.autocorr(lag=1))
        if abs(rho1) > 0.3:
            warnings.append(
                f"{prefix}lag-1 autocorrelation of log excess = {rho1:+.3f} (|ρ₁| > 0.3) — "
                "CUSUM independence assumption may be violated (return-smoothing / stale pricing?)"
            )

    return warnings


def bootstrap_ir_ci(
    returns: pd.DataFrame,
    n_boot: int = 1000,
    alpha: float = 0.05,
    port_col: str = "portfolio",
    bench_col: str = "benchmark",
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Circular block bootstrap CI on the annualised Information Ratio.

    Uses LOG excess returns — same quantity the CUSUM statistic accumulates —
    so the CI is consistent with what is being monitored.

    Block length ~ n^(1/3) (Politis-White 2004) preserves autocorrelation.

    Returns
    -------
    (ir_point, ci_lower, ci_upper)
    """
    # Log excess return — consistent with CUSUM accumulation
    excess    = np.log((1 + returns[port_col]) / (1 + returns[bench_col]))
    n         = len(excess)
    block_len = max(3, int(round(n ** (1 / 3))))
    ir_point  = (excess.mean() * 12) / (excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0

    rng      = np.random.default_rng(seed)
    boot_irs = []
    for _ in range(n_boot):
        starts  = rng.integers(0, n, size=int(np.ceil(n / block_len)))
        indices = np.concatenate([np.arange(s, s + block_len) % n for s in starts])[:n]
        sample  = excess.values[indices]
        std     = sample.std()
        if std > 0:
            boot_irs.append((sample.mean() * 12) / (std * np.sqrt(12)))

    lo = float(np.percentile(boot_irs, 100 * alpha / 2))
    hi = float(np.percentile(boot_irs, 100 * (1 - alpha / 2)))
    return ir_point, lo, hi


def compute_extra_stats(
    returns: pd.DataFrame,
    port_col: str = "portfolio",
    bench_col: str = "benchmark",
) -> dict:
    """
    Compute supplementary per-fund statistics a CIO needs beyond IR and TE.

    All four metrics use log excess returns — consistent with the CUSUM
    accumulation and the bootstrap CI — so hit rate, t-stat, and capture
    ratios are all derived from the same excess series.

    Returns
    -------
    hit_rate : fraction of months log excess return > 0
    t_stat   : annualised t-statistic of the IR  (IR × sqrt(n/12))
    up_cap   : up-capture ratio  (fund avg log-excess / |bench avg| in up months)
    dn_cap   : down-capture ratio (same, in down months)
    """
    excess_log = np.log((1 + returns[port_col]) / (1 + returns[bench_col]))
    n          = len(returns)

    # hit rate: fraction of months with positive log excess
    hit_rate = float((excess_log > 0).mean())

    # t-stat derived from the same log-excess IR
    ir     = (excess_log.mean() * 12) / (excess_log.std() * np.sqrt(12)) if excess_log.std() > 0 else 0.0
    t_stat = ir * np.sqrt(n / 12)

    # Capture ratios use ARITHMETIC excess and benchmark returns — industry
    # convention (GIPS / Morningstar definition). Deliberately inconsistent with
    # hit_rate and t_stat (which use log excess) to match how practitioners
    # publish up/down capture; noted here so the difference is explicit.
    excess_arith = returns[port_col] - returns[bench_col]
    bench_arith  = returns[bench_col]
    bench_up   = bench_arith > 0
    bench_down = ~bench_up

    def _capture(mask):
        if mask.sum() < 6:
            return np.nan
        b_fund  = excess_arith[mask].mean() + bench_arith[mask].mean()  # avg fund return
        b_bench = bench_arith[mask].mean()
        return b_fund / abs(b_bench) if b_bench != 0 else np.nan

    return {
        "hit_rate": hit_rate,
        "t_stat":   t_stat,
        "up_cap":   _capture(bench_up),
        "dn_cap":   _capture(bench_down),
    }


def alarm_regime(alarm_date_str: str) -> str:
    """Return 'GFC', 'COVID', or '' depending on whether the first alarm fell in a crisis."""
    if not alarm_date_str or alarm_date_str == "-":
        return ""
    try:
        d = pd.Timestamp(alarm_date_str)
        if GFC_START <= d <= GFC_END:
            return "GFC"
        if COVID_START <= d <= COVID_END:
            return "COVID"
    except Exception:
        pass
    return ""


def load_vix_history(start: str = "2005-01-01",
                     end: Optional[str] = None) -> pd.Series:
    """
    Pull ^VIX from yfinance, resample to monthly mean closes.

    Returns a pd.Series indexed by month-end dates with the mean daily VIX
    close for each calendar month. Monthly mean is used (rather than month-end
    close) so that a single volatile day at month-end does not unduly inflate
    the regime label.

    Requires yfinance. VIX data is available from 1990 onwards.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    raw = yf.download("^VIX", start=start, end=end,
                      progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data for ^VIX")

    # yfinance may return a (Price, Ticker) MultiIndex DataFrame for a single
    # ticker. Squeeze to a Series regardless of schema.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].squeeze()
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        close = raw.iloc[:, 0]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # final safety squeeze

    return close.resample("ME").mean().rename("vix_monthly_mean")


def tag_alarm_regime(
    alarm_dates: List[str],
    vix: pd.Series,
) -> dict:
    """
    Classify each alarm date by the VIX level in that month.

    IMPORTANT: vix must come from load_vix_history(), which resamples to the
    monthly MEAN before returning. Joining on a single month-end close is a
    silent-failure trap — a spike on the last trading day inflates the label.

    Categories:
      "high_vol" — VIX >= 30  (stress / tail-risk episode)
      "med_vol"  — 20 <= VIX < 30
      "low_vol"  — VIX < 20

    Parameters
    ----------
    alarm_dates : list of "YYYY-MM" strings (from monitor.alarm_date or similar).
    vix         : monthly VIX series from load_vix_history() — index is month-end,
                  values are the MEAN daily close for each calendar month.

    Returns
    -------
    dict mapping alarm_date_str → regime label (or "unknown" if VIX not available
    for that month).
    """
    labels = {}
    for date_str in alarm_dates:
        if not date_str or date_str == "-":
            continue
        try:
            ts  = pd.Timestamp(date_str)
            # Match to month-end in VIX series (within same calendar month).
            # Uses period matching so "2008-10" matches the Oct month-end entry.
            key = vix.index[vix.index.to_period("M") == ts.to_period("M")]
            if len(key) == 0:
                labels[date_str] = "unknown"
                continue
            v = float(vix.loc[key[0]])
            if v >= 30:
                labels[date_str] = "high_vol"
            elif v >= 20:
                labels[date_str] = "med_vol"
            else:
                labels[date_str] = "low_vol"
        except Exception:
            labels[date_str] = "unknown"
    return labels


def signal_quality(
    pa_df: pd.DataFrame,
    horizon: int = 12,
    regime: Optional[str] = None,
) -> Optional[float]:
    """
    Fraction of alarms where the fund underperformed its benchmark in the
    `horizon` months following the alarm.

    A value of 0.5 = coin flip; higher = more actionable signals.

    Parameters
    ----------
    pa_df   : DataFrame from post_alarm_returns(), optionally with a "regime"
              column added by tag_alarm_regime().
    horizon : forward look-through window in months. Default 12.
    regime  : if provided, restrict to alarms whose "regime" column matches
              this label ("high_vol", "med_vol", or "low_vol"). Allows hit
              rates to be stratified by VIX regime.
    """
    if pa_df is None or pa_df.empty:
        return None
    rows = pa_df[pa_df["horizon"] == horizon].dropna(subset=["correct_alarm"])
    if regime is not None and "regime" in rows.columns:
        rows = rows[rows["regime"] == regime]
    return float(rows["correct_alarm"].mean()) if len(rows) > 0 else None


def signal_quality_table(
    pa_df: pd.DataFrame,
    horizons: Tuple[int, ...] = (12, 24),
) -> pd.DataFrame:
    """
    Stratified signal-quality table: Regime | N alarms | Signal quality +12M | +24M.

    Rows: one per VIX regime (high_vol / med_vol / low_vol / all), plus an
    "all" row. Columns: regime, n_alarms, and one sq_+{h}m column per horizon.

    Parameters
    ----------
    pa_df    : DataFrame from post_alarm_returns() with a "regime" column
               added by tag_alarm_regime() and merged in.
    horizons : forward horizons to compute signal quality for. Default (12, 24).

    Returns
    -------
    pd.DataFrame with columns: regime, n_alarms, sq_+12m, sq_+24m (or similar).
    """
    if pa_df is None or pa_df.empty:
        return pd.DataFrame()

    regimes  = ["high_vol", "med_vol", "low_vol", "all"]
    sq_cols  = [f"sq_+{h}m" for h in horizons]
    rows_out = []
    for reg in regimes:
        if reg == "all":
            subset = pa_df
        elif "regime" not in pa_df.columns:
            continue
        else:
            subset = pa_df[pa_df["regime"] == reg]

        # Count distinct alarms (each alarm appears once per horizon)
        n_alarms = subset["alarm_t"].nunique() if "alarm_t" in subset.columns else 0
        row = {"regime": reg, "n_alarms": n_alarms}
        for h, col in zip(horizons, sq_cols):
            row[col] = signal_quality(subset, horizon=h)
        rows_out.append(row)

    return pd.DataFrame(rows_out, columns=["regime", "n_alarms"] + sq_cols)


def post_alarm_returns(
    returns: pd.DataFrame,
    monitor: CUSUMMonitor,
    horizons: Tuple[int, ...] = (12, 24, 36),
    port_col: str = "portfolio",
    bench_col: str = "benchmark",
) -> pd.DataFrame:
    """
    For every skill-loss alarm in monitor.all_alarms, compute cumulative fund
    and benchmark returns over each forward horizon (months).

    Returns a DataFrame with columns:
      alarm_t, alarm_date, horizon, fund_cum, bench_cum, excess_cum, correct_alarm
    where correct_alarm = True when the fund underperformed after the alarm.
    """
    rows = []
    for alarm_t in monitor.all_alarms:
        alarm_date = (returns.index[alarm_t - 1].strftime("%Y-%m")
                      if alarm_t <= len(returns) else "")
        for h in horizons:
            start_idx = alarm_t
            end_idx   = start_idx + h
            if end_idx > len(returns):
                fund_cum = bench_cum = excess_cum = np.nan
                correct  = None
            else:
                window     = returns.iloc[start_idx:end_idx]
                fund_cum   = (1 + window[port_col]).prod() - 1
                bench_cum  = (1 + window[bench_col]).prod() - 1
                # Geometric excess — consistent with CUSUM's log accumulation.
                # (1+r_p).prod() / (1+r_b).prod() - 1, not fund_cum - bench_cum.
                excess_cum = (1 + fund_cum) / (1 + bench_cum) - 1
                correct    = bool(excess_cum < 0)
            rows.append({
                "alarm_t":       alarm_t,
                "alarm_date":    alarm_date,
                "horizon":       h,
                "fund_cum":      fund_cum,
                "bench_cum":     bench_cum,
                "excess_cum":    excess_cum,
                "correct_alarm": correct,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# PART 3 — Real-data loaders
# ===========================================================================

PRESET_PAIRS = {
    # ── Original 8 ──────────────────────────────────────────────────────────
    "FCNTX": ("FCNTX", "IWF",   "Fidelity Contrafund vs Russell 1000 Growth"),
    "DODGX": ("DODGX", "IWD",   "Dodge & Cox Stock vs Russell 1000 Value"),
    "AGTHX": ("AGTHX", "SPY",   "AmFunds Growth Fund of America vs S&P 500"),
    "OTCFX": ("OTCFX", "IWM",   "T. Rowe Price Small-Cap Stock vs Russell 2000"),
    "OAKIX": ("OAKIX", "EFA",   "Oakmark International vs MSCI EAFE"),
    "TEDMX": ("TEDMX", "EEM",   "Templeton Developing Markets vs MSCI EM"),
    "VWELX": ("VWELX", "AOR",   "Vanguard Wellington vs iShares Core Growth Allocation (AOR)"),
    "PTTRX": ("PTTRX", "AGG",   "PIMCO Total Return vs US Agg Bond"),
    # ── Extended universe ───────────────────────────────────────────────────
    "PRWCX": ("PRWCX", "AOR",   "T. Rowe Price Capital Appreciation vs iShares Core Growth Allocation (AOR)"),
    "FMAGX": ("FMAGX", "SPY",   "Fidelity Magellan vs S&P 500"),
    "SEQUX": ("SEQUX", "SPY",   "Sequoia Fund vs S&P 500"),
    "FLPSX": ("FLPSX", "IWM",   "Fidelity Low-Priced Stock vs Russell 2000"),
    "DODWX": ("DODWX", "EFA",   "Dodge & Cox International vs MSCI EAFE"),
    "LSBRX": ("LSBRX", "AGG",   "Loomis Sayles Bond vs US Agg Bond"),
    # ── Emerging Market Equity (benchmark: EEM) ─────────────────────────────
    "NWFFX": ("NWFFX", "EEM", "New World Fund vs MSCI EM"),
    "PRMSX": ("PRMSX", "EEM", "T. Rowe Price EM Stock vs MSCI EM"),
    "JEMWX": ("JEMWX", "EEM", "JPMorgan EM Equity vs MSCI EM"),
    "ODMAX": ("ODMAX", "EEM", "Invesco Developing Markets vs MSCI EM"),
    "PEIFX": ("PEIFX", "EEM", "PGIM Jennison EM Equity vs MSCI EM"),
    "LZOEX": ("LZOEX", "EEM", "Lazard EM Equity vs MSCI EM"),
    "HLMEX": ("HLMEX", "EEM", "Harding Loevner EM Equity vs MSCI EM"),
    "MEMCX": ("MEMCX", "EEM", "Matthews EM vs MSCI EM"),
    "VEIEX": ("VEIEX", "EEM", "Vanguard EM Stock Index vs MSCI EM"),
    "VEMAX": ("VEMAX", "EEM", "Vanguard EM Stock Index Admiral vs MSCI EM"),
    "NEWFX": ("NEWFX", "EEM", "American Funds New World vs MSCI EM"),
    "HAEMX": ("HAEMX", "EEM", "Harbor EM vs MSCI EM"),
    "TEMFX": ("TEMFX", "EEM", "Templeton EM vs MSCI EM"),
    "SFENX": ("SFENX", "EEM", "Seafarer EM Growth & Income vs MSCI EM"),
    "REMIX": ("REMIX",  "EEM", "Royce EM vs MSCI EM"),
    "ABEMX": ("ABEMX", "EEM", "AB EM Growth vs MSCI EM"),
    "GERIX": ("GERIX",  "EEM", "Goldman Sachs EM Equity vs MSCI EM"),
    "HLEMX": ("HLEMX", "EEM", "Harding Loevner EM Institutional vs MSCI EM"),
    "BGEFX": ("BGEFX", "EEM", "Baron EM vs MSCI EM"),
    "WIEMX": ("WIEMX", "EEM", "William Blair EM Growth vs MSCI EM"),
    "WAEMX": ("WAEMX", "EEM", "Western Asset EM vs MSCI EM"),
    "MIEMX": ("MIEMX", "EEM", "MFS EM Equity vs MSCI EM"),
    "EEMAX": ("EEMAX", "EEM", "Eaton Vance EM vs MSCI EM"),
    "FSEAX": ("FSEAX", "EEM", "Fidelity EM vs MSCI EM"),
    "MADCX": ("MADCX", "EEM", "Madison EM vs MSCI EM"),
    "PDEZX": ("PDEZX", "EEM", "PGIM EM vs MSCI EM"),
    # ── EM Hard Currency Bond (benchmark: EMB) ──────────────────────────────
    "PEBIX": ("PEBIX", "EMB", "PIMCO EM Bond vs JPM EM Bond (Hard CCY)"),
    "EBNDX": ("EBNDX", "EMB", "BlackRock EM Bond vs JPM EM Bond (Hard CCY)"),
    "PREMX": ("PREMX", "EMB", "T. Rowe Price EM Bond vs JPM EM Bond (Hard CCY)"),
    "TGBAX": ("TGBAX", "EMB", "Templeton Global Bond vs JPM EM Bond (Hard CCY)"),
    "IEMIX": ("IEMIX", "EMB", "Invesco EM Sovereign Debt vs JPM EM Bond (Hard CCY)"),
    "DBLEX": ("DBLEX", "EMB", "DFA EM Fixed Income vs JPM EM Bond (Hard CCY)"),
    "VWOB":  ("VWOB",  "EMB", "Vanguard EM Govt Bond ETF vs JPM EM Bond (Hard CCY)"),
    "MSEDX": ("MSEDX", "EMB", "MFS EM Debt vs JPM EM Bond (Hard CCY)"),
    # ── EM Local Currency Bond (benchmark: EMLC) ────────────────────────────
    "PFMIX": ("PFMIX", "EMLC", "PIMCO Foreign Bond vs VanEck EM Local CCY Bond"),
    "PRLAX": ("PRLAX", "EMLC", "T. Rowe Price EM Local CCY Bond vs VanEck EM Local CCY Bond"),
    # ── Global Equity (benchmark: ACWI) ─────────────────────────────────────
    "CWGIX": ("CWGIX", "ACWI", "Capital World Growth & Income vs MSCI ACWI"),
    "TEDIX": ("TEDIX", "ACWI", "Templeton Global Equity vs MSCI ACWI"),
    "VHGEX": ("VHGEX", "ACWI", "Vanguard Global Equity vs MSCI ACWI"),
    "OPGIX": ("OPGIX", "ACWI", "Invesco Oppenheimer Global vs MSCI ACWI"),
    "GPROX": ("GPROX", "ACWI", "Goldman Sachs Global Equity vs MSCI ACWI"),
    "SGIGX": ("SGIGX", "ACWI", "Schroders Global vs MSCI ACWI"),
    "MGGIX": ("MGGIX", "ACWI", "MFS Global Equity vs MSCI ACWI"),
    # ── Global Equity (benchmark: URTH — MSCI World) ────────────────────────
    "AIVSX": ("AIVSX", "URTH", "American Investors vs MSCI World"),
    "MGIAX": ("MGIAX", "URTH", "Morgan Stanley Institutional Global vs MSCI World"),
    "JGVAX": ("JGVAX", "URTH", "JPMorgan Global Focus vs MSCI World"),
    "HAINX": ("HAINX", "URTH", "Harbor International vs MSCI World"),
    # ── AIAIM Global Equity (benchmark: ACWI) — Singapore-listed ────────────
    "0P0001I6KI.SI": ("0P0001I6KI.SI", "ACWI", "AIAIM Global Equity Fund A vs MSCI ACWI"),
    "0P00008T6G.SI": ("0P00008T6G.SI", "ACWI", "AIAIM Global Equity Fund C vs MSCI ACWI"),
    "0P00008T6D.SI": ("0P00008T6D.SI", "ACWI", "AIAIM Global Equity Fund D vs MSCI ACWI"),
    "0P0001RMIN.SI": ("0P0001RMIN.SI", "ACWI", "AIAIM Global Equity Fund E vs MSCI ACWI"),
    # ── AIAIM EM Bond (benchmark: EMB) — Singapore-listed ───────────────────
    "0P0001KL95.SI": ("0P0001KL95.SI", "EMB", "AIAIM EM Bond Fund A vs JPM EM Bond"),
    "0P00008T6N.SI": ("0P00008T6N.SI", "EMB", "AIAIM EM Bond Fund B vs JPM EM Bond"),
}

# Maps each PRESET_PAIRS key to an asset class for MC-calibrated threshold selection.
TICKER_ASSET_CLASS: dict = {
    # ── US Equity ────────────────────────────────────────────────────────────
    "FCNTX": "equity",
    "DODGX": "equity",
    "AGTHX": "equity",
    "OTCFX": "equity",
    "FMAGX": "equity",
    "SEQUX": "equity",
    "FLPSX": "equity",
    # ── International Equity ─────────────────────────────────────────────────
    "OAKIX": "equity",
    "TEDMX": "equity",
    "DODWX": "equity",
    # ── Balanced ─────────────────────────────────────────────────────────────
    "VWELX": "balanced",
    "PRWCX": "balanced",
    # ── US Fixed Income ───────────────────────────────────────────────────────
    "PTTRX": "fixed_income",
    "LSBRX": "fixed_income",
    # ── EM Equity ─────────────────────────────────────────────────────────────
    "NWFFX": "equity",
    "PRMSX": "equity",
    "JEMWX": "equity",
    "ODMAX": "equity",
    "PEIFX": "equity",
    "LZOEX": "equity",
    "HLMEX": "equity",
    "MEMCX": "equity",
    "VEIEX": "equity",
    "VEMAX": "equity",
    "NEWFX": "equity",
    "HAEMX": "equity",
    "TEMFX": "equity",
    "SFENX": "equity",
    "REMIX": "equity",
    "ABEMX": "equity",
    "GERIX": "equity",
    "HLEMX": "equity",
    "BGEFX": "equity",
    "WIEMX": "equity",
    "WAEMX": "equity",
    "MIEMX": "equity",
    "EEMAX": "equity",
    "FSEAX": "equity",
    "MADCX": "equity",
    "PDEZX": "equity",
    # ── EM Hard Currency Bond ─────────────────────────────────────────────────
    "PEBIX": "fixed_income",
    "EBNDX": "fixed_income",
    "PREMX": "fixed_income",
    "TGBAX": "fixed_income",
    "IEMIX": "fixed_income",
    "DBLEX": "fixed_income",
    "VWOB":  "fixed_income",
    "MSEDX": "fixed_income",
    # ── EM Local Currency Bond ────────────────────────────────────────────────
    "PFMIX": "fixed_income",
    "PRLAX": "fixed_income",
    # ── Global Equity ─────────────────────────────────────────────────────────
    "CWGIX": "equity",
    "TEDIX": "equity",
    "VHGEX": "equity",
    "OPGIX": "equity",
    "GPROX": "equity",
    "SGIGX": "equity",
    "MGGIX": "equity",
    "AIVSX": "equity",
    "MGIAX": "equity",
    "JGVAX": "equity",
    "HAINX": "equity",
    # ── AIAIM Singapore-listed ────────────────────────────────────────────────
    "0P0001I6KI.SI": "equity",
    "0P00008T6G.SI": "equity",
    "0P00008T6D.SI": "equity",
    "0P0001RMIN.SI": "equity",
    "0P0001KL95.SI": "fixed_income",
    "0P00008T6N.SI": "fixed_income",
}

def load_from_yfinance(
    fund_ticker: str,
    bench_ticker: str,
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull adjusted-close prices via yfinance, compound to monthly returns.

    Downloads fund and benchmark together as a two-ticker batch so yfinance
    returns a (Price, Ticker) MultiIndex, then extracts Close for each.

    Requires `pip install yfinance`. Free, no API key.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    px = yf.download([fund_ticker, bench_ticker], start=start, end=end,
                     progress=False, auto_adjust=True, group_by="ticker")
    if px.empty:
        raise RuntimeError(f"yfinance returned empty result for {fund_ticker}/{bench_ticker}")

    if isinstance(px.columns, pd.MultiIndex):
        close = pd.DataFrame({
            "portfolio": px[fund_ticker]["Close"],
            "benchmark": px[bench_ticker]["Close"],
        })
    else:
        raise RuntimeError("Unexpected yfinance schema; check tickers exist.")

    close      = close.dropna()
    monthly_px = close.resample("ME").last()
    return monthly_px.pct_change().dropna()


def load_from_csv(
    path: str,
    fund_col: str = "fund_nav",
    bench_col: str = "benchmark_nav",
    date_col: str = "date",
    is_returns: bool = False,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load fund + benchmark from a CSV.

    Two input modes:
      - is_returns=False (default): columns are PRICES / NAVs. We compute
        pct_change() and resample to month-end.
      - is_returns=True : columns are already periodic returns; we just align.
    """
    df           = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df           = df.set_index(date_col).sort_index()

    if fund_col not in df.columns or bench_col not in df.columns:
        raise KeyError(f"CSV must contain {fund_col!r} and {bench_col!r}. "
                       f"Got: {list(df.columns)}")

    if is_returns:
        out         = df[[fund_col, bench_col]].copy()
        out.columns = ["portfolio", "benchmark"]
        if pd.infer_freq(out.index) not in (None, "M", "ME", "MS"):
            out = (1 + out).resample("ME").prod() - 1
    else:
        monthly_px  = df[[fund_col, bench_col]].resample("ME").last()
        out         = monthly_px.pct_change()
        out.columns = ["portfolio", "benchmark"]

    return out.dropna()


def load_from_excel(
    path: str,
    sheet: str | int = 0,
    fund_col: str = "fund_nav",
    bench_col: str = "benchmark_nav",
    date_col: str = "date",
    is_returns: bool = False,
    skiprows: int = 0,
) -> pd.DataFrame:
    """
    Load from Excel (e.g. Bloomberg BDH export). Use `skiprows` to skip
    Bloomberg's 2-row header. Same semantics as load_from_csv.
    """
    df           = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows)
    df[date_col] = pd.to_datetime(df[date_col])
    df           = df.set_index(date_col).sort_index()

    if is_returns:
        out         = df[[fund_col, bench_col]].copy()
        out.columns = ["portfolio", "benchmark"]
        if pd.infer_freq(out.index) not in (None, "M", "ME", "MS"):
            out = (1 + out).resample("ME").prod() - 1
    else:
        monthly_px  = df[[fund_col, bench_col]].resample("ME").last()
        out         = monthly_px.pct_change()
        out.columns = ["portfolio", "benchmark"]

    return out.dropna()


# ===========================================================================
# PART 4 — End-to-end pipeline
# ===========================================================================

def monitor_fund(
    returns: pd.DataFrame,
    title: str = "",
    threshold: Optional[float] = None,
    initial_te: Optional[float] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    asset_class: Optional[str] = None,
) -> Tuple[CUSUMMonitor, pd.DataFrame]:
    """
    Run the CUSUM monitor on a fund's monthly returns and report results.

    Parameters
    ----------
    returns     : DataFrame with 'portfolio' and 'benchmark' columns.
    title       : display name for print / plot output.
    threshold   : explicit alarm threshold override. If None and asset_class is
                  given, the MC-calibrated value from ASSET_CLASS_PRESETS is used.
                  If both are None, falls back to 19.81.
    initial_te  : annualised TE seed. If None, estimated from the first ~24 months.
    save_path   : if provided, saves a PNG plot to this path.
    verbose     : print summary statistics.
    asset_class : one of "equity", "fixed_income", "balanced". When provided,
                  CUSUMMonitor is initialised with the matching MC-calibrated
                  mu_good, mu_bad, and threshold from ASSET_CLASS_PRESETS.
                  An explicit threshold argument overrides just the threshold.

    Returns
    -------
    (monitor, history_dataframe)
    """
    if initial_te is None:
        seed_window = returns.iloc[: min(24, max(6, len(returns) // 4))]
        excess      = seed_window["portfolio"] - seed_window["benchmark"]
        initial_te  = excess.std() * np.sqrt(12)
        if verbose:
            print(f"  estimated initial TE from first {len(seed_window)} months: "
                  f"{initial_te:.2%}")

    if asset_class is not None:
        monitor = CUSUMMonitor(initial_te=initial_te, preset=asset_class)
        if threshold is not None:
            monitor.threshold = threshold   # explicit override takes precedence
    else:
        t       = threshold if threshold is not None else 19.81
        monitor = CUSUMMonitor(initial_te=initial_te, threshold=t)

    history = monitor.run(returns)

    excess_log  = np.log((1 + returns["portfolio"]) / (1 + returns["benchmark"]))
    realised_ir = ((excess_log.mean() * 12) / (excess_log.std() * np.sqrt(12))
                   if excess_log.std() > 0 else 0.0)
    # TE uses arithmetic excess std × sqrt(12) — standard industry definition.
    excess_arith = returns["portfolio"] - returns["benchmark"]
    cum_excess   = (1 + returns["portfolio"]).prod() / (1 + returns["benchmark"]).prod() - 1

    if verbose:
        print(f"\n  {title}")
        print(f"  observations         : {len(returns)} months")
        print(f"  realised TE          : {excess_arith.std() * np.sqrt(12):.2%} annualised")
        print(f"  realised IR          : {realised_ir:+.2f}")
        print(f"  cumulative excess    : {cum_excess:+.1%}")
        if monitor.alarm:
            alarm_date = returns.index[monitor.alarm_t - 1]
            print(f"  ALARM                : month {monitor.alarm_t}  ({alarm_date:%Y-%m})")
        else:
            print(f"  ALARM                : none — manager remained on-process")

    if save_path:
        fig = monitor.plot(dates=returns.index, title=title)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  plot saved           : {save_path}")

    return monitor, history


def monitor_book(book: list, start: str = "2005-01-01") -> pd.DataFrame:
    """
    Run CUSUM across a whole book of (fund, benchmark, name) tuples.

    Each fund's asset class is looked up from TICKER_ASSET_CLASS using the fund
    ticker so that the correct MC-calibrated mu_good, mu_bad, and threshold are
    applied per fund. Funds with fewer than MIN_MONITORING_MONTHS months of
    history get status="INSUFFICIENT_HISTORY" and are excluded from the alarm
    summary — route them to a separate report page.

    Returns
    -------
    DataFrame with one row per manager. Key columns:
      name, fund, benchmark, months, realised_IR, alarm, alarm_date, status,
      data_quality_flags
    where status ∈ {"OK", "ALARM", "INSUFFICIENT_HISTORY", "ERROR"}.
    data_quality_flags is a semicolon-delimited string of DQ warnings; empty
    string means clean. Flagged funds should be footnoted in the report.
    """
    rows = []
    for fund, bench, name in book:
        try:
            rets     = load_from_yfinance(fund, bench, start=start)
            n_months = len(rets)

            if n_months < MIN_MONITORING_MONTHS:
                rows.append({
                    "name":        name,
                    "fund":        fund,
                    "benchmark":   bench,
                    "months":      n_months,
                    "realised_IR": None,
                    "n_alarms":    0,
                    "alarm":       None,
                    "alarm_date":  "",
                    "status":             "INSUFFICIENT_HISTORY",
                    "data_quality_flags": "",
                })
                continue

            # Data-quality check — surface warnings before running CUSUM
            dq_flags = sanity_check_returns(rets, fund_name=name)

            asset_class = TICKER_ASSET_CLASS.get(fund)
            m, _ = monitor_fund(
                rets,
                title=name,
                asset_class=asset_class,
                verbose=False,
            )
            excess_log = np.log((1 + rets["portfolio"]) / (1 + rets["benchmark"]))
            ir = (excess_log.mean() * 12) / (excess_log.std() * np.sqrt(12)) if excess_log.std() > 0 else 0.0
            rows.append({
                "name":               name,
                "fund":               fund,
                "benchmark":          bench,
                "months":             n_months,
                "realised_IR":        round(ir, 2),
                "n_alarms":           len(m.all_alarms),
                "alarm":              m.alarm,
                "alarm_date":         (rets.index[m.alarm_t - 1].strftime("%Y-%m")
                                       if m.alarm else ""),
                "status":             "ALARM" if m.alarm else "OK",
                "data_quality_flags": "; ".join(dq_flags) if dq_flags else "",
            })
        except Exception as e:
            rows.append({
                "name":        name,
                "fund":        fund,
                "benchmark":   bench,
                "months":      0,
                "realised_IR": None,
                "n_alarms":    0,
                "alarm":       None,
                "alarm_date":  f"ERROR: {str(e)[:60]}",
                "status":             "ERROR",
                "data_quality_flags": "",
            })
    return pd.DataFrame(rows)


# ===========================================================================
# PART 5 — Self-test demo (works offline)
# ===========================================================================

def _simulate_manager(n_months=240, change_month=120, ir_before=0.5,
                      ir_after=-0.3, te_annual=0.04, seed=11):
    """Synthetic manager with a known skill break — for offline testing."""
    rng      = np.random.default_rng(seed)
    te_m     = te_annual / np.sqrt(12)
    mkt      = rng.normal(0.08 / 12, 0.16 / np.sqrt(12), n_months)
    active   = np.empty(n_months)
    for t in range(n_months):
        ir        = ir_before if (change_month is None or t < change_month) else ir_after
        active[t] = ir * te_m + rng.normal(0, te_m)
    portfolio = mkt + active
    dates     = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    return pd.DataFrame({"portfolio": portfolio, "benchmark": mkt}, index=dates)


def demo():
    """Self-contained demo that runs without any external data."""
    print("=" * 72)
    print("CUSUM ACTIVE-MANAGER MONITOR — SELF-TEST DEMO")
    print("=" * 72)
    print("\nThreshold table (Table 2 of Philips-Yashchin-Stein 2003):")
    print(THRESHOLD_TABLE.to_string(index=False))

    print("\nSimulating a manager: IR=+0.5 for 120 months, then IR=-0.3 for the rest")
    rets = _simulate_manager(n_months=240, change_month=120,
                             ir_before=0.5, ir_after=-0.3, seed=11)
    m, _ = monitor_fund(rets, title="Demo: skill loss at month 120",
                        asset_class="equity", save_path="cusum_demo.png")
    print(f"\n  All skill-loss alarms  : {m.all_alarms}")
    print(f"  Regime outlier months  : {m.regime_outliers}")

    pa = post_alarm_returns(rets, m)
    if not pa.empty:
        print("\n  Post-alarm returns (cumulative excess vs benchmark):")
        for _, row in pa.iterrows():
            print(f"    Alarm {row['alarm_date']} | "
                  f"+{row['horizon']}m excess: {row['excess_cum']:+.1%} "
                  f"({'CORRECT' if row['correct_alarm'] else 'FALSE'})")

    ir_pt, ir_lo, ir_hi = bootstrap_ir_ci(rets)
    print(f"\n  Bootstrap IR 95% CI    : {ir_lo:+.2f} to {ir_hi:+.2f} (point: {ir_pt:+.2f})")
    print("Done.")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="cusum",
        description="CUSUM Active-Manager Monitor (Philips-Yashchin-Stein 2003)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Preset fund tickers: " + ", ".join(PRESET_PAIRS.keys()),
    )
    parser.add_argument("fund",  nargs="?", default=None,
                        help="Fund ticker (e.g. FCNTX) or preset key")
    parser.add_argument("bench", nargs="?", default=None,
                        help="Benchmark ticker (optional; inferred from preset)")
    parser.add_argument("--demo",       action="store_true",
                        help="Run self-contained offline demo")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Alarm threshold h override (default: use asset-class preset)")
    parser.add_argument("--output",     type=str,   default=None,
                        help="Output PNG path (default: <FUND>_cusum.png)")
    parser.add_argument("--start-date", type=str,   default="2005-01-01",
                        dest="start_date",
                        help="Start date for download (YYYY-MM-DD, default 2005-01-01)")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if args.fund is None:
        parser.print_help()
        return

    fund  = args.fund.upper()
    bench = (args.bench.upper() if args.bench
             else PRESET_PAIRS.get(fund, (None, "SPY"))[1])
    title = PRESET_PAIRS.get(fund, (fund, bench, f"{fund} vs {bench}"))[2]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    save  = args.output or os.path.join(out_dir, f"{fund}_cusum.png")

    print(f"Loading {fund} vs {bench} from {args.start_date} via yfinance...")
    try:
        rets = load_from_yfinance(fund, bench, start=args.start_date)
        print(f"  loaded {len(rets)} monthly observations")
    except Exception as e:
        print(f"  yfinance failed: {e}")
        return

    asset_class = TICKER_ASSET_CLASS.get(fund)
    monitor_fund(rets, title=title, threshold=args.threshold,
                 asset_class=asset_class, save_path=save)


if __name__ == "__main__":
    main()
