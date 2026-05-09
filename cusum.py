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


@dataclass
class CUSUMMonitor:
    """
    Sequential change-point detector for active-manager skill.

    Detects BOTH skill loss (lower CUSUM, L) and unexpected skill gain
    (upper CUSUM, L_up). After each alarm the statistic resets to zero
    so monitoring continues — all alarm times are recorded in all_alarms
    / all_alarms_up.

    Parameters
    ----------
    initial_te      : annualised tracking-error seed (e.g. 0.04 = 4 %).
    threshold       : alarm trigger. See THRESHOLD_TABLE. Default 19.81.
    gamma           : EWMA decay for tracking-error estimate (0.9 per paper).
    mu_good         : IR considered "good"        (default 0.5).
    mu_bad          : IR considered "bad"         (default 0.0).
    mu_exceptional  : IR considered "exceptional" (default 1.0) — upper alarm.
    """
    initial_te:     float = 0.04
    threshold:      float = 19.81
    gamma:          float = 0.9
    mu_good:        float = 0.5
    mu_bad:         float = 0.0
    mu_exceptional: float = 1.0

    # ── lower CUSUM state (skill loss) ──────────────────────────────────────
    t:             int            = 0
    L:             float          = 0.0
    sigma_monthly_sq: float       = field(init=False)
    e_prev:        Optional[float] = None
    alarm:         bool           = False
    alarm_t:       Optional[int]  = None
    all_alarms:    List[int]      = field(default_factory=list)

    # ── upper CUSUM state (skill gain) ──────────────────────────────────────
    L_up:          float          = 0.0
    alarm_up:      bool           = False
    alarm_up_t:    Optional[int]  = None
    all_alarms_up: List[int]      = field(default_factory=list)

    history:       list           = field(default_factory=list)

    def __post_init__(self):
        self.sigma_monthly_sq = (self.initial_te ** 2) / 12.0

    def update(self, r_portfolio: float, r_benchmark: float) -> dict:
        """Process one month's return pair."""
        self.t += 1

        # (1) log excess return
        e = np.log((1 + r_portfolio) / (1 + r_benchmark))

        # (2) EWMA tracking-error update (Von Neumann adjacent-difference estimator)
        if self.e_prev is not None:
            innovation = 0.5 * (e - self.e_prev) ** 2
            self.sigma_monthly_sq = (
                self.gamma * self.sigma_monthly_sq + (1 - self.gamma) * innovation
            )
        sigma_monthly = np.sqrt(self.sigma_monthly_sq)

        # (3) annualised IR using LAGGED sigma (keeps it ~unbiased)
        ir_hat = (e * np.sqrt(12)) / sigma_monthly if sigma_monthly > 0 else 0.0

        # (4) Lower Lindley recursion — accumulates when IR < midpoint(good, bad)
        drift_down = 0.5 * (self.mu_good + self.mu_bad) - ir_hat
        self.L = max(0.0, self.L + drift_down)

        if self.L >= self.threshold:
            self.all_alarms.append(self.t)
            if not self.alarm:          # record first alarm for backward compat
                self.alarm   = True
                self.alarm_t = self.t
            self.L = 0.0               # rolling restart — keep monitoring

        # (5) Upper Lindley recursion — accumulates when IR > midpoint(exceptional, good)
        drift_up = ir_hat - 0.5 * (self.mu_exceptional + self.mu_good)
        self.L_up = max(0.0, self.L_up + drift_up)

        if self.L_up >= self.threshold:
            self.all_alarms_up.append(self.t)
            if not self.alarm_up:
                self.alarm_up   = True
                self.alarm_up_t = self.t
            self.L_up = 0.0

        self.e_prev = e
        state = {
            "t":             self.t,
            "excess_return": e,
            "te_annual":     np.sqrt(self.sigma_monthly_sq * 12),
            "ir_hat":        ir_hat,
            "L":             self.L,
            "L_up":          self.L_up,
            "alarm":         self.alarm,
            "alarm_up":      self.alarm_up,
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
        """Two-panel plot: cumulative excess return + both CUSUM statistics."""
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

        # Panel 2: CUSUM statistics (L inverted downward, L_up upward)
        ax2.plot(x, df["L"],    lw=1.5, color="#1E2761", label="L (skill loss)")
        ax2.plot(x, df["L_up"], lw=1.2, color="#1E8449", ls="--", label="L_up (skill gain)")
        ax2.axhline(self.threshold, color="#C0392B", ls="--",
                    label=f"threshold = {self.threshold}")
        ax2.invert_yaxis()
        ax2.set_ylabel("CUSUM statistic (inverted)")
        ax2.grid(alpha=0.3)

        # Mark every skill-loss alarm
        for at in self.all_alarms:
            ax2.axvline(at, color="#C0392B", lw=0.9, alpha=0.6)
        if self.alarm_t:
            label_x = self.alarm_t + len(x) * 0.01
            ax2.text(label_x, self.threshold * 0.55,
                     f"ALARM @ t={self.alarm_t}",
                     color="#C0392B", va="center", fontsize=9, fontweight="bold")

        # Mark every skill-gain alarm
        for at in self.all_alarms_up:
            ax2.axvline(at, color="#1E8449", lw=0.9, alpha=0.6)

        ax2.legend(loc="lower right", fontsize=8)

        # x-axis date labels
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
# PART 2 — Statistical utilities
# ===========================================================================

def bootstrap_ir_ci(
    returns: pd.DataFrame,
    n_boot: int = 1000,
    alpha: float = 0.05,
    port_col: str = "portfolio",
    bench_col: str = "benchmark",
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Circular block bootstrap confidence interval on the annualised IR.

    Block length ~ n^(1/3) preserves autocorrelation in the excess return series.

    Returns
    -------
    (ir_point, ci_lower, ci_upper)  at the (0.5, alpha/2, 1-alpha/2) quantiles.
    """
    excess    = returns[port_col] - returns[bench_col]
    n         = len(excess)
    block_len = max(3, int(round(n ** (1 / 3))))
    ir_point  = (excess.mean() * 12) / (excess.std() * np.sqrt(12))

    rng      = np.random.default_rng(seed)
    boot_irs = []
    for _ in range(n_boot):
        starts  = rng.integers(0, n, size=int(np.ceil(n / block_len)))
        indices = np.concatenate([np.arange(s, s + block_len) % n for s in starts])[:n]
        sample  = excess.iloc[indices].values
        std     = sample.std()
        if std > 0:
            boot_irs.append((sample.mean() * 12) / (std * np.sqrt(12)))

    lo = float(np.percentile(boot_irs, 100 * alpha / 2))
    hi = float(np.percentile(boot_irs, 100 * (1 - alpha / 2)))
    return ir_point, lo, hi


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
    where correct_alarm = True when the fund underperformed after the alarm
    (i.e. the alarm was justified).
    """
    rows = []
    for alarm_t in monitor.all_alarms:
        alarm_date = (returns.index[alarm_t - 1].strftime("%Y-%m")
                      if alarm_t <= len(returns) else "")
        for h in horizons:
            start_idx = alarm_t        # 1-based alarm_t = 0-based start of post period
            end_idx   = start_idx + h
            if end_idx > len(returns):
                fund_cum = bench_cum = excess_cum = np.nan
                correct  = None
            else:
                window     = returns.iloc[start_idx:end_idx]
                fund_cum   = (1 + window[port_col]).prod() - 1
                bench_cum  = (1 + window[bench_col]).prod() - 1
                excess_cum = fund_cum - bench_cum
                correct    = bool(excess_cum < 0)
            rows.append({
                "alarm_t":      alarm_t,
                "alarm_date":   alarm_date,
                "horizon":      h,
                "fund_cum":     fund_cum,
                "bench_cum":    bench_cum,
                "excess_cum":   excess_cum,
                "correct_alarm": correct,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# PART 3 — Real-data loaders
# ===========================================================================

PRESET_PAIRS = {
    # ── Original 8 ──────────────────────────────────────────────────────────
    "FCNTX": ("FCNTX", "IWF", "Fidelity Contrafund vs Russell 1000 Growth"),
    "DODGX": ("DODGX", "IWD", "Dodge & Cox Stock vs Russell 1000 Value"),
    "AGTHX": ("AGTHX", "SPY", "AmFunds Growth Fund of America vs S&P 500"),
    "OTCFX": ("OTCFX", "IWM", "T. Rowe Price Small-Cap Stock vs Russell 2000"),
    "OAKIX": ("OAKIX", "EFA", "Oakmark International vs MSCI EAFE"),
    "TEDMX": ("TEDMX", "EEM", "Templeton Developing Markets vs MSCI EM"),
    "VWELX": ("VWELX", "AOR", "Vanguard Wellington vs 60/40 Allocation"),
    "PTTRX": ("PTTRX", "AGG", "PIMCO Total Return vs US Agg Bond"),
    # ── Extended universe ───────────────────────────────────────────────────
    "PRWCX": ("PRWCX", "AOR", "T. Rowe Price Capital Appreciation vs 60/40 Allocation"),
    "FMAGX": ("FMAGX", "SPY", "Fidelity Magellan vs S&P 500"),
    "SEQUX": ("SEQUX", "SPY", "Sequoia Fund vs S&P 500"),
    "FLPSX": ("FLPSX", "IWM", "Fidelity Low-Priced Stock vs Russell 2000"),
    "DODWX": ("DODWX", "EFA", "Dodge & Cox International vs MSCI EAFE"),
    "LSBRX": ("LSBRX", "AGG", "Loomis Sayles Bond vs US Agg Bond"),
    # ── Emerging Market Equity (27 funds, benchmark: EEM) ───────────────────
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
    "LZOXX": ("LZOXX", "EEM", "Lazard EM Institutional vs MSCI EM"),
    "MADCX": ("MADCX", "EEM", "Madison EM vs MSCI EM"),
    "PDEZX": ("PDEZX", "EEM", "PGIM EM vs MSCI EM"),
}


def load_from_yfinance(
    fund_ticker: str,
    bench_ticker: str,
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull adjusted-close prices via yfinance, compound to monthly returns.
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
    threshold: float = 19.81,
    initial_te: Optional[float] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[CUSUMMonitor, pd.DataFrame]:
    """
    Run the CUSUM monitor on a fund's monthly returns and report results.

    If `initial_te` is None, it's estimated from the first ~24 months of data.
    Returns (monitor, history_dataframe).
    """
    if initial_te is None:
        seed_window = returns.iloc[: min(24, max(6, len(returns) // 4))]
        excess      = seed_window["portfolio"] - seed_window["benchmark"]
        initial_te  = excess.std() * np.sqrt(12)
        if verbose:
            print(f"  estimated initial TE from first {len(seed_window)} months: "
                  f"{initial_te:.2%}")

    monitor = CUSUMMonitor(initial_te=initial_te, threshold=threshold)
    history = monitor.run(returns)

    excess      = returns["portfolio"] - returns["benchmark"]
    realised_ir = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
    cum_excess  = (1 + returns["portfolio"]).prod() / (1 + returns["benchmark"]).prod() - 1

    if verbose:
        print(f"\n  {title}")
        print(f"  observations         : {len(returns)} months")
        print(f"  realised TE          : {excess.std() * np.sqrt(12):.2%} annualised")
        print(f"  realised IR          : {realised_ir:+.2f}")
        print(f"  cumulative excess    : {cum_excess:+.1%}")
        if monitor.alarm:
            alarm_date = returns.index[monitor.alarm_t - 1]
            n_alarms   = len(monitor.all_alarms)
            print(f"  ALARM (first)        : month {monitor.alarm_t}  ({alarm_date:%Y-%m})")
            print(f"  total alarms         : {n_alarms}")
        else:
            print(f"  ALARM                : none — manager remained on-process")
        if monitor.alarm_up:
            up_date = returns.index[monitor.alarm_up_t - 1]
            print(f"  SKILL-GAIN alarm     : month {monitor.alarm_up_t}  ({up_date:%Y-%m})")

    if save_path:
        fig = monitor.plot(dates=returns.index, title=title)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  plot saved           : {save_path}")

    return monitor, history


def monitor_book(book: list, start: str = "2005-01-01",
                 threshold: float = 19.81) -> pd.DataFrame:
    """
    Run CUSUM across a whole book of (fund, benchmark, name) tuples.
    Returns a DataFrame with one row per manager.
    """
    rows = []
    for fund, bench, name in book:
        try:
            rets   = load_from_yfinance(fund, bench, start=start)
            m, _   = monitor_fund(rets, title=name, threshold=threshold, verbose=False)
            excess = rets["portfolio"] - rets["benchmark"]
            ir     = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
            rows.append({
                "name": name, "fund": fund, "benchmark": bench,
                "months":       len(rets),
                "realised_IR":  round(ir, 2),
                "n_alarms":     len(m.all_alarms),
                "alarm":        m.alarm,
                "alarm_date":   (rets.index[m.alarm_t - 1].strftime("%Y-%m")
                                 if m.alarm else ""),
                "alarm_up":     m.alarm_up,
                "alarm_up_date":(rets.index[m.alarm_up_t - 1].strftime("%Y-%m")
                                 if m.alarm_up else ""),
            })
        except Exception as e:
            rows.append({"name": name, "fund": fund, "benchmark": bench,
                         "months": 0, "realised_IR": None, "n_alarms": 0,
                         "alarm": None, "alarm_date": f"ERROR: {str(e)[:60]}",
                         "alarm_up": None, "alarm_up_date": ""})
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
        ir       = ir_before if (change_month is None or t < change_month) else ir_after
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
                        threshold=19.81, save_path="cusum_demo.png")
    print(f"\n  Rolling restart alarms : {m.all_alarms}")
    print(f"  Skill-gain alarms      : {m.all_alarms_up}")

    # Post-alarm returns
    pa = post_alarm_returns(rets, m)
    if not pa.empty:
        print("\n  Post-alarm returns (cumulative excess vs benchmark):")
        for _, row in pa.iterrows():
            print(f"    Alarm {row['alarm_date']} | "
                  f"+{row['horizon']}m excess: {row['excess_cum']:+.1%} "
                  f"({'CORRECT' if row['correct_alarm'] else 'FALSE'})")

    # Bootstrap CI
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
    parser.add_argument("--threshold",  type=float, default=19.81,
                        help="Alarm threshold h (default 19.81)")
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
    save  = args.output or f"{fund}_cusum.png"

    print(f"Loading {fund} vs {bench} from {args.start_date} via yfinance...")
    try:
        rets = load_from_yfinance(fund, bench, start=args.start_date)
        print(f"  loaded {len(rets)} monthly observations")
    except Exception as e:
        print(f"  yfinance failed: {e}")
        return

    monitor_fund(rets, title=title, threshold=args.threshold, save_path=save)


if __name__ == "__main__":
    main()
