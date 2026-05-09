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
       python cusum.py FCNTX SPY 2005-01-01

2) Run on your own CSV (cols: date, fund_nav, benchmark_nav):
       python -c "from cusum import load_from_csv, monitor_fund; \\
                  r = load_from_csv('myfund.csv', 'fund_nav', 'benchmark_nav'); \\
                  monitor_fund(r, title='my fund')"

3) Run synthetic self-test (no internet, no data file):
       python cusum.py --demo

DEPENDENCIES
------------
Required: numpy, pandas, matplotlib
Optional: yfinance  (only needed for `load_from_yfinance`)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
    Sequential change-point detector for active-manager skill loss.

    Parameters
    ----------
    initial_te : annualised tracking-error seed (e.g. 0.04 = 4%).
    threshold  : alarm trigger. See THRESHOLD_TABLE. Default 19.81 ~ 1 false alarm / 5 yrs.
    gamma      : EWMA decay for tracking-error estimate. 0.9 per the paper.
    mu_good    : IR considered "good" (default 0.5)
    mu_bad     : IR considered "bad"  (default 0.0)
    """
    initial_te: float = 0.04
    threshold: float = 19.81
    gamma: float = 0.9
    mu_good: float = 0.5
    mu_bad: float = 0.0

    # state
    t: int = 0
    L: float = 0.0
    sigma_monthly_sq: float = field(init=False)
    e_prev: Optional[float] = None
    alarm: bool = False
    alarm_t: Optional[int] = None
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.sigma_monthly_sq = (self.initial_te ** 2) / 12.0

    def update(self, r_portfolio: float, r_benchmark: float) -> dict:
        """Process one month's return."""
        self.t += 1

        # (1) log excess return
        e = np.log((1 + r_portfolio) / (1 + r_benchmark))

        # (3) EWMA tracking-error update (Von Neumann adjacent-difference estimator)
        if self.e_prev is not None:
            innovation = 0.5 * (e - self.e_prev) ** 2
            self.sigma_monthly_sq = (
                self.gamma * self.sigma_monthly_sq + (1 - self.gamma) * innovation
            )
        sigma_monthly_prev = np.sqrt(self.sigma_monthly_sq)

        # (5) annualised IR using LAGGED sigma (keeps it ~unbiased)
        ir_hat = (e * np.sqrt(12)) / sigma_monthly_prev if sigma_monthly_prev > 0 else 0.0

        # (8) Lindley recursion. Drift = (mu_good + mu_bad)/2 - ir_hat.
        # Defaults give 0.25 - ir_hat: clear the bar (IR>=0.5) -> tally shrinks.
        drift = 0.5 * (self.mu_good + self.mu_bad) - ir_hat
        self.L = max(0.0, self.L + drift)

        if (not self.alarm) and self.L >= self.threshold:
            self.alarm = True
            self.alarm_t = self.t

        self.e_prev = e
        state = {
            "t": self.t, "excess_return": e,
            "te_annual": np.sqrt(self.sigma_monthly_sq * 12),
            "ir_hat": ir_hat, "L": self.L, "alarm": self.alarm,
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
        """Two-panel plot: cumulative excess return + Page's plot of L_t."""
        df = pd.DataFrame(self.history)
        if df.empty:
            raise RuntimeError("no history; run the monitor first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        x = df["t"].values

        ax1.plot(x, df["excess_return"].cumsum(), lw=1.5, color="#1E2761")
        ax1.axhline(0, color="grey", lw=0.6)
        ax1.set_ylabel("cumulative log excess return")
        ax1.set_title(title)
        ax1.grid(alpha=0.3)

        ax2.plot(x, df["L"], lw=1.5, color="#1E2761")
        ax2.axhline(self.threshold, color="#F96167", ls="--",
                    label=f"alarm threshold = {self.threshold}")
        if self.alarm_t:
            ax2.axvline(self.alarm_t, color="#F96167", lw=1.2, alpha=0.7)
            label_x = self.alarm_t + len(x) * 0.01
            ax2.text(label_x, self.threshold * 0.6, f"ALARM @ t={self.alarm_t}",
                     color="#F96167", va="center", fontsize=10, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_ylabel("CUSUM statistic L_t (inverted)")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

        # if we have real dates, replace the x-axis ticks with year labels
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
# PART 2 — Real-data loaders
# ===========================================================================

# Pre-loaded fund/benchmark pairs to make `python cusum.py FCNTX` quick.
PRESET_PAIRS = {
    "FCNTX": ("FCNTX", "IWF",  "Fidelity Contrafund vs Russell 1000 Growth"),
    "DODGX": ("DODGX", "IWD",  "Dodge & Cox Stock vs Russell 1000 Value"),
    "AGTHX": ("AGTHX", "SPY",  "AmFunds Growth Fund of America vs S&P 500"),
    "OTCFX": ("OTCFX", "IWM",  "T. Rowe Price Small-Cap Stock vs Russell 2000"),
    "OAKIX": ("OAKIX", "EFA",  "Oakmark International vs MSCI EAFE"),
    "TEDMX": ("TEDMX", "EEM",  "Templeton Developing Markets vs MSCI EM"),
    "VWELX": ("VWELX", "AOR",  "Vanguard Wellington vs 60/40 Allocation"),
    "PTTRX": ("PTTRX", "AGG",  "PIMCO Total Return vs US Agg Bond"),
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

    close = close.dropna()
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
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.set_index(date_col).sort_index()

    if fund_col not in df.columns or bench_col not in df.columns:
        raise KeyError(f"CSV must contain {fund_col!r} and {bench_col!r}. "
                       f"Got: {list(df.columns)}")

    if is_returns:
        out = df[[fund_col, bench_col]].copy()
        out.columns = ["portfolio", "benchmark"]
        if pd.infer_freq(out.index) not in (None, "M", "ME", "MS"):
            out = (1 + out).resample("ME").prod() - 1
    else:
        monthly_px = df[[fund_col, bench_col]].resample("ME").last()
        out = monthly_px.pct_change()
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
    df = pd.read_excel(path, sheet_name=sheet, skiprows=skiprows)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    if is_returns:
        out = df[[fund_col, bench_col]].copy()
        out.columns = ["portfolio", "benchmark"]
        if pd.infer_freq(out.index) not in (None, "M", "ME", "MS"):
            out = (1 + out).resample("ME").prod() - 1
    else:
        monthly_px = df[[fund_col, bench_col]].resample("ME").last()
        out = monthly_px.pct_change()
        out.columns = ["portfolio", "benchmark"]
    return out.dropna()


# ===========================================================================
# PART 3 — End-to-end pipeline
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
    """
    if initial_te is None:
        seed_window = returns.iloc[: min(24, max(6, len(returns) // 4))]
        excess = seed_window["portfolio"] - seed_window["benchmark"]
        initial_te = excess.std() * np.sqrt(12)
        if verbose:
            print(f"  estimated initial TE from first {len(seed_window)} months: {initial_te:.2%}")

    monitor = CUSUMMonitor(initial_te=initial_te, threshold=threshold)
    history = monitor.run(returns)

    excess = returns["portfolio"] - returns["benchmark"]
    realised_ir = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
    cum_excess = (1 + returns["portfolio"]).prod() / (1 + returns["benchmark"]).prod() - 1

    if verbose:
        print(f"\n  {title}")
        print(f"  observations         : {len(returns)} months")
        print(f"  realised TE          : {excess.std() * np.sqrt(12):.2%} annualised")
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


def monitor_book(book: list, start: str = "2005-01-01",
                 threshold: float = 19.81) -> pd.DataFrame:
    """
    Run CUSUM across a whole book of (fund, benchmark, name) tuples.

    Returns a DataFrame with one row per manager: status, alarm date, IR, etc.
    """
    rows = []
    for fund, bench, name in book:
        try:
            rets = load_from_yfinance(fund, bench, start=start)
            m, _ = monitor_fund(rets, title=name, threshold=threshold, verbose=False)
            excess = rets["portfolio"] - rets["benchmark"]
            ir = (excess.mean() * 12) / (excess.std() * np.sqrt(12))
            rows.append({
                "name": name, "fund": fund, "benchmark": bench,
                "months": len(rets),
                "realised_IR": round(ir, 2),
                "alarm": m.alarm,
                "alarm_date": rets.index[m.alarm_t - 1].strftime("%Y-%m") if m.alarm else "",
            })
        except Exception as e:
            rows.append({"name": name, "fund": fund, "benchmark": bench,
                         "months": 0, "realised_IR": None, "alarm": None,
                         "alarm_date": f"ERROR: {str(e)[:60]}"})
    return pd.DataFrame(rows)


# ===========================================================================
# PART 4 — Self-test demo (works offline)
# ===========================================================================

def _simulate_manager(n_months=240, change_month=120, ir_before=0.5,
                      ir_after=-0.3, te_annual=0.04, seed=11):
    """Synthetic manager with a known skill break — for offline testing."""
    rng = np.random.default_rng(seed)
    te_m = te_annual / np.sqrt(12)
    mkt = rng.normal(0.08 / 12, 0.16 / np.sqrt(12), n_months)
    active = np.empty(n_months)
    for t in range(n_months):
        ir = ir_before if (change_month is None or t < change_month) else ir_after
        active[t] = ir * te_m + rng.normal(0, te_m)
    portfolio = mkt + active
    dates = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    return pd.DataFrame({"portfolio": portfolio, "benchmark": mkt}, index=dates)


def demo():
    """Self-contained demo that runs without any external data."""
    print("=" * 72)
    print("CUSUM ACTIVE-MANAGER MONITOR — SELF-TEST DEMO")
    print("=" * 72)
    print("\nThreshold table (Table 2 of Philips-Yashchin-Stein 2003):")
    print(THRESHOLD_TABLE.to_string(index=False))

    print("\nSimulating a manager: IR=+0.5 for 120 months, then IR=-0.3 for the rest")
    rets = _simulate_manager(n_months=240, change_month=120, ir_before=0.5, ir_after=-0.3, seed=11)
    monitor_fund(rets, title="Demo: skill loss at month 120",
                 threshold=19.81, save_path="cusum_demo.png")
    print("Done.")


# ===========================================================================
# CLI
# ===========================================================================

def _print_help():
    print(__doc__)
    print("\nPRESET FUND/BENCHMARK PAIRS (use first ticker as arg):")
    for k, (f, b, name) in PRESET_PAIRS.items():
        print(f"  {f:7s} -> bench {b:5s}  {name}")


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    if args[0] == "--demo":
        demo()
        return

    fund = args[0].upper()
    bench = args[1].upper() if len(args) > 1 else PRESET_PAIRS.get(fund, (None, "SPY"))[1]
    start = args[2] if len(args) > 2 else "2005-01-01"
    title = PRESET_PAIRS.get(fund, (fund, bench, f"{fund} vs {bench}"))[2]

    print(f"Loading {fund} vs {bench} from {start} via yfinance...")
    print("(Requires internet + `pip install yfinance`)\n")

    try:
        rets = load_from_yfinance(fund, bench, start=start)
        print(f"  loaded {len(rets)} monthly observations")
    except Exception as e:
        print(f"  yfinance failed: {e}")
        print("\nFALLBACK: save your data to a CSV with columns 'date', 'fund_nav',")
        print("'benchmark_nav' and run:")
        print("  python -c \"from cusum import load_from_csv, monitor_fund; "
              "monitor_fund(load_from_csv('your.csv'), title='your fund', save_path='out.png')\"")
        return

    monitor_fund(rets, title=title, save_path=f"{fund}_cusum.png")


if __name__ == "__main__":
    main()
