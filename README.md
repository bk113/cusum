# CUSUM Active Manager Monitor

A quantitative framework for detecting persistent active manager underperformance using the **Philips-Yashchin-Stein (2003)** CUSUM change-point methodology.

## Overview

The tool monitors a universe of 67 active funds against their respective benchmarks, flagging statistically significant degradation in Information Ratio in real time. Key design features:

- **CUSUM statistic** with threshold h = 19.81 (ARL ≈ 60 months per fund under H₀)
- **24-month calibration gate**: sigma EWMA warms up from day 1, but L accumulation begins only after the calibration window, preventing in-sample contamination
- **IR winsorisation** at ±5: prevents a single crash month (e.g. March 2020, |IR_hat| > 40 with stale EWMA sigma) from single-handedly crossing the threshold
- **Log-excess returns** used consistently across CUSUM accumulation, bootstrap CI, and IR computation
- **Rolling restart**: CUSUM resets after each alarm, capturing all alarm events across history
- **Two-sided CUSUM**: monitors both deterioration (L↑) and exceptional outperformance (L_up↑)
- **Multiple-testing disclosure**: with 67 funds at ARL=60m, expected false alarms ≈ 13.4/year across the universe

## Fund Universe

| Asset Class | Funds | Benchmark |
|---|---|---|
| US Core Equity | 7 | S&P 500 (SPY) |
| US Bond / Fixed Income | 4 | AGG / PTTRX |
| Balanced | 2 | Blended |
| EM Equity | 27 | EEM |
| EM Hard CCY Bonds | 9 | EMB |
| EM Local CCY Bonds | 2 | EMLC |
| Global Equity (ACWI) | 7 | ACWI |
| Global Equity (URTH) | 4 | URTH |
| AIAIM Singapore | 7 | ACWI / EMB |

## Statistical Outputs (per fund)

- IR with 95% circular block bootstrap CI (Politis-White 2004), block length = n^(1/3)
- t-statistic of alpha: IR × √(n/12)
- Hit rate: % months fund beats benchmark
- Up/Down capture ratios
- GFC (Oct 2007–Mar 2009) and COVID (Feb–Apr 2020) regime tagging of first alarm
- Signal quality: % of ALL alarm events where fund underperformed at +12M

## Installation

```bash
pip install yfinance pandas numpy matplotlib reportlab scipy
```

## Usage

```bash
# Run CUSUM analysis and save individual fund chart
python cusum.py FCNTX SPY

# Generate full PDF report
python report.py
```

Output files are saved to the `output/` folder:
- `output/cusum_report.pdf` — full paginated report

## Report Structure

1. Cover page
2. Methodology overview
3. Fund universe pages (6 funds/page, 3×2 grid with CUSUM charts)
4. Summary results table (IR, t-stat, Hit%, Up/Dn Cap, Regime, alarms)
5. Post-alarm return analysis (paginated, 14 rows/page)

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Threshold h | 19.81 | ARL ≈ 60 months under H₀ |
| Calibration window | 24 months | L frozen; sigma EWMA runs |
| IR cap | ±5 | Winsorise extreme monthly IR |
| EWMA gamma | 0.90 | Variance decay factor |
| μ_good | 0.5 | IR threshold for "good" regime |
| μ_bad | 0.0 | IR threshold for "bad" regime |
| μ_exceptional | 1.0 | IR threshold for exceptional regime |
| Bootstrap block | n^(1/3) | Circular block bootstrap |
| Min monitoring | 36 months | Short history flag |
