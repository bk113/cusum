# CUSUM Active Manager Monitor

A quantitative framework for detecting persistent active manager underperformance using the **Philips-Yashchin-Stein (2003)** CUSUM change-point methodology.

## Overview

The tool monitors a universe of 67 active funds against their respective benchmarks, flagging statistically significant degradation in Information Ratio in real time. Key design features:

- **CUSUM statistic** with threshold h = 19.81 (ARL ≈ 60 months per fund under H₀)
- **24-month calibration gate**: sigma EWMA warms up from day 1, but L accumulation begins only after the calibration window, preventing in-sample contamination
- **Regime outlier handling**: freeze L if |IR_hat_annual| > 14.0 — crash months (e.g. March 2020) are skipped entirely; raw IR preserved unchanged (not winsorised)
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

<!-- GENERATED PARAMETERS — DO NOT EDIT BY HAND -->
<!-- Regenerate with: python generate_parameter_doc.py -->

| Parameter | Value | Source | Mechanism / Notes |
| --- | --- | --- | --- |
| h (threshold) | 19.81 | CUSUMMonitor.threshold (dataclass default) | Alarm fires when L >= h. Fallback; overwritten at import by thresholds_cache.json (MC-calibrated per asset class). |
| gamma | 0.9 | CUSUMMonitor.gamma (dataclass default) | EWMA decay for tracking-error estimate (Von Neumann estimator). ~10-month half-life. |
| mu_good -- equity | 0.5 | cusum._MU_EQUITY_GOOD (module constant) | Annualised IR considered 'good' for equity mandates. Moustakides-optimal drift midpoint. |
| mu_good -- fixed income / EM debt | 0.3 | cusum._MU_FIXEDINC_GOOD (module constant) | Annualised IR considered 'good' for fixed income and EM debt mandates. |
| mu_good -- balanced | 0.4 | cusum._MU_BALANCED_GOOD (module constant) | Annualised IR considered 'good' for balanced mandates. |
| mu_bad | 0.0 | CUSUMMonitor.mu_bad (dataclass default) | Annualised IR considered 'bad' (underperforming). Same across all asset classes. |
| calibration_window | 24 months | CUSUMMonitor.calibration_months (dataclass default) | L frozen at 0 during this window; sigma EWMA warms up from month 1. Prevents in-sample contamination. |
| regime_threshold | 14.0 | CUSUMMonitor.regime_threshold (dataclass default) | If |ir_hat_annual| exceeds this value, the month is a regime outlier: L is FROZEN (not updated), raw IR preserved unchanged. NOT winsorisation -- see cusum.py lines 149-153, 239. |
| MIN_MONITORING_MONTHS | 60 | cusum.MIN_MONITORING_MONTHS (module constant) | Minimum months of live CUSUM history required for a fund to appear in the main analysis. Funds below this threshold are flagged as insufficient-history. |

<!-- END GENERATED PARAMETERS -->
