# CUSUM Active-Manager Monitor

Single-file implementation of the **Philips-Yashchin-Stein (2003)** statistical process control procedure for monitoring active portfolio managers.

## What it does

Applies a CUSUM (cumulative sum) change-point detector to a fund's monthly excess returns. It raises an alarm when the manager's Information Ratio falls below a threshold — signalling a potential loss of skill.

## Installation

**Clone the repo:**
```bash
git clone https://github.com/bk113/cusum.git
cd cusum
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib pymupdf yfinance
```

> `yfinance` is required for live Yahoo Finance data.  
> `pymupdf` is required for PDF report generation (`report.py`).  
> All other packages are required for core CUSUM functionality.

## Usage

### 1. Live fund from Yahoo Finance
```bash
python cusum.py FCNTX
python cusum.py FCNTX SPY 2005-01-01
```

### 2. Your own CSV
```python
from cusum import load_from_csv, monitor_fund
returns = load_from_csv('myfund.csv', 'fund_nav', 'benchmark_nav')
monitor_fund(returns, title='My Fund', save_path='out.png')
```

### 3. Your own Excel
```python
from cusum import load_from_excel, monitor_fund
returns = load_from_excel('myfund.xlsx', fund_col='fund_nav', bench_col='benchmark_nav')
monitor_fund(returns, title='My Fund', save_path='out.png')
```

### 4. Offline self-test demo
```bash
python cusum.py --demo
```

## Preset fund/benchmark pairs

| Ticker | Benchmark | Description |
|--------|-----------|-------------|
| FCNTX | IWF | Fidelity Contrafund vs Russell 1000 Growth |
| DODGX | IWD | Dodge & Cox Stock vs Russell 1000 Value |
| AGTHX | SPY | AmFunds Growth Fund of America vs S&P 500 |
| OTCFX | IWM | T. Rowe Price Small-Cap Stock vs Russell 2000 |
| OAKIX | EFA | Oakmark International vs MSCI EAFE |
| TEDMX | EEM | Templeton Developing Markets vs MSCI EM |
| VWELX | AOR | Vanguard Wellington vs 60/40 Allocation |
| PTTRX | AGG | PIMCO Total Return vs US Agg Bond |

## Alarm threshold table (Table 2, Philips-Yashchin-Stein 2003)

| Threshold | ARL (IR=+0.5) | ARL (IR=0) | ARL (IR=-0.5) |
|-----------|--------------|------------|---------------|
| 11.81 | 24 months | 16 | 11 |
| 15.00 | 36 months | 22 | 15 |
| 17.60 | 48 months | 27 | 18 |
| **19.81** | **60 months** | **32** | **21** |
| 21.79 | 72 months | 37 | 23 |
| 23.59 | 84 months | 41 | 25 |

Default threshold is **19.81** (~1 false alarm per 5 years).

## Reference

Philips, T., Yashchin, E., & Stein, D. (2003). *Using statistical process control to monitor active managers.* Journal of Portfolio Management, 30(1), 86–94.
