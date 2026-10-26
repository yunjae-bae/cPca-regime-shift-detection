# Multivariate Time-Series Change Detection

PCA and contrastive PCA analysis of six U.S. macroeconomic series. A Dual-State Structural Index (DSSI) combines covariance trace ratios and principal-component similarity to detect changes sequentially. The saved example flags May 2020.

## Usage

```sh
pip install -r requirements.txt
```

Download these FRED series as CSV files, named by series ID, beside `PCA_economy.ipynb`:

| File | Series |
|---|---|
| `INDPRO.csv` | Industrial production |
| `RSAFS.csv` | Retail and food services sales |
| `PCE.csv` | Personal consumption expenditures |
| `UNRATE.csv` | Unemployment rate |
| `CPIAUCSL.csv` | Consumer price index |
| `DGS10.csv` | 10-year Treasury yield |

Run the notebook from top to bottom. It aligns observations monthly and compares an expanding background with 12-month target windows. The final example uses DSSI threshold 1.0 and projection exit threshold 1.5.

Run the synthetic checks with `python test_analysis.py`.

## Sources

- [FRED — Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/)
