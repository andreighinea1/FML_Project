# Data Analysis and Visualization

## Introduction

This section presents the data analysis conducted on historical stock market data. The goal is to identify key trends, correlations, and relationships among financial indicators. These insights help in understanding market behavior and selecting features for predictive modeling.

---

## Correlation Analysis

### Correlation Heatmap

![Correlation Heatmap](../plots/analysis/correlation_heatmap.png)

- This heatmap shows how different financial indicators relate to each other.
- **Red areas** indicate strong positive correlations, meaning the variables move together.
- **Blue areas** represent negative correlations, where one increases as the other decreases.
- The **S&P 500** is **strongly correlated** with **DJIA** and **HSI**, making them valuable for prediction.

---

## Market Trends and Comparisons

### Normalized Stock Indices Over Time

![Normalized Stock Indices](../plots/analysis/normalized_stock_indices.png)

- Compares the performance of **S&P 500 (blue)**, **DJIA (green)**, and **HSI (orange)** over time.
- Indices are **normalized** for direct comparison.
- Major economic downturns like the **2008 Financial Crisis** and **2020 COVID-19 crash** are clearly visible.

### S&P 500 Over Time

![S&P 500 Over Time](../plots/analysis/sp500_over_time.png)

- Shows the **long-term movement** of the **S&P 500** index.
- Highlights major **market crashes** and **economic recoveries**.

---

## Relationship Between Market Indices

### S&P 500 vs DJIA

![S&P 500 vs DJIA](../plots/analysis/sp500_vs_djia.png)

- Shows a **strong linear relationship** between the **S&P 500** and **DJIA**.
- Indicates that both indices tend to move **in the same direction**.

### S&P 500 vs Hang Seng Index (HSI)

![S&P 500 vs HSI](../plots/analysis/sp500_vs_hsi.png)

- Displays a **positive correlation** between **S&P 500** and **HSI**.
- However, **HSI shows more variation**, indicating **regional differences** in market behavior.

---

## Trading Volume and Macroeconomic Factors

### S&P 500 Trading Volume vs DJIA Trading Volume

![S&P 500 Volume vs DJIA Volume](../plots/analysis/sp500_volume_vs_djia_volume.png)

- **Trading volumes** of **S&P 500** and **DJIA** move together.
- Suggests that **investor activity** is similar across both indices.

### Interest Rates (US3M) vs S&P 500 Trading Volume

![US3M vs S&P 500 Volume](../plots/analysis/us3m_vs_sp500_volume.png)

- **Negative correlation** between **interest rates (US3M)** and **S&P 500 trading volume**.
- Higher interest rates usually lead to **lower market activity**.

### VIX (Volatility Index) vs S&P 500

![VIX vs S&P 500](../plots/analysis/vix_vs_sp500.png)

- The **VIX (Volatility Index)** measures market uncertainty.
- **High volatility** is often associated with **S&P 500 declines**.

---

## Joblessness and Market Trends

### S&P 500 Over Time with Joblessness Heatmap

![S&P 500 with Joblessness](../plots/analysis/sp500_joblessness_heatmap.png)

- This plot overlays **joblessness rates** on the **S&P 500 trend**.
- **Red areas** indicate **high unemployment**, often following **economic downturns**.
- The **2008 crisis** and **2020 pandemic** both saw **rising unemployment and falling stock prices**.

---

## Conclusion

The analysis highlights key relationships between stock indices, trading volumes, interest rates, and macroeconomic factors. These insights will guide feature selection for machine learning models to improve stock market predictions.
