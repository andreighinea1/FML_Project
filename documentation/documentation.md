# **Predicting the S&P 500**

## **1. Introduction**

This project aims to predict future values of the S&P 500 index using machine learning models based on historical
financial and economic data. We developed a feature engineering pipeline, tested multiple regression models, and
optimized hyperparameters to improve forecasting accuracy.

---

## **2. Data Description**

### **2.1 Dataset Overview**

The dataset consists of historical daily stock market data, including:

- **Market indices:** S&P 500, DJIA, HSI
- **Trading volumes:** S&P 500 volume, DJIA volume
- **Volatility index:** VIX
- **Macroeconomic indicators:** ADS index, US 3-month bond yield, joblessness
- **Uncertainty metrics:** Economic Policy Uncertainty (EPU), Geopolitical Risk Index (GPRD)

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

### **2.2 Feature Engineering**

We constructed various engineered features to enhance model performance:

#### **Rolling Statistics**

Used **rolling mean** and **standard deviations** for different time windows to capture market trends over
different periods.

- Windows used: **7, 14, 30, 90, 365 days**
- Applied to:
    - S&P 500 index and volume
    - DJIA index and volume
    - HSI index
    - VIX

#### **Lagged Features**

Introduced **365-day lagged features** for:

- S&P 500 index
- VIX
- S&P 500 volume

This helps capture past trends and seasonality.

#### **Autoencoder Embeddings**

A **feedforward autoencoder** was trained to **compress lagged features** into a **lower-dimensional representation**.
The embeddings were later used as additional input features for prediction models.

---

## **3. Machine Learning Models**

We tested **three** regression models to predict the next **1, 7, 14, 21, and 28** days of the S&P 500 index:

### **3.1 Models Used**

| Model                               | Description                                                    |
|-------------------------------------|----------------------------------------------------------------|
| **Linear Regression**               | Baseline model, simple but interpretable                       |
| **Ridge Regression**                | Regularized linear model to prevent overfitting                |
| **Support Vector Regression (SVR)** | Captures nonlinear relationships, sensitive to hyperparameters |

### **3.2 Training Process**

- **Train-Test Split:** 80% training, 20% testing (strict time-based split to prevent data leakage).
- **Feature Scaling:** MinMax scaling applied to continuous features, joblessness treated as an ordinal categorical
  feature.
- **Multi-Horizon Forecasting:** Models predict the next **1, 7, 14, 21, and 28 days** of S&P 500 index movement.

---

## **4. Hyperparameter Tuning with Optuna**

To optimize model performance, we used **Bayesian Optimization with Median Pruning** to efficiently tune
hyperparameters through Optuna.

### **4.1 Autoencoder Hyperparameter Tuning**

We optimized the following parameters:

| Hyperparameter         | Range             |
|------------------------|-------------------|
| **Encoding Dimension** | 10 to 30          |
| **Hidden Layer Size**  | 128 to 512        |
| **Dropout Rate**       | 0.1 to 0.3        |
| **Learning Rate**      | 0.0001 to 0.01    |
| **L1 Regularization**  | 0.00001 to 0.01   |
| **Weight Decay**       | 0.000001 to 0.001 |
| **Batch Size**         | 256, 512, 1024    |
| **Epochs**             | 75                |

The optimization process minimized the **Mean Squared Error (MSE)** on the reconstruction task, selecting the best model
state based on the lowest loss.

### **4.2 Ridge and SVR Hyperparameter Tuning**

Hyperparameters were tuned for **Ridge Regression** and **SVR**, focusing on maximizing **R² Score**:

| Model                | Hyperparameters Optimized                            | Range                                                       |
|----------------------|------------------------------------------------------|-------------------------------------------------------------|
| **Ridge Regression** | Alpha (Regularization Strength)                      | 0.01 to 10                                                  |
| **SVR**              | C (Regularization), Epsilon (Tolerance), Kernel Type | C: 0.1 to 10, Epsilon: 0.01 to 1, Kernel: rbf, linear, poly |

The optimization targeted the **first-day-ahead prediction (`sp500_next_1`)**, assuming improvements in this would
generalize to longer-term predictions.

---

## **5. Results and Evaluation**

We evaluated models based on **Mean Squared Error (MSE)** and **R² Score** for each forecasting horizon.

### **5.1 Model Performance Table**

| Model                 | Day 1                    | Day 7                    | Day 14                   | Day 21                   | Day 28                   |
|-----------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **Linear Regression** | MSE: 7.59e-6, R²: 0.9997 | MSE: 0.00026, R²: 0.9902 | MSE: 0.00068, R²: 0.9743 | MSE: 0.00095, R²: 0.9639 | MSE: 0.00114, R²: 0.9563 |
| **Ridge Regression**  | MSE: 1.21e-5, R²: 0.9995 | MSE: 0.00024, R²: 0.9906 | MSE: 0.00063, R²: 0.9762 | MSE: 0.00091, R²: 0.9652 | MSE: 0.00106, R²: 0.9593 |
| **SVR**               | MSE: 0.00068, R²: 0.9744 | MSE: 0.00061, R²: 0.9767 | MSE: 0.00110, R²: 0.9584 | MSE: 0.00099, R²: 0.9624 | MSE: 0.00137, R²: 0.9476 |

### **5.2 Visualizing Predictions**

Below are the prediction plots for each model over different time horizons.

#### **Linear Regression Predictions**

![Linear Regression (1 Day)](../plots/training/LinearRegression_day_1.png)  
![Linear Regression (7 Days)](../plots/training/LinearRegression_day_7.png)  
![Linear Regression (28 Days)](../plots/training/LinearRegression_day_28.png)

#### **Ridge Regression Predictions**

![Ridge Regression (1 Day)](../plots/training/Ridge_day_1.png)  
![Ridge Regression (7 Days)](../plots/training/Ridge_day_7.png)  
![Ridge Regression (28 Days)](../plots/training/Ridge_day_28.png)

#### **SVR Predictions**

![SVR (1 Day)](../plots/training/SVR_day_1.png)  
![SVR (7 Days)](../plots/training/SVR_day_7.png)  
![SVR (28 Days)](../plots/training/SVR_day_28.png)

---

## **6. Conclusion**

### **6.1 Key Takeaways**

- **Feature Engineering**: Using rolling averages and lagged features helped capture market trends effectively.
- **Model Performance**: Ridge Regression was the best-performing model across all prediction horizons, followed by
  Linear Regression. SVR struggled even with short-term predictions, but deteriorating fast for longer-term predictions.
- **Prediction Accuracy**: Short-term (1-7 days) predictions were highly accurate (**R² > 0.99**), but accuracy
  decreased for longer-term forecasts.

### **6.2 Future Improvements**

- **Deep Learning Models:** Investigate LSTMs or Transformers for better sequential modeling.
- **Additional Features:** Introduce macroeconomic variables such as bond yields, credit spreads, and investor
  sentiment.
- **Extended Forecasts:** Experiment with prediction windows beyond **28 days**.

---

### **Final Thoughts**

This project successfully demonstrated that machine learning models, particularly **Ridge Regression**, can provide
highly accurate short-term forecasts for the S&P 500. However, longer-term predictions remain challenging, requiring
further feature engineering and potential deep learning techniques.