# 📈 US Bond Yield Prediction

A time-series forecasting project that predicts **10-Year US Treasury Bond Yields (DGS10)** using macroeconomic indicators and multiple machine learning approaches. The project compares statistical (ARIMA) and deep learning (LSTM, MLP) models to evaluate forecasting accuracy on interest rate data spanning **2000–2024**.

---

## 🎯 Objective

US Treasury yields are a key benchmark in financial markets, influencing mortgage rates, corporate borrowing costs, and investment decisions. This project builds predictive models for the 10-year yield using publicly available economic data from the **Federal Reserve Economic Data (FRED)** database.

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| **FF** | Federal Funds Rate — the interest rate at which banks lend to each other overnight |
| **INDPRO** | Industrial Production Index — a measure of real output in manufacturing, mining, and utilities |
| **CPI** | Consumer Price Index — tracks changes in the price level of a basket of consumer goods |
| **VIX** | CBOE Volatility Index — measures market expectations of near-term volatility |
| **DGS10** | 10-Year Treasury Constant Maturity Rate *(target variable)* |

- **Time Range:** January 2000 – August 2024 (296 monthly observations)
- **Source:** [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/)
- **Train/Test Split:** 2000–2020 (training) / 2020–2024 (testing)

### Key Correlations with DGS10
| Feature | Correlation |
|---------|------------|
| FF | **0.75** (strong positive) |
| INDPRO | 0.15 |
| CPI | 0.09 |
| VIX | -0.005 |

> The Federal Funds Rate is the strongest predictor of 10-year yields, consistent with macroeconomic theory.

---

## 🧪 Methodology

### 1. EDA & Preprocessing (`01_EDA_Preprocessing.ipynb`)
- Loaded and inspected raw FRED data (296 rows × 6 columns, no missing values)
- Computed correlation matrix to identify feature relationships
- Applied **MinMaxScaler** normalization to input features (FF, INDPRO, CPI, VIX)
- Exported processed dataset for use across all modeling notebooks

### 2. ARIMA Model (`02_ARIMA.ipynb`)
- Performed **Augmented Dickey-Fuller (ADF)** stationarity test on DGS10 series
  - Raw series: p-value = **0.127** (non-stationary)
  - After differencing: p-value = **9.77e-22** (stationary ✅)
- Applied **seasonal decomposition** to extract trend, seasonal, and residual components
- Fitted ARIMA model for univariate time-series forecasting

### 3. LSTM-Dense Hybrid Model (`03_LSTM.ipynb`)
- Built a **Sequential LSTM model** with 2 LSTM layers (64 units each) + Dense output
- Used early stopping (patience=10) to prevent overfitting
- Training converged in **23 epochs** (out of 100 max)

### 4. MLP Model (`04_MLP.ipynb`)
- Constructed a **4-layer feedforward neural network**: Dense(128) → Dense(64) → Dense(64) → Dense(32) → Output
- Applied **Dropout (0.3)** after each hidden layer for regularization
- Trained with Adam optimizer (lr=0.0001) and early stopping (patience=10)

---

## 📈 Results

| Model | MSE | MAE |
|-------|-----|-----|
| **LSTM-Dense** | **1.08** | **0.91** |
| MLP | 1.38 | 1.06 |

> The LSTM-Dense hybrid model achieved the best performance, with **21% lower MSE** and **14% lower MAE** than the MLP baseline — demonstrating that sequential/temporal modeling captures yield dynamics more effectively than static feature-based approaches.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **ML/DL** | TensorFlow/Keras, Scikit-learn |
| **Statistical** | Statsmodels (ARIMA, ADF test) |
| **Visualization** | Matplotlib, Seaborn |

---

## 📁 Repository Structure

```
BondYeild_Prediction/
├── data/
│   ├── raw/
│   │   └── newfredgraph.csv          # Raw FRED economic data
│   └── processed/
│       └── normalized_bonddata.csv   # MinMaxScaler-normalized features
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb    # Data exploration & feature engineering
│   ├── 02_ARIMA.ipynb                # ARIMA time-series model
│   ├── 03_LSTM.ipynb                 # LSTM-Dense hybrid deep learning model
│   └── 04_MLP.ipynb                  # Multi-Layer Perceptron model
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Vineeth-chaitanya/USBondYield_Prediction.git
cd USBondYield_Prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
# 1. Start with 01_EDA_Preprocessing.ipynb to generate processed data
# 2. Then run any model notebook (02, 03, or 04)
```

---

## 📌 Key Takeaways

- **Federal Funds Rate** is the dominant predictor of 10-year Treasury yields (r = 0.75)
- **Temporal models outperform static models** — LSTM captures sequential dependencies that MLP misses
- **Stationarity matters** — ADF testing confirmed the need for differencing before applying ARIMA
- The project demonstrates end-to-end ML workflow: data sourcing → EDA → preprocessing → modeling → evaluation
