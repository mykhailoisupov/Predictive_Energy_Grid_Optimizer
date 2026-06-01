# Predictive Energy Grid Optimizer  
### Time-Series Forecasting & Market Regime Analysis

## Overview
Electricity demand must be balanced with generation in real time, making short-term load forecasting an important problem in power systems.

This project implements an end-to-end machine learning pipeline to predict short-term electrical grid load using historical time-series data. The main objective is to compare well-tuned classical machine learning models (Decision Tree and Random Forest) with a deep learning sequence model (stacked LSTM network) and evaluate whether increased model complexity improves forecasting performance on structured temporal data.

---

## Project Goals
- Build a clean, leakage-free time-series forecasting pipeline  
- Engineer meaningful temporal, autoregressive, and rolling statistical features  
- Identify hidden market regimes using unsupervised learning and interpret their characteristics  
- Compare classical ML (Decision Tree, Random Forest) and deep learning (LSTM) approaches  
- Evaluate models using walk-forward cross-validation for robust results  
- Analyze per-regime performance to understand where models succeed or struggle  

---

## Key Results
- **Feature engineering had the largest impact** on predictive performance — day-of-week encoding, weekly lag features, and rolling statistics were particularly valuable.
- Using strict chronological validation and walk-forward CV, a **tuned Random Forest outperformed both the Decision Tree and LSTM** on this dataset.
- For structured, medium-scale tabular time-series data, additional model complexity (LSTM) did not necessarily improve results.
- Proper methodology (fixing scaler leakage, walk-forward validation) is critical for trustworthy results.

Evaluation metrics:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)

---

## Project Architecture

### Phase 1: Feature Engineering & Preprocessing

**Cyclical Time Encoding**  
Converted timestamps into sine and cosine transformations for hour, month, and day-of-week to preserve cyclical relationships (e.g., 23:00 and 00:00 are close in time; Friday and Monday are close in the work week).

**Autoregressive Lag Features**  
Created 1-hour, 24-hour (daily), and 168-hour (weekly) lag features to capture short-term momentum, daily cycles, and weekly seasonality patterns.

**Rolling Statistics**  
Added 24-hour rolling mean and standard deviation of load to capture recent trend and volatility.

**Data Cleaning**  
Analyzed missing data gap structure before applying linear interpolation to maintain temporal consistency.

---

### Phase 2: Unsupervised Market Regime Detection

**Dimensionality Reduction (PCA)**  
Reduced high-dimensional temporal and pricing features to 3 principal components with variance explained reporting. Used `numpy.linalg.eigh` (numerically stable for symmetric covariance matrices) instead of `eig`.

**Clustering (K-Means)**  
Clustered observations into 4 distinct market regimes using both the Elbow Method and Silhouette Score analysis to determine the optimal number of clusters.

**Regime Interpretation**  
Each regime is characterized by its average load, average price, dominant hours, dominant months, and weekend share — providing actionable insight into grid operating states.

---

### Phase 3: Supervised Learning — Classical Baselines

**Walk-Forward Cross-Validation**  
Used `TimeSeriesSplit` with 5 folds for hyperparameter tuning, providing mean ± std performance estimates instead of single-split results.

**Decision Tree Regressor**  
Hyperparameter search over `max_depth`, `min_samples_split`, and `min_samples_leaf` using walk-forward CV.

**Random Forest (Ensemble)**  
An ensemble of 200 decision trees validated with the same walk-forward procedure. Reduces overfitting and typically provides state-of-the-art performance on tabular data.

---

### Phase 4: Deep Learning Model (Stacked LSTM)

**Proper Scaling Discipline**  
`StandardScaler` is fit on training data only, then applied to both train and test sets — preventing test set statistics from leaking into training.

**Data Reshaping**  
Implemented a sliding window approach with bridge sequences to transform tabular data into 3D tensors while maintaining consistent test periods across all models.

**Architecture**
- Stacked LSTM (128 → 64 units)
- Dropout layers for regularization
- Dense hidden layer with ReLU activation
- Dense output layer
- EarlyStopping and ReduceLROnPlateau callbacks

---

### Phase 5: Visualization & Analysis

**Comprehensive Metrics**  
MAE, RMSE, MAPE, and R² reported for all three models.

**Visualizations**
- Time-series overlay with date-based x-axis
- Error distribution histograms
- Performance scorecard bar charts
- Actual vs. Predicted scatter plots
- Per-regime MAE breakdown

---

## Tech Stack

**Language**
- Python  

**Libraries**
- Scikit-Learn  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

**Techniques**
- Time-Series Feature Engineering (cyclical encoding, lags, rolling stats)
- Principal Component Analysis (PCA) — manual implementation
- K-Means Clustering with Silhouette Validation
- Decision Tree Regression
- Random Forest Regression
- Stacked LSTM Neural Networks with EarlyStopping
- Walk-Forward Cross-Validation (`TimeSeriesSplit`)

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API credentials (for data download)
# Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars

# Run the notebook
jupyter notebook Model.ipynb
```

---

## What I Learned

- Proper chronological validation and walk-forward CV are critical in time-series modeling.
- Feature engineering (especially day-of-week and weekly lags) often has a larger impact than increasing model complexity.
- Deep learning is not always superior for structured tabular datasets.
- **Methodology matters:** fixing scaler data leakage, using consistent train/test splits, and proper hyperparameter search all contribute to trustworthy results.
- Ensemble methods (Random Forest) are a strong default for tabular data.
- Unsupervised regime detection provides interpretable structure, but its value for downstream prediction should be validated per-regime.

---

## Future Improvements

- Compare against Gradient Boosted Trees (XGBoost, LightGBM)
- Implement full walk-forward evaluation for the LSTM
- Explore Transformer-based sequence models (e.g., Temporal Fusion Transformer)
- Perform automated regime selection for adaptive clustering
- Explore probabilistic forecasting approaches (prediction intervals)
- Experiment with different `TIME_STEPS` lookback windows for LSTM

---

**Author:** Mykhailo Isupov  
