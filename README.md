# Predictive Energy Grid Optimizer  
### Time-Series Forecasting & Market Regime Analysis

## Overview
Electricity demand must be balanced with generation in real time, making short-term load forecasting an important problem in power systems.

This project implements an end-to-end machine learning pipeline to predict short-term electrical grid load using historical time-series data. The main objective is to compare a well-tuned classical machine learning model (Decision Tree Regressor) with a deep learning sequence model (Long Short-Term Memory network) and evaluate whether increased model complexity improves forecasting performance on structured temporal data.

---

## Project Goals
- Build a clean, leakage-free time-series forecasting pipeline  
- Engineer meaningful temporal and autoregressive features  
- Identify hidden market regimes using unsupervised learning  
- Compare classical ML and deep learning approaches  
- Analyze the trade-off between model complexity and performance  

---

## Key Results
- **Feature engineering had a significant impact** on predictive performance.
- Using strict chronological validation, a **tuned Decision Tree outperformed a baseline LSTM model** on this dataset.
- For structured, medium-scale tabular time-series data, additional model complexity did not necessarily improve results.

Evaluation metrics:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

---

## Project Architecture

### Phase 1: Feature Engineering & Preprocessing

**Cyclical Time Encoding**  
Converted timestamps into sine and cosine transformations to preserve cyclical relationships (e.g., 23:00 and 00:00 are close in time).

**Autoregressive Lag Features**  
Created 1-hour and 24-hour lag features to capture short-term and daily momentum patterns.

**Data Cleaning**  
Handled missing values using linear interpolation to maintain temporal consistency.

---

### Phase 2: Unsupervised Market Regime Detection

**Dimensionality Reduction (PCA)**  
Reduced 8-dimensional temporal and pricing features to 3 principal components to remove multicollinearity and simplify clustering.

**Clustering (K-Means)**  
Clustered observations into 4 distinct market regimes using the Elbow Method to determine the optimal number of clusters.

> Note: In a real-time production system, automated metrics (e.g., Calinski-Harabasz Index) could be used to dynamically re-evaluate the optimal number of clusters.

---

### Phase 3: Supervised Learning — Classical Baseline

**Chronological Validation**  
Performed an 80/20 chronological train-test split to eliminate lookahead bias and prevent data leakage.

**Hyperparameter Optimization**  
Systematically tuned `max_depth` to analyze the bias-variance tradeoff and select the optimal model complexity.

**Model**
- Decision Tree Regressor  
- Evaluated using MAE and RMSE  

---

### Phase 4: Deep Learning Model (LSTM)

**Data Reshaping**  
Implemented a sliding window approach to transform tabular data into 3D tensors:

`(Samples, Time Steps, Features)`

**Feature Scaling**  
Applied `StandardScaler` normalization to stabilize training and improve convergence.

**Architecture**
- LSTM layer  
- Dropout layer  
- Dense output layer  
- Optimizer: Adam  

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
- Time-Series Feature Engineering  
- Principal Component Analysis (PCA)  
- K-Means Clustering  
- Decision Tree Regression  
- LSTM Neural Networks  
- Chronological Cross-Validation  

---

## What I Learned

- Proper chronological validation is critical in time-series modeling.
- Feature engineering often has a larger impact than increasing model complexity.
- Deep learning is not always superior for structured tabular datasets.
- Simpler models can be more efficient, interpretable, and competitive when carefully tuned.

---

## Future Improvements

- Implement walk-forward cross-validation  
- Compare against additional baselines (e.g., Gradient Boosting, Random Forest)  
- Perform automated regime selection for adaptive clustering  
- Explore probabilistic forecasting approaches  

---

**Author:** Mykhailo Isupov  
