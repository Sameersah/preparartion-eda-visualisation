# Data Analysis Assignment: Tabular and Time Series Data Analysis

This repository contains a comprehensive analysis of both tabular and time series datasets using various machine learning techniques. The analysis is structured into three main parts for each dataset type.


## Datasets

### 1. Tabular Dataset: Credit Card Fraud Detection
- **Source**: Kaggle ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Features**: `Time`, `Amount`, `V1-V28` (PCA transformed features)
- **Target**: `Class` (fraud or not)

### 2. Time Series Dataset: Store Sales Data
- **Source**: Kaggle ([rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting))
- **Features**: Order details, customer information, product categories
- **Target**: Sales

## Analysis Components

For each dataset, the following analyses were performed:

### Part A: EDA and Data Preprocessing
- Detailed exploratory data analysis
- Data cleaning and preprocessing
- Feature engineering
- Visualization of patterns and distributions

### Part B: Clustering and Anomaly Detection
- K-means clustering
- Isolation Forest for anomaly detection
- Pattern analysis
- Visualization of clusters and anomalies

### Part C: Machine Learning Models
- AutoML implementation
- Ensemble model building
- Model evaluation and comparison
- Feature importance analysis

## Technologies Used

- **Python 3.10**
- **Key Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `lazypredict`
  - `xgboost`
  - `lightgbm`

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
## Download datasets using kagglehub
import kagglehub

     # Credit Card Fraud Dataset
     path_credit = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

     # Sales Dataset
     path_sales = kagglehub.dataset_download("rohitsahoo/sales-forecasting")


