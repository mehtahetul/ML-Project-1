# ML Project 1

Two supervised learning projects built with Python, pandas and scikit-learn.

## Projects

### ✈️ Domestic Flight Price Prediction
Predict the ticket `price` of a domestic flight from airline, route, timings, stops, class and days left before departure.

- **Type:** Regression (scored on RMSE)
- **Approach:** EDA, feature engineering and imputation, then linear models (Linear, Ridge, Lasso) and tree models (Decision Tree, Random Forest, XGBoost, LightGBM), all compared with the same 5-fold cross-validation
- **Final model:** average of the strongest tree-based models
- **Notebook:** `Domestic Flight Prediction/flight_price_prediction_v7 -- final.ipynb`

### 🚗 Vehicle Classification
Classify a vehicle as **Sedan (0)**, **SUV (1)** or **Sports Car (2)** from 20 numeric measurements (engine, size, performance).

- **Type:** Multi-class classification (scored on accuracy)
- **Challenge:** the test data has extra measurement noise, so models are checked for robustness
- **Approach:** Logistic Regression baseline, Decision Tree and XGBoost
- **Notebooks:** `Vehicle Class Classification/*.ipynb`

## Report

The full write-up for both projects is in [`Report (Regression + Classification).pdf`](<Report (Regression + Classification).pdf>).

## Repository structure

```
ML-Project-1/
├── README.md
├── Report (Regression + Classification).pdf
├── Domestic Flight Prediction/
│   ├── train.csv, test.csv, sample_submission.csv
│   ├── flight_price_prediction_v7 -- final.ipynb
│   └── submission--v7.csv
└── Vehicle Class Classification/
    ├── vehicle_train_10000.csv, vehicle_test.csv, vehicle_sample_submission.csv
    ├── Vehicle_Classification_LogisticRegression_Yashik.ipynb
    ├── Girish_DecisionTree_final.ipynb
    └── *_submission.csv
```

## Getting started

```bash
git clone https://github.com/mehtahetul/ML-Project-1.git
cd ML-Project-1
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
jupyter notebook
```

Open a notebook and run all cells. The data files are already in each project folder.
