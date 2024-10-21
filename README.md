# Sales Data Analysis and Prediction

## Overview
This project focuses on analyzing and predicting sales data using machine learning. The process includes data preprocessing, exploratory data analysis (EDA), and building predictive models.

## Project Files
- **EDA.ipynb:** Notebook for exploratory data analysis.
- **Model.ipynb:** Notebook for training and evaluating machine learning models.
- **MarketSales.xlsx:** Dataset containing sales data. [Download from Kaggle](https://www.kaggle.com/datasets/sezginildes/marketsales)

## Libraries used
- Pandas
- NumPy
- Seaborn
- Matplotlib
- datetime
- missingno
- warnings
- joblib
- category_encoders
- sklearn
  - LabelEncoder
  - MinMaxScaler
  - train_test_split
  - mean_squared_error
  - GridSearchCV
  - RandomForestRegressor
  - cross_val_score
  - learning_curve
- statsmodels
  - ARIMA
- xgboost
- catboost
- lightgbm

## How to Use
1. Clone the repository.
2. Install the required libraries:
    ```bash
    pip install pandas scikit-learn catboost jupyter category_encoders lightgbm
    ```
3. Open `EDA.ipynb` and `Model.ipynb` in Jupyter Notebook or JupyterLab.
4. If you want to see my analysis, open the `EDA.ipynb` file.
5. If you want to see the finalized version with a clean look after moving the analyses from the EDA file, review the `Model.ipynb` file.

## Results
- **CatBoost Model MSE:** 0.4378510756095068
- **LightGBM Model MSE:** 0.5672705699379459

## Contact
For questions or suggestions, contact me at [zelihguven@gmail.com](mailto:zelihguven@gmail.com)

