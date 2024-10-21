# Sales Data Analysis and Prediction

## Overview
This project focuses on analyzing and predicting sales data using machine learning. The process includes data preprocessing, exploratory data analysis (EDA), and building predictive models.

## Project Files
- **EDA.ipynb:** Notebook for exploratory data analysis.
- **Model.ipynb:** Notebook for training and evaluating machine learning models.
- **preprocess.py:** Script for data preprocessing.
- **MarketSales.xlsx:** Dataset containing sales data.

## How to Use
1. Clone the repository.
2. Install the required libraries:
    ```bash
    pip install pandas scikit-learn catboost jupyter category_encoders lightgbm
    ```
3. Open `EDA.ipynb` and `Model.ipynb` in Jupyter Notebook or JupyterLab.
4. Use `preprocess.py` to preprocess the dataset before analysis:
    ```python
    import pandas as pd
    from preprocess import preprocess

    # Load your dataset
    df = pd.read_excel('MarketSales.xlsx')

    # Apply preprocessing
    df = preprocess(df)
    ```

5. Train and evaluate the model:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score, learning_curve
    import matplotlib.pyplot as plt
    import lightgbm as lgb

    X = df[["month", "day", "PRICE", "CATEGORY_NAME1", "CITY", "CLIENTCODE"]]
    y = df["AMOUNT"]
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # CatBoost Model
    cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=0)
    cat_model.fit(X_train, y_train)
    cat_predictions = cat_model.predict(X_test)
    cat_mse = mean_squared_error(y_test, cat_predictions)
    print(f"CatBoost ile Ortalama Kare Hata: {cat_mse}")

    cross_val_scores = cross_val_score(cat_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print("Cross-Validation Ortalama Kare Hata: ", -cross_val_scores.mean())

    # Train Errors
    train_predictions = cat_model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    print(f"Eğitim Hatası (MSE): {train_mse}")

    # Test Errors
    test_mse = mean_squared_error(y_test, cat_predictions)
    print(f"Test Hatası (MSE): {test_mse}")

    # Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(cat_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim Hatası")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test Hatası")
    plt.xlabel("Eğitim Veri Boyutu")
    plt.ylabel("Hata (MSE)")
    plt.legend(loc="best")
    plt.title("Öğrenme Eğrileri")
    plt.show()

    # LightGBM Model
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X_train, y_train)
    lgb_predictions = lgb_model.predict(X_test)
    lgb_mse = mean_squared_error(y_test, lgb_predictions)
    print(f"LightGBM ile Ortalama Kare Hata: {lgb_mse}")
    ```

## Contact
For questions or suggestions, contact me at [zelihguven@gmail.com](mailto:zelihguven@gmail.com)
