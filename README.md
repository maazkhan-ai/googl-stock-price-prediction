# GOOGL Stock Price Prediction

Welcome to the **GOOGL Stock Price Prediction** project! This repository contains a Jupyter notebook (`stock_price_prediction.ipynb`) that builds a machine learning pipeline to predict the next-day closing price of Alphabet Inc. (GOOGL) stock using historical data from 2020 to 2024. With a focus on accuracy and reproducibility, the pipeline leverages technical indicators and achieves a directional accuracy of **65.3%** with **79.1% precision** for upward price movements. Whether you're a data scientist, financial analyst, or developer, this project offers a robust foundation for stock price forecasting.

## Project Overview

This project develops a machine learning model to predict the next-day closing price of GOOGL stock and its directional movement (up or down) using historical price and volume data. The pipeline, implemented in Python, includes data preprocessing, feature engineering, model selection, evaluation, and artifact storage. The best model, **Lasso regression**, delivers reliable predictions, with key features like Open price, Close price, and 50-day SMA driving results. Artifacts are saved for easy deployment in applications like Streamlit or APIs.

## Research Objective

The goal is to create a data-driven model to accurately predict GOOGL's next-day closing price using historical data from January 2020 to December 2024. By incorporating technical indicators and lagged price features, the project aims to capture market trends and support financial analysis or trading strategies, ensuring reproducibility through saved model artifacts.

## Methodology

The pipeline follows a structured approach:

1. **Data Preprocessing**:
   - Loads `googl_data_2020_2025.csv` (1,258 daily records).
   - Handles irregular headers, converts data types, removes missing values, and sorts by date.
2. **Feature Engineering**:
   - Generates technical indicators (e.g., SMA_7, SMA_50, RSI14, MACD, Bollinger Bands).
   - Adds lagged features (e.g., Close_lag_1, Close_lag_10) and metrics like percentage change and log returns.
3. **Model Selection**:
   - Evaluates models (Linear Regression, Ridge, Lasso, Random Forest, XGBoost, SVR) using TimeSeriesSplit (5 folds).
   - Tunes hyperparameters with GridSearchCV or RandomizedSearchCV.
4. **Evaluation**:
   - Tests the best model (Lasso) on 2024 data using MAE, RMSE, R², and directional metrics (accuracy, precision, recall).
5. **Feature Importance**:
   - Analyzes model coefficients to identify key predictors.
6. **Artifacts**:
   - Saves the trained model, feature list, and test splits in `artifacts_stock_research/`.

## Key Findings

- **Best Model**: Lasso regression outperformed other models based on cross-validation metrics (MAE, RMSE, R²).
- **Performance**: Achieved **65.3% directional accuracy**, **79.1% precision**, and **60.3% recall** for upward price movements on the 2024 test set.
- **Key Features**:
  - **Open price** (coef: 20.76)
  - **Close price** (coef: 5.74)
  - **SMA_50** (coef: 1.55)
- **Insights**: EDA revealed an upward price trend with periods of volatility; technical indicators like RSI and Bollinger Bands highlighted overbought/oversold conditions.
- **Limitations**: The model misses external factors (e.g., news, macroeconomic events), limiting its ability to capture all price movements.

## Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook
- Required libraries (see `requirements.txt` or install manually)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/maazkhan-ai/googl-stock-price-prediction.git
   cd googl-stock-price-prediction
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install ta scikit-learn xgboost joblib matplotlib pandas numpy seaborn statsmodels yfinance nbformat
   ```
3. **Prepare Data**:
   - Place `googl_data_2020_2025.csv` in the project root. The file should contain daily GOOGL data (Date, Open, High, Low, Close, Adj Close, Volume).
4. **Run the Notebook**:
   ```bash
   jupyter notebook
   ```
   Open `stock_price_prediction.ipynb` and execute cells sequentially.

## Repository Structure

```
googl-stock-price-prediction/
├── stock_price_prediction.ipynb  # Main notebook with the pipeline
├── googl_data_2020_2025.csv     # Historical data (user-provided)
├── artifacts_stock_research/     # Saved model and data
│   ├── best_model_Lasso.joblib   # Trained Lasso model
│   ├── feature_list.csv         # Feature list
│   ├── X_test.csv               # Test features
│   └── y_test.csv               # Test target
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Usage

1. **Run the Notebook**:
   - Execute `stock_price_prediction.ipynb` to preprocess data, train the model, and visualize results (e.g., confusion matrix, feature importance).
2. **Deploy the Model**:
   - Load the saved model for predictions:
     ```python
     import joblib
     model = joblib.load('artifacts_stock_research/best_model_Lasso.joblib')
     ```
   - Use in Streamlit, APIs, or other applications.
3. **Customize**:
   - Modify features, models, or hyperparameters in the notebook to experiment with new configurations.

## Next Steps / Future Work

- **Enhance Data**: Incorporate news sentiment, earnings reports, or macroeconomic indicators.
- **Advanced Models**: Experiment with deep learning (e.g., LSTM, GRU) for temporal modeling.
- **New Features**: Add indicators like Stochastic Oscillator or ATR.
- **Real-Time Deployment**: Build a Streamlit dashboard or API for live predictions.
- **Robust Backtesting**: Include transaction costs and slippage for realistic simulations.
- **Cross-Asset Testing**: Apply the pipeline to other stocks or asset classes.

## Limitations

- Does not account for external factors (e.g., news, economic events).
- Moderate directional accuracy (65.3%) and recall (60.3%) indicate room for improvement.
- Trained on historical data; performance may vary in future market conditions.

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please follow PEP 8 guidelines and include tests where applicable.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, open an issue on GitHub or email [maazkhan-ai@gmail.com].

---

 **Star this repository** if you find it useful! Happy predicting! 
