import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import itertools
import sys
from datetime import datetime
import warnings
import argparse
import statsmodels.api as sm
warnings.filterwarnings('ignore')



# outliers values for small dataset (1, 1)
# outliers values for large dataset (0.1, 1.5)

# '2023-01-01':'2024-01-31'
# 2022-07-04 : 2023-06-28

def remove_outliers(dataframe, threshold):
    z_scores = np.abs(stats.zscore(dataframe['Debit']))
    dataframe = dataframe[(z_scores <= threshold)]
    return dataframe

def fillna_with_nearest_mean(series):
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            left_index = i - 1
            right_index = i + 1

            while left_index >= 0 and pd.isna(series.iloc[left_index]):
                left_index -= 1

            while right_index < len(series) and pd.isna(series.iloc[right_index]):
                right_index += 1

            while left_index >= 0 and right_index < len(series) and pd.isna(series.iloc[left_index]) and pd.isna(series.iloc[right_index]):
                left_index -= 1
                right_index += 1

            if left_index >= 0 and right_index < len(series):
                series.iloc[i] = (series.iloc[left_index] + series.iloc[right_index]) / 2
            elif left_index >= 0:
                series.iloc[i] = series.iloc[left_index]
            elif right_index < len(series):
                series.iloc[i] = series.iloc[right_index]

    return series

def grid_search_sarima_params(data):
    # Define ranges for parameters
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)

    # Create a list of all possible combinations of parameters
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, [12]))

    # Perform grid search
    best_aic = float('inf')
    best_params = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            model = SARIMAX(data, order=param, seasonal_order=param_seasonal)
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_params = (param, param_seasonal)

    return best_params

def main(dataframe, outlier_value1, outlier_value2, start_date, end_date):
    register_matplotlib_converters()

    df = pd.read_csv(dataframe) 
    df = df.rename(columns={'Transaction Date': 'date'})
    df = df[['date', 'Debit']]

    # Pre-processing
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.set_index('date')
    df = df.sort_index()  # Sort the DataFrame by date
    df['Debit'] = df['Debit'].replace('[\$,]', '', regex=True).astype(float)

    df_no_outliers = remove_outliers(df, outlier_value1)

    filtered_data = df_no_outliers[start_date:end_date]
    daily_data = filtered_data.resample('D').sum()
    daily_data.index = pd.DatetimeIndex(daily_data.index, freq='D')

    daily_no_outliers = remove_outliers(daily_data,outlier_value2)

    filtered_data = daily_no_outliers[start_date : end_date]
    daily_no_outliers = filtered_data.resample('D').sum()
    daily_no_outliers.index = pd.DatetimeIndex(daily_no_outliers.index, freq='D')

    daily_no_outliers['Debit'] = daily_no_outliers['Debit'].replace(0.00, np.nan)
    daily_no_outliers['Debit'] = fillna_with_nearest_mean(daily_no_outliers['Debit'])

    pred = None
    test = None

    # Redirect stdout to a file
    with open('output.txt', 'w') as f:
        sys.stdout = f  # Redirect stdout to the file
        sys.stderr = f
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        print("Current Date and Time:", current_time)
        print("----------------------------------------------------------------------------------------")
        print("")
        print("")
        best_params = grid_search_sarima_params(daily_no_outliers)
        # print('Best Parameters:', best_params)
        print("")
        print("")
        sarima_order, seasonal_order = best_params

        sarima_model = SARIMAX(daily_no_outliers, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
        sarima_results = sarima_model.fit()

        train = daily_no_outliers.iloc[:-30]
        test = daily_no_outliers.iloc[-30:]

        start = len(train)
        end = len(train) + len(test) - 1
        pred = sarima_results.predict(start=start, end=end, type='levels')

        pred.index = daily_no_outliers.index[start:end + 1]

        mae = mean_absolute_error(test, pred)
        mse = mean_squared_error(test, pred)
        rmse = np.sqrt(mse)
        percentage_accuracy = 100 * (1 - (mae / test.mean()))
        # Print the accuracy and predictions
        print('Percentage Accuracy:', percentage_accuracy, '%')
        print("")
        print("")
        print("MAE : ", mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('Predictions:')
        print("")
        print(pred)
        print("")
        print(dataframe)
        print(outlier_value1)
        print(outlier_value2)

        # Restore stdout to the default (console)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    plt.figure(figsize=(15, 6))
    plt.plot(test, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.legend()
    plt.savefig('forecast_plot.png')  # Save the plot to a file
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data and perform SARIMA forecasting.')
    parser.add_argument('dataframe', type=str, help='File path of the primary dataset')
    parser.add_argument('outlier_value1', type=float, help='Outlier value for small dataset')
    parser.add_argument('outlier_value2', type=float, help='Outlier value for large dataset')
    parser.add_argument('start_date', type=str, help='Outlier value for large dataset')
    parser.add_argument('end_date', type=str, help='Outlier value for large dataset')
    args = parser.parse_args()
    
    main(args.dataframe, args.outlier_value1, args.outlier_value2, args.start_date, args.end_date)


# python forecast.py secondary_df.csv 1 1 2022-07-04 2023-06-28