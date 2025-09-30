import pandas as pd # for dealing with csv data
import numpy as np # for mathematical functions 
import matplotlib.pyplot as plt # to plot the graph 
from statsmodels.tsa.stattools import adfuller # adfuller used to check if data is stationary (p-value<0.05)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # used to identify the q part of the model 
from statsmodels.tsa.arima.model import ARIMA # import the algorithm itself 
from sklearn.metrics import mean_squared_error # computing error between actual and predicted data
from pmdarima import auto_arima #tries multiple arima combinations and picks the best one 


def load_and_clean_data(filepath):
    ''' Cleans the data and fills up null values if necesarry'''
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    flow_rate = data['Flow_Rate'].ffill()
    return flow_rate


def plot_series(series, title='Time Series', ylabel='Value'):
    '''Creates graph using mathlib'''
    plt.figure(figsize=(12, 5))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def check_stationarity(series, differenced=False):
    '''runs adf test to check whether the data is stationary'''
    result = adfuller(series)
    if not differenced:
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
    else:
        print("Differenced ADF Statistic:", result[0])
        print("Differenced p-value:", result[1])
    return result[1]  # return p-value


def plot_acf_pacf(series, lags=30):
    ''' helps in identifying patterns for ARIMA algorithm'''
    plot_acf(series, lags=lags)
    plot_pacf(series, lags=lags)
    plt.show()


def train_test_split(series, train_ratio=0.8):
    ''' returns 2 subsets - training and testing for training the model and evaluation'''
    split_idx = int(len(series) * train_ratio)
    return series[:split_idx], series[split_idx:]


def fit_auto_arima(train_series):
    '''Finds best arima model for the training data'''
    print("\nFitting auto_arima...")
    model = auto_arima(train_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    print("Best ARIMA order:", model.order)
    return model.order


def fit_arima_model(train_series, order):
    '''Fits ARIMA model to the training data'''
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model


def forecast_and_plot(train, test, model_fit):
    '''Generates predictions according to model and compares the results to the actual data'''
    forecast = model_fit.forecast(steps=len(test))
    forecast_series = pd.Series(forecast, index=test.index)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.title('ARIMA Forecast vs Actual')
    plt.xlabel('Index')
    plt.ylabel('Flow Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_series


def evaluate_forecast(test, forecast_series):
    """measure how accurate the forecasted values are
    compared to the actual test data using RMSE"""
    rmse = np.sqrt(mean_squared_error(test, forecast_series))
    print(f"Root Mean Squared Error: {rmse:.2f}")
    return rmse


def detect_leaks(test, forecast_series, rmse):
    ''' Identifies and visualizes potential leak events by 
    flagging points where the forecast error is unusually high'''
    error = abs(test - forecast_series)
    threshold = 2 * rmse
    leaks = error > threshold

    plt.figure(figsize=(12, 6))
    plt.plot(test, label='Actual')
    plt.plot(forecast_series, label='Forecast')
    plt.scatter(test[leaks].index, test[leaks], color='red', label='Potential Leak', marker='x')
    plt.title('Leak Detection Based on Forecast Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    return leaks


def main():
    filepath = 'C:/Users/vedik/Downloads/location_aware_gis_leakage_dataset.csv'
    
    # 1 - Load and preprocess data
    flow_rate = load_and_clean_data(filepath)
    plot_series(flow_rate, title='Flow Rate Over Index', ylabel='Flow Rate')

    # 2 - Check stationarity
    p_value = check_stationarity(flow_rate)
    if p_value > 0.05:
        flow_rate_diff = flow_rate.diff().dropna()
        check_stationarity(flow_rate_diff, differenced=True)
        plot_series(flow_rate_diff, title='Differenced Flow Rate')
        plot_acf_pacf(flow_rate_diff)
    else:
        flow_rate_diff = flow_rate

    # 3 - Split data
    train, test = train_test_split(flow_rate)

    # 4 - Fit Auto ARIMA
    best_order = fit_auto_arima(train)

    # 5 - Fit ARIMA Model
    model_fit = fit_arima_model(train, best_order)

    # 6 -  Forecast
    forecast_series = forecast_and_plot(train, test, model_fit)

    # 7 - Evaluate
    rmse = evaluate_forecast(test, forecast_series)

    # 8 - Leak Detection
    leaks = detect_leaks(test, forecast_series, rmse)

if __name__ == "__main__":
    main()
