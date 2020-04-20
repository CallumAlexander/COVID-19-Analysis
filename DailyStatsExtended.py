from main import *


def plotAllDaily():
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.suptitle('Daily figures (Extended) - '+ placeholder, fontsize=16)

    plotDailyExtended(uk_X, uk_dailyY, 14, 'United Kingdom', fig, ax, 0, 0)
    plotDailyExtended(france_X, france_daily, 14, 'France', fig, ax, 0, 1)
    plotDailyExtended(spain_X, spain_daily, 14, 'Spain', fig, ax, 1, 0)
    plotDailyExtended(italy_X, italy_daily, 14, 'Italy', fig, ax, 1, 1)
    plotDailyExtended(germany_X, germany_daily, 14, 'Germany', fig, ax, 2, 0)


def plotDailyExtended(X, y, future, label, figure, axis, axX, axY):
    # Calculating moving averages
    movingAverage7, X_mean7 = movingAverage(y, 7)
    movingAverage14, X_mean14 = movingAverage(y, 14)
    movingAverage21, X_mean21 = movingAverage(y, 21)
    movingAverage30, X_mean30 = movingAverage(y, 30)

    # Calculating the forecast for moving averages
    # X_7_forecast, y_7_forecast = maForecast(X_mean7, movingAverage7, future, 7)
    # X_14_forecast, y_14_forecast = maForecast(X_mean14, movingAverage14, future, 14)
    # X_30_forecast, y_30_forecast = maForecast(X_mean30, movingAverage30, future, 30)

    axis[axX, axY].bar(X, y, label=label)
    axis[axX, axY].plot(X_mean7, movingAverage7, 'g', label='1 week moving average')
    axis[axX, axY].plot(X_mean14, movingAverage14, 'orange', label='2 week moving average')
    axis[axX, axY].plot(X_mean21, movingAverage21, 'red', label='3 week moving average')
    axis[axX, axY].plot(X_mean30, movingAverage30, 'purple', label='1 month moving average')
    # axis[axX, axY].plot(X_7_forecast, y_7_forecast.predicted_mean, label='Forecast: 1 week moving average')
    # axis[axX, axY].plot(X_14_forecast, y_14_forecast.predicted_mean, label='Forecast: 2 week moving average')
    # axis[axX, axY].plot(X_30_forecast, y_30_forecast.predicted_mean, label='Forecast: 1 month moving average')
    axis[axX, axY].grid()
    axis[axX, axY].legend()
    axis[axX, axY].set(title=label)


def maForecast(X, y, future, shift):
    global placeholder
    start = len(X) - 1 + shift
    distance = future
    pdq, seasonal_pdq, aic = findOptimumForecastParam(y)
    results = forecast(y, pdq, seasonal_pdq)
    y_forecast = results.get_forecast(steps=distance)
    X_forecast = np.arange(start, start + distance)
    return X_forecast, y_forecast


if __name__ == '__main__':
    plotAllDaily()
