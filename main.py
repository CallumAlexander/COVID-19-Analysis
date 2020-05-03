#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- This file is Property of Callum 'Cal' Alexander -*-
# -*- CONTACT INFORMATION -*-
# -*- Email : s1931801@ed.ac.uk -*-
# -*- Instagram : cal.zander -*-
# -*- Twitter : calzander -*-
# -*- GitHub : www.github.com/CallumAlexander -*-

"""
Created on Thu Mar 26 17:58:00 2020
@author: cal
"""

"""
THEORY behind all this code:

My understanding is that if you can track and predict the change in number
of daily cases, i.e. the 2nd order of the number of confirmed cases, then 
you can estimate the trajectory of spread of the virus.
"""

from forecasting import *
from utils import *

# Read data from source
caseData = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com' \
           '%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series' \
           '%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv '
deathData = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com' \
            '%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series' \
            '%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv '

choice = input('>>>>>  ')
placeholder = ''
if choice == 'c':
    df = pd.read_csv(caseData)
    placeholder = 'Confirmed cases'
else:
    df = pd.read_csv(deathData)
    placeholder = 'Deaths'

# Preprocessing and gathering data
italy_X, italy_y = preprocess(df, 'Italy')
spain_X, spain_y = preprocess(df, 'Spain')
uk_X, uk_y = preprocess(df, 'United Kingdom')
germany_X, germany_y = preprocess(df, 'Germany')
france_X, france_y = preprocess(df, 'France')
us_X, us_y = preprocess(df, 'US')


# Gathering derivatives
uk_dailyY = delta(uk_y)
uk_deltaDaily = delta(uk_dailyY)

italy_daily = delta(italy_y)
italy_deltaDaily = delta(italy_daily)

spain_daily = delta(spain_y)
spain_deltaDaily = delta(spain_daily)

germany_daily = delta(germany_y)
germany_deltaDaily = delta(germany_daily)

france_daily = delta(france_y)
france_deltaDaily = delta(france_daily)

us_daily = delta(us_y)
us_deltaDaily = delta(us_daily)

# --------------------------------
# Plotting data
fig, ax = plt.subplots(nrows=3, ncols=2)

# Plotting
ax[0, 0].plot(uk_X, uk_y, label='UK')
ax[0, 0].plot(italy_X, italy_y, label='Italy')
ax[0, 0].plot(spain_X, spain_y, label='Spain')
ax[0, 0].plot(germany_X, germany_y, label='Germany')
ax[0, 0].plot(france_X, france_y, label='France')
ax[0, 0].plot(us_X, us_y, label='United States')
ax[0, 0].legend(fontsize='x-small')
ax[0, 0].grid()
ax[0, 0].set(xlabel='Number of days', ylabel=placeholder,
             title=placeholder)

# Plotting daily
ax[0, 1].plot(uk_X, uk_dailyY, label='UK - daily ' + placeholder)
ax[0, 1].plot(italy_X, italy_daily, label='Italy - daily ' + placeholder)
ax[0, 1].plot(spain_X, spain_daily, label='Spain - daily ' + placeholder)
ax[0, 1].plot(germany_X, germany_daily, label='Germany - daily ' + placeholder)
# ax[0, 1].plot(france_X, france_daily, label='France - daily ' + placeholder)
ax[0, 1].legend(fontsize='x-small')
ax[0, 1].grid()
ax[0, 1].set(xlabel='Number of days', ylabel='Number of daily ' + placeholder,
             title='Daily ' + placeholder)

# Plotting differences in daily cases
ax[1, 0].plot(uk_X, uk_deltaDaily, label='UK - rate of change of daily ' + placeholder)
ax[1, 0].plot(italy_X, italy_deltaDaily, label='Italy - rate of change of daily ' + placeholder)
ax[1, 0].plot(spain_X, spain_deltaDaily, label='Spain - rate of change of daily ' + placeholder)
ax[1, 0].plot(germany_X, germany_deltaDaily, label='Germany - rate of change of daily ' + placeholder)
ax[1, 0].plot(france_X, france_deltaDaily, label='France - rate of change of daily ' + placeholder)
ax[1, 0].legend(fontsize='xx-small')
ax[1, 0].grid()
ax[1, 0].set(xlabel='Number of days', ylabel='Change in daily ' + placeholder,
             title='Change in daily ' + placeholder)

# Predicting the future differences in daily cases for UK
# -------------------------------------------------------------
uk_deltaDaily_bestFit, uk_coef = regression(uk_X, uk_deltaDaily)
italy_deltaDaily_bestFit, italy_coef = regression(italy_X, italy_deltaDaily)
spain_deltaDaily_bestFit, spain_coef = regression(spain_X, spain_deltaDaily)
france_deltaDaily_bestFit, france_coef = regression(france_X, france_deltaDaily)
germany_deltaDaily_bestFit, germany_coef = regression(germany_X, germany_deltaDaily)

# Plotting current trajectories
ax[2, 1].plot(uk_X, uk_deltaDaily_bestFit, label='UK')
ax[2, 1].plot(italy_X, italy_deltaDaily_bestFit, label='Italy')
ax[2, 1].plot(spain_X, spain_deltaDaily_bestFit, label='Spain')
ax[2, 1].plot(france_X, france_deltaDaily_bestFit, label='France')
ax[2, 1].plot(germany_X, germany_deltaDaily_bestFit, label='Germany')
ax[2, 1].grid()
ax[2, 1].legend()
ax[2, 1].set(title='Current Trajectories for changes in daily ' + placeholder)


def plotPredictions(X, y, pred, country):
    movingAverage10, X_mean10 = movingAverage(y, 7)
    movingAverage20, X_mean20 = movingAverage(y, 7)

    ax[1, 1].bar(X, y, label=country)
    # ax[1, 1].plot(X, pred, 'r', label='Predicted trajectory')
    ax[1, 1].plot(X_mean10, movingAverage10, 'g', label='1 week moving average')
    ax[1, 1].plot(X_mean20, movingAverage20, 'r', label='2 week moving average')
    ax[1, 1].set(xlabel='Number of days', ylabel='Change in daily ' + placeholder,
                 title='Predicted trajectory for ' + country)
    ax[1, 1].legend(fontsize='x-small')
    ax[1, 1].grid()


plotPredictions(spain_X, spain_daily, spain_deltaDaily_bestFit, 'Spain')
printBestFitCoef(uk_coef, italy_coef, spain_coef, france_coef, germany_coef)

# Calculating and plotting the adjusted confirmed cases
# -----------------------------------------------------------

# Working out when all the data intersects, in this case they all intersect at 2 cases
A = np.array([italy_y, spain_y, germany_y, france_y])
print([np.intersect1d(uk_y, A_i) for A_i in A])

# Computing the indices where the cases = 2
uk_y_adj_index = np.argwhere(uk_y == 2)[0].item()
italy_y_adj_index = np.argwhere(italy_y == 2)[0].item()
spain_y_adj_index = np.argwhere(spain_y == 2)[0].item()
france_y_adj_index = np.argwhere(france_y == 2)[0].item()

# Adjusting the arrays to start when the cases = 2
uk_y_adj = uk_y[uk_y_adj_index:]
italy_y_adj = italy_y[italy_y_adj_index:]
spain_y_adj = spain_y[spain_y_adj_index:]
france_y_adj = france_y[france_y_adj_index:]

# Computing appropriate X values for each set of data
X_uk_adj = np.arange(0, len(uk_y_adj))
X_italy_adj = np.arange(0, len(italy_y_adj))
X_spain_adj = np.arange(0, len(spain_y_adj))
X_france_adj = np.arange(0, len(france_y_adj))

# Plotting the data
'''ax[2, 0].plot(X_uk_adj, uk_y_adj, label='UK')
ax[2, 0].plot(X_italy_adj, italy_y_adj, label='Italy')
ax[2, 0].plot(X_spain_adj, spain_y_adj, label='Spain')
ax[2, 0].plot(X_france_adj, france_y_adj, label='France')
ax[2, 0].legend()
ax[2, 0].grid()
ax[2, 0].set(xlabel='Number of days since 2nd ' + placeholder, ylabel=placeholder,
             title=placeholder + ' (adjusted)')
'''


# Forecasting Uk deaths
def makeForecast(X, y, future_val, label):
    global placeholder
    start = len(X) - 1
    distance = future_val
    pdq, seasonal_pdq, aic = findOptimumForecastParam(y)
    results = forecast(y, pdq, seasonal_pdq)
    y_forecast = results.get_forecast(steps=distance)
    X_forecast = np.arange(start, start + distance)

    print('AIC: ' + str(aic))

    ax[2, 0].plot(X, y, label=label + placeholder)
    ax[2, 0].plot(X_forecast, y_forecast.predicted_mean, label='Forecast: ' + placeholder)
    ax[2, 0].grid()
    ax[2, 0].legend()
    ax[2, 0].set(xlabel=label + placeholder, ylabel=placeholder,
                 title=placeholder + ' (forecast)')


makeForecast(uk_X, uk_dailyY, 45, 'United Kingdom')
