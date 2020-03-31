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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data from source
data = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
df = pd.read_csv(data)

#Data preprocessing function
def preprocess(dataset, country):
    isCountry = (df['Country/Region'] == country)
    countryData = df[isCountry]
    isCountry = countryData['Province/State'].isna()
    countryData = countryData[isCountry]
    
    y = countryData.iloc[:, 4:].values
    y = np.transpose(y)
    y = y.flatten()
    X = np.arange(len(y))
    return X, y

def delta(cases):
    y = [t - s for s, t in zip(cases, cases[1:])]
    y = [0] + y
    y = np.asarray(y, dtype=np.int64)
    return y

    

    
#Preprocessing data
italy_X, italy_y = preprocess(df, 'Italy')
spain_X, spain_y = preprocess(df, 'Spain')
uk_X, uk_y = preprocess(df, 'United Kingdom')
germany_X, germany_y = preprocess(df, 'Germany')
france_X, france_y = preprocess(df, 'France')

#Gathering derivatives
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



#Plotting data
fig, ax = plt.subplots(nrows=2, ncols=2)

# Plotting confirmed cases
ax[0,0].plot(uk_X, uk_y, label='UK')
ax[0,0].plot(italy_X, italy_y, label='Italy')
ax[0,0].plot(spain_X, spain_y, label='Spain')
ax[0,0].plot(germany_X, germany_y, label='Germany')
ax[0,0].plot(france_X, france_y, label='France')
ax[0,0].legend()
ax[0,0].grid()
ax[0,0].set(xlabel='Number of days', ylabel='Confirmed cases',
       title='Confirmed cases')

#Plotting daily cases
ax[0,1].plot(uk_X, uk_dailyY, label='UK - daily cases')
ax[0,1].plot(italy_X, italy_daily, label='Italy - daily cases')
ax[0,1].plot(spain_X, spain_daily, label='Spain - daily cases')
ax[0,1].plot(germany_X, germany_daily, label='Germany - daily cases')
ax[0,1].plot(france_X, france_daily, label='France - daily cases')
ax[0,1].legend()
ax[0,1].grid()
ax[0,1].set(xlabel='Number of days', ylabel='Number of daily cases',
       title='Daily cases')


# Plotting differences in daily cases
ax[1,0].plot(uk_X, uk_deltaDaily, label='UK - rate of change of daily cases')
ax[1,0].plot(italy_X, italy_deltaDaily, label='Italy - rate of change of daily cases')
ax[1,0].plot(spain_X, spain_deltaDaily, label='Spain - rate of change of daily cases')
ax[1,0].plot(germany_X, germany_deltaDaily, label='Germany - rate of change of daily cases')
ax[1,0].plot(france_X, france_deltaDaily, label='France - rate of change of daily cases')
ax[1,0].legend()
ax[1,0].grid()
ax[1,0].set(xlabel='Number of days', ylabel='Change in daily cases',
       title='Change in daily cases')

plt.show()


# Predicting the future differences in daily cases for UK
#-------------------------------------------------------------
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(uk_X, uk_deltaDaily, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(uk_X, uk_deltaDaily)

futureDays = np.arange(start=(len(uk_X)+1),stop=147)
futureDays.reshape(1, -1)

uk_pred = regression.predict(futureDays)

ax[1,1].plot(futureDays, uk_pred)
ax[1,1].plot(uk_X, uk_deltaDaily)
ax[1,1].grid()
'''







