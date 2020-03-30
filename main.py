#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- This file is Property of Callum 'Cal' Alexander -*-
# -*- CONTACT INFORMATION -*-
# -*- Email : s1931801@ed.ac.uk -*-
# -*- Instagram : cal.zander -*-
# -*- Twitter : calzander -*-

"""
Created on Thu Mar 26 17:58:00 2020
@author: cal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


data = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
df = pd.read_csv(data)

#Data preprocessing for Italy
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

# I'll update preprocess for taking in territories

#Plotting data
italy_X, italy_y = preprocess(df, 'Italy')
spain_X, spain_y = preprocess(df, 'Spain')
uk_X, uk_y = preprocess(df, 'United Kingdom')

fig, ax = plt.subplots()
ax.plot(italy_X, italy_y, label='Italy')
ax.plot(spain_X, spain_y, label='Spain')
ax.plot(uk_X, uk_y, label='UK')

ax.legend()
ax.set(xlabel='Number of days', ylabel='Confirmed cases',
       title='Italy confirmed cases')
ax.grid()
plt.show()

