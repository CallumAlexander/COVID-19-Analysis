import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def findOptimumForecastParam(y):
    # This lazy programming but it works
    # TODO - convert to a 2d array
    paramList = []
    param_seasonalList = []
    aicList = []

    fullList = []

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                # fullList.append([param, param_seasonal, results.aic])
                aicList.append(results.aic)
                paramList.append(param)
                param_seasonalList.append(param_seasonal)
            except:
                continue

        optimumIndex = np.argmin(np.asarray(aicList))
        print('{}, {}, {}'.format(aicList[optimumIndex], paramList[optimumIndex], param_seasonalList[optimumIndex]))
        return paramList[optimumIndex], param_seasonalList[optimumIndex], aicList[optimumIndex]


def forecast(y, pdq, seasonal_pdq):
    # Fit model to data
    model = sm.tsa.statespace.SARIMAX(y,
                                      order=pdq,
                                      seasonal_order=seasonal_pdq,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)

    results = model.fit()
    return results
