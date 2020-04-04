#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- This file is Property of Callum 'Cal' Alexander -*-
# -*- CONTACT INFORMATION -*-
# -*- Email : s1931801@ed.ac.uk -*-
# -*- Instagram : cal.zander -*-
# -*- Twitter : calzander -*-
# -*- GitHub : www.github.com/CallumAlexander -*-

"""
Created on Sat Apr  4 10:46:29 2020
@author: cal
"""

import numpy as np


# Data preprocessing function
def preprocess(dataset, country):
    isCountry = (dataset['Country/Region'] == country)
    countryData = dataset[isCountry]
    isCountry = countryData['Province/State'].isna()
    countryData = countryData[isCountry]

    y = countryData.iloc[:, 4:].values
    y = np.transpose(y)
    y = y.flatten()
    X = np.arange(len(y))
    return X, y


# Calculates the difference in adjacent array values
def delta(cases):
    y = [t - s for s, t in zip(cases, cases[1:])]
    y = [0] + y
    y = np.asarray(y, dtype=np.int64)
    return y


# Linear Regression
# *NOTE - unable to fit the data to sklearn's Linear Model so this was the alternative
def regression(X, y):
    denominator = X.dot(X) - X.mean() * X.sum()
    m = (X.dot(y) - y.mean() * X.sum()) / denominator
    b = (y.mean() * X.dot(X) - X.mean() * X.dot(y)) / denominator

    return (m * X + b), m