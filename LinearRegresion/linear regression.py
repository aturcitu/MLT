# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:25:05 2018

@author: Aitor Sanchez
"""

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# Googl Stock Prices
df = quandl.get("WIKI/GOOGL", authtoken="_8hSVL5tS9zEoBf2wh3G")
# Simplify your data, keep meaningful data 
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low']) / df['Adj. Close']*100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open']*100
# Keep relevant
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
# Target for regresion
forecast_col = 'Adj. Close'
# Fill holes into your data set, use an Outlier value
df.fillna(-99999, inplace=True)
# % of forecast for the regresion
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
# FEATURES
# returns a new dataframe excluding dropping list, in this case, label column
X = np.array(df.drop(['label'],1))
# Normalize data (needs to be done all together)
# Center to the mean and component wise scale to unit variance.
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)

# Labels
y = np.array(df['label'])

# Split data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
# Create Classifier
# n_jobs used to multi-trhead the algorithmn, -1 for all you can use!

# TRAIN
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
# save
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuray = clf.score(X_test, y_test)
# Prediction
forecast_set = clf.predict(X_lately)
# Ads new column to the DF for the forecasgted data
df['Forecast'] = np.nan
# Ads time to forecasted data
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
# time frame for the first predicte value
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()