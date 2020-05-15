from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.reset_index()
        X['datetime'] = X['time'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%dT%H:%M:%S'))
        X['year'] = X['datetime'].apply(lambda x: x.year)
        X['month'] = X['datetime'].apply(lambda x: x.month)
        X['day'] = X['datetime'].apply(lambda x: x.day)
        X['hour'] = X['datetime'].apply(lambda x: x.hour)
        X['minute'] = X['datetime'].apply(lambda x: x.minute)
        X['day_of_week'] = X['datetime'].apply(lambda x: x.dayofweek)
        print(X.shape)
        return X.set_index('datetime')

class Shifter(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None, weeks = 2):
        return self

    def fit_transform(self, X, y = None, weeks = 2):
        for i in range(weeks):
            colname = f"{i+1}weekback"
            X[colname] = X['Total'].shift((i+1) * 7, freq = 'D')
        X = X.dropna()
        #X = X.drop(columns = ['Total'], axis = 1)
        print(X.shape)
        return X


class HourlyAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_grouped = X.groupby(['year', 'month', 'day', 'hour']).mean()['Total']
        for i in range(4)[::-1]:
            X_grouped = X_grouped.reset_index(level = i)
        
        X_grouped['datetime'] = X_grouped.apply(datetime_builder, axis = 1)
        X_grouped['day_of_week'] = X_grouped['datetime'].apply(lambda x: x.dayofweek)
        y = X_grouped['Total']
        print(X_grouped.shape)
        #X = X.drop(columns = ['Total'], axis = 1)
        return X_grouped.set_index('datetime')


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        for col in ['hour', 'day_of_week']:
            X = X.join(pd.get_dummies(X[col], prefix=col))
        y = X['Total']
        print(X.shape)
        #X = X.drop(columns = ['Total'], axis = 1)
        return X

class DateTrendsAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X['distance_from_july'] = X['month'].apply(lambda x: np.abs(x - 7))
        X['is_week_day'] = X['day_of_week'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
        X['is_day'] =  X['hour'].apply(lambda x: 1 if x > 6 and x < 21 else 0)
        y = X['Total']
        print(X.shape)
        #X = X.drop(columns = ['Total'], axis = 1)
        return X

def datetime_builder(x):
    try:
        return datetime(year = int(x['year']), month = int(x['month']), day = int(x['day']), hour = int(x['hour']))
    except:
        print(x)
