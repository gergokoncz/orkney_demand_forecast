from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

hour_list = ['hour_{}'.format(str(i)) for i in range(24)]
month_list = ['month_{}'.format(str(i)) for i in range(1,13)]
day_of_week_list = ['day_of_week_{}'.format(str(i)) for i in range(7)]

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        try:
            X = X.reset_index()
            X['datetime'] = X['time'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%dT%H:%M:%S'))
        except:
            X['datetime'] = X['datetime'].apply(lambda x: datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S'))
            if 'index' in X.columns:
                X = X.drop('index', axis = 1)
        X['year'] = X['datetime'].apply(lambda x: x.year)
        X['month'] = X['datetime'].apply(lambda x: x.month)
        X['day'] = X['datetime'].apply(lambda x: x.day)
        X['hour'] = X['datetime'].apply(lambda x: x.hour)
        #X['minute'] = X['datetime'].apply(lambda x: x.minute)
        X['day_of_week'] = X['datetime'].apply(lambda x: x.dayofweek) # if no aggregation
        return X.set_index('datetime')

class Shifter(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None, weeks = 2):
        return self

    def transform(self, X, y = None, weeks = 2):
        for i in range(weeks):
            colname = f"{i+1}weekback"
            X[colname] = X['Total'].shift((i+1) * 7, freq = 'D')
        X = X.dropna()
        return X

class LookUpOneWeekAgo(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None, weeks = 2):
        return self

    def transform(self, X, y = None):
        last_week_df = pd.read_pickle('last_week.pkl')
        last_week_df['datetime'] = last_week_df['week_later']
        last_week_df['LW_Total'] = last_week_df['Total']
        last_week_df = last_week_df.drop(['week_later', 'Total'], axis = 1).set_index('datetime')
        X = X.join(last_week_df, on = 'datetime')
        X = X.dropna()
        return X


class HourlyAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        
        # first drop zero values then check if the hour contains at least 30 measurements
        X = X.dropna()
        X_count_checker = X.groupby(['year', 'month', 'day', 'hour']).count()
        X_grouped = X.groupby(['year', 'month', 'day', 'hour']).mean()
        X_grouped = X_grouped.join(X_count_checker, on = ['year', 'month', 'day', 'hour'], rsuffix = '_check')
        
        # if the hour contains less then 30 measurements, drop the value
        X_grouped = X_grouped[X_grouped['Total_check'] > 30]['Total']
        for i in range(4)[::-1]:
            X_grouped = X_grouped.reset_index(level = i)
        
        # if your total value is outside 3 standard deviation for the period you are considered an outlier
        total_mean = X_grouped['Total'].mean()
        diff = 3 * X_grouped['Total'].std()

        X_grouped['stays'] = X_grouped['Total'].apply(lambda x: 1 if np.abs(x - total_mean) < diff else 0)
        
        X_grouped = X_grouped[X_grouped['stays'] == 1]
        X_grouped = X_grouped.drop('stays', axis = 1)
        X_grouped['datetime'] = X_grouped.apply(datetime_builder, axis = 1)
        X_grouped['day_of_week'] = X_grouped['datetime'].apply(lambda x: x.dayofweek)
        return X_grouped.set_index('datetime')


class DummyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        for col in ['hour', 'day_of_week', 'month']:
            X = X.join(pd.get_dummies(X[col], prefix=col))
        for c_list in [hour_list, day_of_week_list, month_list]:
            for col in c_list:
                if col not in X.columns:
                    X[col] = 0
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
        return X

def datetime_builder(x):
    try:
        return datetime(year = int(x['year']), month = int(x['month']), day = int(x['day']), hour = int(x['hour']))
    except:
        print(x)
