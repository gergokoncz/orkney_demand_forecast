"""
last edited: 2020.05.15 11:03
author: gergokoncz
"""
# imports
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
import mytransformers

import os, warnings, sys

import logging
logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def get_data(weeks = 10):
    client = InfluxDBClient(host = 'influxus.itu.dk', port = 8086, username = 'lsda', password = 'icanonlyread')
    client.switch_database('orkney')

    results = client.query('SELECT * FROM "Demand" WHERE time > now() - {}w'.format(str(weeks)))
    points = results.get_points()
    values = results.raw['series'][0]['values']
    columns = results.raw['series'][0]['columns']
    return pd.DataFrame(values, columns = columns).set_index("time")

class OrkneyDemand(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.train_pipeline = Pipeline([
            ('date_worker', mytransformers.DateTransformer()),
            ('aggregator', mytransformers.HourlyAggregator()),
            ('onehot', mytransformers.DummyEncoder()),
            ('dateFeatures', mytransformers.DateTrendsAdder())
            #('shifter', mytransformers.Shifter())
            ])
        self.model = None

    def fit(self, X, alpha):
        #X = self.train_pipeline.fit_transform(X, shifter__weeks = 1)
        X = self.train_pipeline.fit_transform(X)
        y = X.pop('Total')
        self.model = Lasso(alpha).fit(X, y)
        print('model_trained')
        return self

    def predict(self, samples, context = None):
        samples = self.train_pipeline.fit_transform(samples)
        labels = samples.pop('Total')
        preds = self.model.predict(samples)
        return preds

if __name__ == '__main__':
    weeks = int(sys.argv[1] if len(sys.argv) > 1 else 10)
    alpha = float(sys.argv[2] if len(sys.argv) > 2 else 0.2)

    demand_df = get_data(weeks)

    # cross validate
    tscv = TimeSeriesSplit(5)
    for train_idx, test_idx in tscv.split(demand_df['Total']):
        model = OrkneyDemand().fit(demand_df.iloc[train_idx], alpha)
        preds = model.predict(samples = demand_df.iloc[test_idx])
        print(preds.shape)
        # need to aggregate test set to
        actual_values = model.train_pipeline.fit_transform(demand_df.iloc[test_idx])
        rmse, mae, r2 = eval_metrics(preds, actual_values['Total'])
        print(rmse)

