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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import mytransformers as mt
import last_week_saver as ls

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

    def __init__(self, weeks, model):
        self.train_pipeline = Pipeline([
            ('date_worker', mt.DateTransformer()),
            ('aggregator', mt.HourlyAggregator()),
            ('onehot', mt.DummyEncoder()),
            ('dateFeatures', mt.DateTrendsAdder()),
            ('last_week', mt.LookUpOneWeekAgo())
            ])
        self.model = model
        self.test_pipeline = Pipeline([
            ('date_worker', mt.DateTransformer()),
            ('onehot', mt.DummyEncoder()),
            ('dateFeatures', mt.DateTrendsAdder()),
            ('last_week', mt.LookUpOneWeekAgo())
        ])
        ls.save_last_week(weeks)

    def fit(self, X):
        #X = self.train_pipeline.fit_transform(X, shifter__weeks = 1)
        X = self.train_pipeline.fit_transform(X)
        y = X.pop('Total')
        self.model.fit(X, y)
        #for col, val in zip(X.columns, self.model.coef_):
        #    print(col, '\t', val)
        return self

    def predict(self, context, samples):
        if 'Total' in samples.columns:
            samples = self.train_pipeline.fit_transform(samples)
            labels = samples.pop('Total')
        else:
            samples = self.test_pipeline.fit_transform(samples)
        preds = self.model.predict(samples)
        return preds

if __name__ == '__main__':
    weeks = int(sys.argv[1] if len(sys.argv) > 1 else 10)
    alpha = float(sys.argv[2] if len(sys.argv) > 2 else 0.2)

    demand_df = get_data(weeks)

    models = [Lasso(alpha), Ridge(alpha), RandomForestRegressor(max_depth = 8), DecisionTreeRegressor(max_depth = 20), SVR(kernel = 'poly', degree = 4)]
    model_names = ['lasso', 'ridge', 'randomforest', 'tree', 'svr']
    for idx,c_model in enumerate(models):
        with mlflow.start_run():
    # cross validate
            tscv = TimeSeriesSplit(4)
            for train_idx, test_idx in tscv.split(demand_df['Total']):
                model = OrkneyDemand(weeks, c_model).fit(demand_df.iloc[train_idx])
                preds = model.predict(context = None, samples = demand_df.iloc[test_idx])
                # need to aggregate test set to
                actual_values = model.train_pipeline.fit_transform(demand_df.iloc[test_idx])
                rmse, mae, r2 = eval_metrics(actual_values['Total'], preds)

                mlflow.log_param('name', model_names[idx])
                mlflow.log_param('weeks', weeks)
                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('r2', r2)
    
    #mlflow.pyfunc.save_model("model", python_model = model, conda_env = 'conda.yaml')

