"""
last edited: 2020.05.16 15:21
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
import matplotlib.pyplot as plt

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

    pre_pipeline = Pipeline([
            ('date_worker', mt.DateTransformer()),
            ('aggregator', mt.HourlyAggregator()),
            ('onehot', mt.DummyEncoder()),
            ('dateFeatures', mt.DateTrendsAdder()),
            ('last_week', mt.LookUpOneWeekAgo())
            ])
    # make sure we have     
    demand_df = get_data(60)
    ls.save_last_week(70)

    demand_df = pre_pipeline.fit_transform(demand_df)

    models = [Lasso(alpha), Ridge(alpha), RandomForestRegressor(max_depth = 8)]
    model_names = ['lasso', 'ridge', 'randomforest']
    for idx,c_model in enumerate(models):
        print(model_names[idx])
        all_actuals = []
        all_preds = []
        year_ago = datetime.now() - timedelta(days = 400)
        for i in range(1, 45):
            #print(year_ago + timedelta(days = 7 * i))
            #print(year_ago + timedelta(days = 7 * (i + weeks)))
            #print(year_ago + timedelta(days = 7 * (i + weeks + 1)))
            train_df = demand_df.loc[demand_df.index > year_ago + timedelta(days = 7 * i)]
            train_df = train_df.loc[train_df.index < year_ago + timedelta(days = 7 * (i + weeks))]
            test_df = demand_df.loc[demand_df.index > year_ago + timedelta(days = 7 * (i + weeks))]
            test_df = test_df.loc[test_df.index < year_ago + timedelta(days = 7 * (i + weeks + 1))]

            train_y = train_df.pop('Total')
            test_y = test_df.pop('Total')
            #print(train_df)
            #print(test_df)
            c_model.fit(train_df, train_y)
            preds = c_model.predict(test_df)
            for pred in preds:
                all_preds.append(pred)
            for el in test_y:
                all_actuals.append(el)
            # need to aggregate test set to
            rmse, mae, r2 = eval_metrics(test_y, preds)

            #print(mae)
        rmse, mae, r2 = eval_metrics(all_actuals, all_preds)
        print('\n\n')
        print(mae)
        print(r2)
        print(rmse)
        print('\n\n')
        plt.scatter(all_actuals, all_preds)
        plt.savefig('{}_{}.png'.format(model_names[idx], weeks))
    
    #mlflow.pyfunc.save_model("model", python_model = model, conda_env = 'conda.yaml')