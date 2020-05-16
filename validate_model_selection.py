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
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import mytransformers as mt
import last_week_saver as ls
import matplotlib.pyplot as plt

import os, warnings, sys

import logging
logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

plt.style.use('seaborn')

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

def evaluate_best_model(weeks = 10, alpha = 0.5):
    # cannot import pipeline from train
    pre_pipeline = Pipeline([
            ('date_worker', mt.DateTransformer()),
            ('aggregator', mt.HourlyAggregator()),
            ('onehot', mt.DummyEncoder()),
            ('dateFeatures', mt.DateTrendsAdder()),
            ('last_week', mt.LookUpOneWeekAgo())
            ])
    # make sure we have enough data
    demand_df = get_data(weeks * 2)
    ls.save_last_week(weeks * 2 + 10)

    demand_df = pre_pipeline.fit_transform(demand_df)

    models = [Lasso(alpha), Ridge(alpha), RandomForestRegressor(max_depth = 8)]
    model_names = ['lasso', 'ridge', 'randomforest']
    maes = []
    rmses = []
    r2s = []
        # get predictions for each model for  straight weeks
    for idx, c_model in enumerate(models):
        with mlflow.start_run():
            mlflow.log_param('model', model_names[idx])
            all_actuals = []
            all_preds = []
            year_ago = datetime.now() - timedelta(days = (weeks + 6) * 7)
            for i in range(1, 6):
                #print(year_ago + timedelta(days = 7 * i))
                mlflow.log_param('model', model_names[idx])
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
                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mae', mae)
                mlflow.log_metric('r2', r2)

            #print(mae)
        rmse, mae, r2 = eval_metrics(all_actuals, all_preds)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
    return maes, rmses, r2s


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
        plt.figure(0)
        plt.scatter(all_actuals, all_preds, alpha = 0.5, s = 0.8, c = 'r')
        plt.title(model_names[idx])
        plt.xlabel('actual values')
        plt.ylabel('predicted values')
        plt.savefig('images/{}_{}.png'.format(model_names[idx], weeks))