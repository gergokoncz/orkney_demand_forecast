from influxdb import InfluxDBClient
import pandas as pd
import mytransformers
from sklearn.pipeline import Pipeline
from datetime import timedelta, datetime

def save_last_week(weeks):
    import mytransformers
    client = InfluxDBClient(host = 'influxus.itu.dk', port = 8086, username = 'lsda', password = 'icanonlyread')
    client.switch_database('orkney')

    results = client.query('SELECT * FROM "Demand" WHERE time > now() - {}w'.format(str(weeks)))
    points = results.get_points()
    values = results.raw['series'][0]['values']
    columns = results.raw['series'][0]['columns']
    this_df = pd.DataFrame(values, columns = columns)

    pipeline = Pipeline([
        ('date_worker', mytransformers.DateTransformer()),
        ('aggregator', mytransformers.HourlyAggregator()),
        ])

    this_df = pipeline.fit_transform(this_df)
    this_df = this_df.reset_index(level = 0)
    this_df['week_later'] = this_df['datetime'].apply(lambda x: x + timedelta(days = 7))
    this_df[['week_later', 'Total']].to_pickle('last_week.pkl')

if __name__ == '__main__':
    save_last_week(15)
