import requests, json
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    test_df = pd.DataFrame({'datetime': ['2020-05-17 09:00:00']})
    to_test = test_df.to_json(orient = 'split')
    print(to_test)
    endpoint = "http://127.0.0.1:1234/invocations"
    headers = {"Content-type": "application/json; format=pandas-split"}
    response = requests.post(endpoint, json = json.loads(to_test), headers = headers)
    print(response.text)


