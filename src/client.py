import os
import requests
import pandas as pd

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)
    data_dir = "data"
    test_file_name = "test_data.csv"

    url = "http://127.0.0.1:5000/predict"
    headers = {"Content-Type": "application/json"}

    data = pd.read_csv(os.path.join(parent_dir, data_dir, test_file_name))
    data_json = data.to_json()
    response = requests.post(url, headers=headers, data=data_json)

    if response.status_code != 200:
        print(response.reason)
        print(response.text)
    else:
        print("Predictions:", response.json())
