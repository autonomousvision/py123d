import io
import json
import pickle

import pandas as pd


class LoadScenarioFromCeph:
    def __init__(self):
        from petrel_client.client import Client

        self.file_client = Client("~/petreloss.conf")

    def list(self, dir_path):
        return list(self.file_client.list(dir_path))

    def save(self, data, url):
        self.file_client.put(url, pickle.dumps(data))

    def read_correct_csv(self, scenario_path):
        output = pd.read_csv(io.StringIO(self.file_client.get(scenario_path).decode("utf-8")), engine="python")
        return output

    def contains(self, url):
        return self.file_client.contains(url)

    def read_string(self, csv_url):
        from io import StringIO

        df = pd.read_csv(StringIO(str(self.file_client.get(csv_url), "utf-8")), sep=r"\s+", low_memory=False)
        return df

    def read(self, scenario_path):
        with io.BytesIO(self.file_client.get(scenario_path)) as f:
            datas = pickle.load(f)
            return datas

    def read_json(self, path):
        with io.BytesIO(self.file_client.get(path)) as f:
            data = json.load(f)
            return data

    def read_csv(self, scenario_path):
        return pickle.loads(self.file_client.get(scenario_path))

    def read_model(self, model_path):
        with io.BytesIO(self.file_client.get(model_path)) as f:
            pass
