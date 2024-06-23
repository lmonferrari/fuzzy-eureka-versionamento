import hashlib
import os
import time
import pandas as pd

from model import Model
from data_creation import create_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)

    artifacts_dir = "artifacts"
    model_dir = "model"
    data_dir = "data"
    dataset_name = "dataset.csv"
    data_path = os.path.join(parent_dir, data_dir, dataset_name)
    execution_id = str(len(os.listdir(os.path.join(parent_dir, artifacts_dir))) + 1)
    model_path = os.path.join(parent_dir, artifacts_dir, execution_id, model_dir)

    create_data(data_path)
    df = pd.read_csv(data_path)
    X = df.drop("y", axis=1)
    y = df["y"]

    modelo = Model(
        model_type=RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        X=X,
        y=y,
    )

    modelo.train()
    os.makedirs(model_path, exist_ok=True)
    modelo.save_model(model_path + f"/model_v{execution_id}.pkl")
