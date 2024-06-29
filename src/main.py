import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from model import Model
from data_creation import create_data

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)

    production_dir = "production"
    artifacts_dir = "artifacts"
    model_subdir = "model"
    data_dir = "data"
    dataset_name = "dataset.csv"
    data_path = os.path.join(parent_dir, data_dir)

    execution_id = str(len(os.listdir(os.path.join(parent_dir, artifacts_dir))) + 1)
    model_path = os.path.join(parent_dir, artifacts_dir, execution_id, model_subdir)
    production_path = os.path.join(parent_dir, production_dir)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(production_dir, exist_ok=True)

    create_data(os.path.join(data_path, dataset_name))
    df = pd.read_csv(os.path.join(data_path, dataset_name))
    X = df.drop("y", axis=1)
    y = df["y"]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model_instance = Model(rf_model)
    model_instance.train(X, y)

    model_instance.save_model(model_path)
    model_instance.save_model(production_path)
    model_instance.save_data(
        model_instance.X_train,
        model_instance.y_train,
        model_instance.X_test,
        model_instance.y_test,
        data_path,
    )
