import os
import pandas as pd
from model import Model

if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)

    artifacts_dir = "artifacts"
    model_subdir = "model"
    data_dir = "data"
    dataset_test_name = "test_data.csv"

    execution_id = str(len(os.listdir(os.path.join(parent_dir, artifacts_dir))))
    model_path = os.path.join(parent_dir, artifacts_dir, execution_id, model_subdir)
    data_path = os.path.join(parent_dir, data_dir, dataset_test_name)

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("No models or test data to evaluate...")
    else:
        print(f"Loaded model: {execution_id}")
        model_instance = Model.load_model(model_path)
        df_test = pd.read_csv(data_path)
        X_test = df_test.drop("y", axis=1)
        y_test = df_test["y"]

        score, precision, recall, cls_report = model_instance.evaluate(X_test, y_test)

        print(f"Accuracy: {score}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"Classification Report:\n{cls_report}")
