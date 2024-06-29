import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)


class Model:
    def __init__(self, model):
        self.model = model

    def train(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size
        )
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred)
        return score, precision, recall, cls_report

    def save_data(self, X_train, y_train, X_test, y_test, data_path):
        pd.concat([X_train, y_train], axis=1).to_csv(
            f"{data_path}/train_data.csv", index=False
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            f"{data_path}/test_data.csv", index=False
        )

    def save_model(self, model_path):
        joblib.dump(self.model, f"{model_path}/model.pkl")

    @classmethod
    def load_model(cls, model_path):
        model = joblib.load(f"{model_path}/model.pkl")
        return cls(model)
