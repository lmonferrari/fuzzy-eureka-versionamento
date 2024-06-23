import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class Model:
    def __init__(self, X, y, model_type, test_size=0.2):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.test_size = test_size

    def train_test_data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )

    def train(self):
        self.train_test_data_split()
        self.model = self.model_type
        self.model.fit(self.X_train, self.y_train)

    def prediction(self):
        self.y_pred = self.model.prediction(self.X_test)

    def evaluate(self):
        self.prediction()
        self.score = accuracy_score(self.y_test, self.y_pred)
        self.cls_report = classification_report(self.y_test, self.y_pred)

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
