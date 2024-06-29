import os
import pandas as pd
from model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    return jsonify(model.predict(df.drop("y", axis=1)).tolist())


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(basedir)
    production_dir = "production"
    model_name = "model.pkl"
    production_path = os.path.join(parent_dir, production_dir)
    model = Model.load_model(production_path)
    app.run()
