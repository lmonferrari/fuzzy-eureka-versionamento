import pandas as pd
import numpy as np


def create_data(data_dir):

    np.random.seed(42)

    data_size = 1000

    X1 = np.random.normal(0, 1, data_size)
    X2 = np.random.normal(5, 2, data_size)
    X3 = np.random.randint(0, 2, data_size)

    y = (2 * X1 - 3 * X2 + X3 > 0).astype(int)

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "y": y})

    df.to_csv(data_dir, index=False)
