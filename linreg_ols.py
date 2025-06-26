import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import mse, rss, tss, r2

class LinearModel():
    
    def __init__(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        x, y = np.array(x).flatten(), np.array(y).flatten()

        self.x = x
        self.y = y

        self.Sxy = np.sum( (x - np.mean(x)) * (y - np.mean(y)))
        self.Sxx = np.sum( (x - np.mean(x)) ** 2)

        self.b_1 = self.Sxy/self.Sxx
        self.b_0 = np.mean(y) - self.b_1 * np.mean(x)
    
    def predict(self, x):
        return self.b_0 + self.b_1 * np.array(x)

    def __str__(self):
        return f"y = {self.b_0:.4f} + {self.b_1:.4f}x"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for generating the line of best fit")
    parser.add_argument("--data_path", type=str, default="test_data/data.csv", help="Path to a csv file with starting with 1 feature column, then a target column")
    parser.add_argument("--graph_name", type=str, default="line_ols.png", help="File name for plot regression line with data, stored in images folder")
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    linear_model = LinearModel(X, Y)
    Y_pred = linear_model.predict(X)
    r2_score = r2(Y, Y_pred)
    MSE = mse(Y, Y_pred)

    print(f"Line of Best Fit: {linear_model}")
    print(f"MSE: {MSE}")
    print(f"R^2: {r2_score}")

    def createRegressionPlot():
        print("Creating Regression Plot...")
        plt.scatter(X, Y, color='blue', label='Data')
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        Y_line = linear_model.predict(X_line)
        plt.plot(X_line, Y_line, color='red', label='Regression Line')
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("Data and Regression Line (OLS)")
        plt.legend()
        plt.savefig(f"images/{args.graph_name}")
        plt.close()
    
    createRegressionPlot()

