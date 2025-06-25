import sys
import argparse
import numpy as np

class LinearModel():
    
    def __init__(self, x: np.ndarray | list, y: np.ndarray | list) -> None:
        x, y = np.array(x), np.array(y)

        self.x = x
        self.y = y

        self.Sxy = np.sum( (x - np.mean(x)) * (y - np.mean(y)))
        self.Sxx = np.sum( (x - np.mean(x)) ** 2)

        self.b_1 = self.Sxy/self.Sxx
        self.b_0 = np.mean(y) - self.b_1 * np.mean(x)

        self.RSS = np.sum( (y - self.predict(x)) ** 2 )
        self.TSS = np.sum( (y - np.mean(y)) ** 2)
        self.R_2 = 1 - (self.RSS/self.TSS)
    
    def predict(self, x):
        return self.b_0 + self.b_1 * np.array(x)

    def __str__(self):
        return f"y = {self.b_0:.4f} + {self.b_1:.4f}x"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for generating the line of best fit")
    parser.add_argument("--x", nargs="+", type=float, help="A nonempty list of x values")
    parser.add_argument("--y", nargs="+", type=float, help="A nonempty list of y values")
    args = parser.parse_args()

    if not args.x: 
        print("No x values provided")
        sys.exit(1)
    if not args.y: 
        print("No y values provided")
        sys.exit(1)
    if not len(args.x) == len(args.y): 
        print("Mismatch between length of x and y")
        sys.exit(1)

    linear_model = LinearModel(args.x, args.y)
    y_pred = linear_model.predict(args.x)
    R_2 = linear_model.R_2
    print(f"Line of Best Fit: {linear_model}")
    print(f"R^2: {R_2}")
