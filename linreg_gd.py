import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import mse, rss, tss, r2

class LinRegGD():
    def __init__(self, lr=1e-2, max_iter=10000, tolerance=1e-6, save_loss=True, save_weights=True):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.save_loss = save_loss
        self.save_weights = save_weights

        self.m = 0
        self.n = 0

        self.W = 0.0
        self.b = 0.0

        self.losses = []
        self.W_history = []
        self.b_history = []

        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, Y):
        # ensure proper dimensions:
        # X: (mxn), Y: (nx1)
        if len(X.shape) == 1: X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        # scale input features, avoiding division by 0
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        X = (X - self.mean_) / self.std_

        # m = num observations
        # n = num features
        self.m, self.n = X.shape
        
        self.losses = []
        self.W_history = []
        self.b_history = []
        
        #initalize weights and bias to 0
        self.W = np.zeros([self.n, 1])
        self.b = 0.0

        prev_loss = float('inf')

        for i in range(self.max_iter):
            #make a prediction
            Y_pred = self.predict(X, scale=False)

            #calculate gradient
            error = Y - Y_pred
            dmse_dW = -(2/self.m)*(X.T.dot(error)) # (nxm)(mx1) => (nx1)
            dmse_db = -(2/self.m)*np.sum(error)

            #update parameters
            self.W -= self.lr * dmse_dW
            self.b -= self.lr * dmse_db

            #save loss and weights
            if self.save_loss:
                current_loss = mse(Y, Y_pred)
                self.losses.append(current_loss)
            if self.save_weights:
                self.W_history.append(self.W)
                self.b_history.append(self.b)

            #check for convergence based on tolerance
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged at iteration {i+1}. Change in loss: {abs(prev_loss - current_loss):.8f}")
                break
            prev_loss = current_loss
    
        return self

    def predict(self, X, scale=True):
        #ensure proper shape
        if len(X.shape) == 1: X = X.reshape(-1, 1)

        # scale using training mean and std
        if scale and (self.mean_ is not None) and (self.std_ is not None):
            X = (X - self.mean_) / self.std_

        return self.b + X.dot(self.W) #(mx1) + (mxn)(nx1) => (mx1)
    
    def __str__(self):
        # displays equation line in format y = b + (w0)(x0) + (w1)(x1) + (w2)(x2) + ...
        coef_str = " + ".join([f"({w:.4f})x{i}" for i, w in enumerate(self.W.flatten())])
        return f"y = {self.b:.4f} + {coef_str}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for generating the line of best fit")
    parser.add_argument("--data_path", type=str, default="test_data/data.csv", help="Path to a csv file with n columns, starting with n-1 feature columns and 1 target column")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for gradient descent")
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum number of iterations for gradient descent")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance level for convergence")

    parser.add_argument("--graph_name", type=str, default="line_gd.png", help="File name for plot regression line with data, stored in images folder")
    parser.add_argument("--lossplot_name", type=str, default="losses_gd.png", help="File name for loss plot, stored in images folder")
    parser.add_argument("--animation_name", type=str, default="animation_gd.gif", help="File name for animation of lin reg, stored in images folder")

    args = parser.parse_args()

    # read in data from csv
    data = pd.read_csv(args.data_path)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    #initialize model, fit, and predict
    linear_model = LinRegGD(lr=args.lr, max_iter=args.max_iter, tolerance=args.tolerance)
    linear_model.fit(X, Y)
    Y_pred = linear_model.predict(X)

    #calculate metrics
    r2_score = r2(Y, Y_pred)
    MSE = mse(Y, Y_pred)
    print(f"Best Fit: {linear_model}")
    print(f"MSE: {MSE}")
    print(f"R^2: {r2_score:.4f}")

    def createLossPlot():
        print("Creating Loss Plot...")
        plt.plot(linear_model.losses)
        plt.xlabel("Iterations")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve")
        plt.savefig(f"images/{args.lossplot_name}")
        plt.close()

    def createRegressionPlot():
        print("Creating Regression Plot...")
        plt.scatter(X, Y, color='blue', label='Data')
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        Y_line = linear_model.predict(X_line)
        plt.plot(X_line, Y_line, color='red', label='Regression Line')
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("Data and Regression Line (GD)")
        plt.legend()
        plt.savefig(f"images/{args.graph_name}")
        plt.close()

    def createAnimation():
        print("Creating Animation...")
        from matplotlib.animation import FuncAnimation

        # Prepare X for line plotting (already scaled)
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_line_scaled = (X_line - linear_model.mean_) / linear_model.std_

        # Select step at every interval (plus last)
        interval = len(linear_model.W_history) // 100 if len(linear_model.W_history) > 100 else 1
        indices = list(range(0, len(linear_model.W_history), interval))
        if indices[-1] != len(linear_model.W_history) - 1:
            indices.append(len(linear_model.W_history) - 1)

        fig, ax = plt.subplots()
        ax.scatter(X, Y, color='blue', label='Data')
        line, = ax.plot([], [], color='red', label='Regression Line')
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.set_title("Regression Line (GD) Animation")
        ax.legend()

        def animate(idx):
            #print(idx)
            W = linear_model.W_history[indices[idx]]
            b = linear_model.b_history[indices[idx]]
            Y_line = b + X_line_scaled.dot(W)
            line.set_data(X_line, Y_line)
            ax.set_title(f"Iteration: {indices[idx]}, Loss: {linear_model.losses[indices[idx]]:.4f}")
            return line,

        anim = FuncAnimation(fig, animate, frames=len(indices), interval=200, blit=True)
        anim.save(f"images/{args.animation_name}", writer='pillow')
        plt.close(fig)

    createLossPlot()

    # Plot data and regression line if only 1 input feature
    if X.shape[1] == 1:
        createRegressionPlot()
    
    if X.shape[1] == 1 and len(linear_model.W_history) > 1:
        createAnimation()
    
    


        

    
    
