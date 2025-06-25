import numpy as np

class LinRegGD():
    def __init__(self, lr=0.01, num_iter=100):
        self.lr = lr
        self.num_iter = num_iter

        self.m = 0
        self.n = 0

        self.W = 0
        self.b = 0

        self.losses = []
    
    def fit(self, X, Y, save_losses=False):
        self.m, self.n = X.shape
        self.losses = np.empty(self.num_iter)

        self.W = np.zeros([self.n, 1])
        self.b = 0

        for i in range(self.num_iter):
            #make a prediction
            Y_pred = self.predict(X)

            #calculate gradient
            dmse_dW = -(2/self.m)*(X.T.dot(Y - Y_pred)) # (nxm)(mx1) => (nx1)
            dmse_db = -(2/self.m)*np.sum(Y-Y_pred)

            #update parameters
            self.W -= self.lr * dmse_dW
            self.b -= self.lr * dmse_db

            #optionally save loss
            if save_losses: self.losses[i] = self.mse(Y, Y_pred)
    
        return self

        
    def predict(self, X):
        return self.b + X.dot(self.W) #(mx1) + (mxn)(nx1) => (mx1)
    
    def mse(self, Y_true, Y_pred):
        return (1/self.m)*np.sum( (Y_true - Y_pred) ** 2)
    
    
