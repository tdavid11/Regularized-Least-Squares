import numpy as np
class OLS():
    """
    This class implements Linear Regression Using QR Factorization. 

    No intercept is used, as data is expected to be scaled or normalized.
    """
    def __init__(self):
        self.beta = []

    def fit(self,X,y):

        Q, R = np.linalg.qr(X)
        R_inv = np.linalg.inv(R)
        self.beta = np.dot(R_inv, np.dot(Q.T, y))
        print(self.beta)

    def predict(self,X):
        return np.dot(X,self.beta)
        
class Ridge():
    """
    This class implements Ridge Regression Using QR Factorization. 

    No intercept is used, as data is expected to be scaled or normalized.
    """
    def __init__(self, lambda_ridge):
        self.beta = []
        self.lambda_ridge = lambda_ridge
        
    def fit(self,X,y):
        I_lambda = np.sqrt(self.lambda_ridge)*np.identity(np.shape(X)[1])
        X_star = np.concatenate((X,I_lambda))
        y_star = np.concatenate((y,np.zeros(np.shape(X)[1])))
        Q, R = np.linalg.qr(X_star)
        R_inv = np.linalg.inv(R)
        self.beta = np.dot(R_inv, np.dot(Q.T, y_star))
        print(self.beta)

    def predict(self,X):
        return np.dot(X,self.beta)

class Lasso():
    """
    This class implements Lasso Regression Using Coordinate Descent. 

    No intercept is used, as data is expected to be scaled or normalized.
    """
    def __init__(self, lambda_lasso):
        self.beta = []
        self.lambda_lasso = lambda_lasso
        
    def fit(self,X,y,n):
        self.beta = np.random.randn(np.shape(X)[1])
        #print(X)
        #print(self.beta)
        for i in range(n):
            for j in range(np.shape(X)[1]):
                y_pred = X@self.beta
                #print(y_pred)
                col = X[:,j]
                b = self.beta[j]
                #term=y-y_pred + b*col 
                #print(term)
                rho = col.T@(y-y_pred + b*col )
                #print(rho)
                #print(rho)
                if rho<-self.lambda_lasso:
                    self.beta[j]=rho+self.lambda_lasso
                elif rho > self.lambda_lasso:
                    self.beta[j]=rho - self.lambda_lasso
                else:
                    self.beta[j]=0
        print(self.beta)
    def predict(self,X):
        return np.dot(X,self.beta)