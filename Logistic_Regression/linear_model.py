import  numpy as np
import copy
class LogisticRegression:
    def __init__(self, params):

        self.no_of_iterations=params["no_of_iterations"]
        self.alpha=params["alpha"]

        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement

    def sigmoid(self,z):
        #print(z)
        sig=1/(1+np.exp(-z))
        return sig
    def update_params(self,X,y,y_hat):
        m=len(y)
        #print("eta yy")
        #print(y)
        #print("eta yy hat")
        #print(y_hat)

        #y=y.values.reshape(m,1)
        diff=y_hat-y
        dw = (1/m)*np.dot(X.T,diff)
        db = (1/m)*np.sum(diff)
        self.weights=self.weights-self.alpha*dw
        self.bias=self.bias-self.alpha*db
    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        # X=X.to_numpy()
        # y=y.to_numpy()
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(self.no_of_iterations):

            x_weights=X.dot(self.weights)+self.bias
            #print(x_weights)
            #x_weights=np.matmul(self.weights,X.T)+self.bias
            y_hat=self.sigmoid(x_weights)
            self.update_params(X,y,y_hat)
        # print("yy")
        # print(y_hat.shape)
        # print(y_hat)
        #print(self.sigmoid(X))

        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

    def predict(self, X):
        #print(self.weights)
        #print(self.bias)
        x_weights = X.dot(self.weights) + self.bias
        #print(x_weights)
        #x_weights=np.matmul(self.weights, X.T)+self.bias
        #print(x_weights)
        y_hat = self.sigmoid(x_weights)
        y_hat=np.round(y_hat)
        return y_hat
        # todo: implement