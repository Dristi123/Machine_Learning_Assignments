import numpy as np

from data_handler import bagging_sampler


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator=base_estimator
        self.n_estimator=n_estimator
        self.models=[]
        # todo: implement

    def fit(self, X, y):
        for i in range(self.n_estimator):
            X_c, y_c = bagging_sampler(X, y)
            self.base_estimator.fit(X_c,y_c)
            self.models.append(self.base_estimator)


        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

    def predict(self, X):
        y_hat = np.empty((len(self.models), len(X)))
        i=0
        for model in (self.models):
            y_hat[i] = model.predict(X)
            i=i+1
        #print("eta y")
        # print(y_test_hats)
        #print(np.round(y_hat.mean(0)))
        #print(y_hat.mean(0))
        y_hat=y_hat.mean(0)
        y_hat=np.where(y_hat>0.5, 1, 0)
        return y_hat

        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement

