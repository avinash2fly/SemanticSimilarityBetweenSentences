import numpy as np


class perceptron():
    def __init__(self, rate = 0.1, n_iter = 1000):
        self.rate = rate
        self.n_iter = n_iter

    def predict(self, x = []):
        y_predict = np.dot(x, self.weights[1:]) + self.weights[0]
        return np.where(y_predict >= 0, 1, 0)

    def fit(self, X = [], Y = []):
        if len(X) == 0:
            return self
        #Initialize weights
        self.weights = np.random.normal(loc=0, scale=1, size=len(X[0]) + 1)
        # Keep track of incorrect prediction
        self.error = []

        for i in range(self.n_iter):
            incorrect_cnt = 0
            for x, y in zip(X, Y):
                y_pred = self.predict(x)
                incorrect_cnt += int(y_pred != y)
                # Update weight
                self.weights += np.multiply(self.rate*(y - y_pred), [1]+x)
            self.error.append(incorrect_cnt)
        return self

    def performance(self, Y_pred = [], Y = []):
        '''
           TP  |  FN
           ----------
           FP  |  TN
        '''
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1:
                if Y[i] == 1:
                    TP += 1
                else:
                    FP += 1
            elif Y_pred[i] == 0 and Y[i] == 1:
                FN += 1
        if (TP + FP) == 0:
            precision = 0
        else:
            precision = TP/(TP + FP)
        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP/(TP + FN)
        if precision + recall == 0:
            return 0, 0, 0
        F = 2*precision*recall/(precision + recall)

        #print("threshod : {0:4.2f}, alpha : {1:4.2f}, beta : {6:4.2f}, beta : {7:4.2f}, Precision : {2:6.4f}, Recall : {3:6.4f}, F : {4:6.4f}, Run time : {5:6.4f} seconds".format(threshold, alpha, precision, recall, F, time.time()-start_time, beta, gamma))
        return precision, recall, F
