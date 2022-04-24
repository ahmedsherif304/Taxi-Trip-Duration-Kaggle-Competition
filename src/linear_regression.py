import numpay as np
class OurLinearRegression:
    def _prepare_inputs(self, X):
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate((ones, X), axis=1)

    def fit(self, X, y):
        X = self._prepare_inputs(X)
        X_transpose = X.transpose()
        self.w =  np.linalg.solve( np.matmul(X_transpose, X), np.matmul(X_transpose, y)  ) 
        return self
    
    def predict(self, X):
        X = self._prepare_inputs(X) 
        return np.matmul( X , self.w)
def our_mean_square_error(true, predicted):
    return np.mean(np.square(predicted - true))

def main():
    X_train = 1 
    y_train = 1
    X_test = 1
    y_test = 1
    our_model = OurLinearRegression().fit(X_train, y_train)
    y_train_predict = our_model.predict(X_train)
    training_error = our_mean_square_error(y_train, y_train_predict)
    print(f"Training Error: {training_error} (RMS: {training_error**0.5})")
    y_test_predict = our_model.predict(X_test)
    testing_error = our_mean_square_error(y_test, y_test_predict)
    print(f"Testing Error: {testing_error} (RMS: {testing_error**0.5})")
    

if __name__ == "__main__":
    main()