class DataChecker:
    """
    A class to check the shapes of the training and test sets, and to check for negative values and NaN values in the 
    target and feature variables.

    Methods:
    --------
    check_shapes():
        Checks if the shapes of the training and test sets match and if the number of features and targets match.

    check_negative_values_y():
        Checks if there are negative values in the target variables of the training and test sets.

    check_nan_values_X():
        Checks if there are NaN values in the feature variables of the training and test sets.

    Parameters:
    -----------
    X_train : numpy array or pandas dataframe
        Training set features.
    y_train : numpy array or pandas dataframe
        Training set target variable.
    X_test : numpy array or pandas dataframe
        Test set features.
    y_test : numpy array or pandas dataframe
        Test set target variable.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def check_shapes(self):
        """
        Checks if the shapes of the training and test sets match and if the number of features and targets match.
        """
        assert self.X_train.shape[0] == self.y_train.shape[0], "Mismatched shapes between X_train and y_train"
        assert self.X_test.shape[0] == self.y_test.shape[0], "Mismatched shapes between X_test and y_test"
        assert self.X_train.shape[1] == self.X_test.shape[1], "Mismatched number of features between X_train and X_test"
        assert self.y_train.shape[1] == self.y_test.shape[1], "Mismatched number of targets between y_train and y_test"

    def check_negative_values_y(self):
        """
        Checks if there are negative values in the target variables of the training and test sets.
        """
        assert (self.y_train >= 0).all().all(), "Negative values in training data"
        assert (self.y_test >= 0).all().all(), "Negative values in test data"

    def check_nan_values_X(self):
        """
        Checks if there are NaN values in the feature variables of the training and test sets.
        """
        assert not self.X_train.isnull().values.any(), "NaN values in training data"
        assert not self.X_test.isnull().values.any(), "NaN values in test data"
