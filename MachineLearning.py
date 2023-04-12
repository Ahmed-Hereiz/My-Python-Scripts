import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class ModelEvaluate:
    """
    A class that takes a list of regression models and evaluates their performance
    using cross-validation.

    Parameters
    ----------
    models : list
        A list of regression models to evaluate.

    Methods
    -------
    fit(X_train, y_train)
        Fits the regression models on the training data using cross-validation and
        stores the evaluation results in the 'results' attribute.

    get_results()
        Returns the evaluation results as a pandas DataFrame.
    """
    def __init__(self, models):
        self.models = models

    def fit(self, X_train, y_train):
        self.results = []
        for model in self.models:
            start = time.time()
            scores = cross_validate(model, X_train, y_train, cv=5,
                                    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                    return_train_score=False)
            end = time.time()
            results_dict = {}
            results_dict['model'] = model.__class__.__name__
            results_dict['mean_mae'] = -np.mean(scores['test_neg_mean_absolute_error'])
            results_dict['mean_rmse'] = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
            results_dict['mean_r2'] = np.mean(scores['test_r2'])
            results_dict['time'] = end - start
            self.results.append(results_dict)

    def get_results(self):
        return pd.DataFrame(self.results)    



class RegressionPlot:
    """A class for creating a set of plots to visualize the performance of a regression model.

    Parameters
    ----------
    y_test : pandas.DataFrame
        The actual target values for the test set.
    y_pred : array-like
        The predicted target values for the test set.
    color : str, optional
        The color to use for the plot markers and histograms.

    Methods
    -------
    plot()
        Creates a set of three plots to visualize the performance of the regression model.

    """

    def __init__(self, y_test, y_pred, color='b'):
        self.y_test = y_test
        self.y_pred = y_pred
        self.color = color
    
    def plot(self):
        """Creates a set of three plots to visualize the performance of the regression model.

        The three plots are: a scatter plot with regression line, a histogram of errors, and a residual plot.
        """

        # Create subplots
        fig, axs = plt.subplots(ncols=3, figsize=(15,5))

        # Plot scatter plot with regression line
        axs[0].scatter(self.y_test[self.y_test.columns[0]], self.y_pred, color=self.color)
        axs[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=3, color='k')
        axs[0].set_xlabel('Actual Values')
        axs[0].set_ylabel('Predicted Values')
        axs[0].set_title('Scatter Plot with Regression Line')

        # Plot histogram of errors
        errors = self.y_test[self.y_test.columns[0]] - self.y_pred
        axs[1].hist(errors, bins=50, color=self.color)
        axs[1].axvline(x=errors.median(), color='k', linestyle='--', lw=3)
        axs[1].set_xlabel('Errors')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Errors')

        # Plot residual plot
        axs[2].scatter(self.y_pred, errors, color=self.color)
        axs[2].axhline(y=0, color='k', linestyle='-', lw=3)
        axs[2].set_xlabel('Predicted Values')
        axs[2].set_ylabel('Errors')
        axs[2].set_title('Residual Plot')

        # Show the plots
        plt.tight_layout()
        plt.show()