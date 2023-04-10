import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from scipy.stats import yeojohnson, boxcox
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder


class ColumnSelector(TransformerMixin, BaseEstimator):
    """
    Transformer that selects only the specified columns from a data frame.

    Parameters:
    ----------
    columns : list or array-like, default=None
        List of column names to select from the input data frame. If None, all columns are selected.
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X[self.columns]
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

    
    
class ArithmeticColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to add, subtract, multiply, or divide two columns in a Pandas DataFrame and drop the original columns.

    Parameters
    ----------
    col1 : str
        The name of the first column to operate on.
    col2 : str
        The name of the second column to operate on.
    operator : str, optional
        The arithmetic operator to use. Allowed values are '+', '-', '*', and '/'.
    new_col_name : str, optional
        Name for the new column (default = 'New column').

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the result of the operation and the original columns dropped.
    """
    def __init__(self, col1, col2, operator='+', new_col_name='New column'):
        self.col1 = col1
        self.col2 = col2
        self.operator = operator
        self.new_col_name = new_col_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if self.operator == '+':
            X_new[self.new_col_name] = X_new[self.col1] + X_new[self.col2]
        elif self.operator == '-':
            X_new[self.new_col_name] = X_new[self.col1] - X_new[self.col2]
        elif self.operator == '*':
            X_new[self.new_col_name] = X_new[self.col1] * X_new[self.col2]
        elif self.operator == '/':
            X_new[self.new_col_name] = X_new[self.col1] / X_new[self.col2]
        else:
            raise ValueError("Invalid value for 'operator'. Allowed values are '+', '-', '*', and '/'.")
        return X_new.drop([self.col1, self.col2], axis=1)


    
class DataFrameImputer(TransformerMixin, BaseEstimator):
    """
    A class to impute missing values in a Pandas DataFrame using a combination of median, knn, and most frequent
    imputers on specified columns.

    Parameters:
    -----------
    median_cols : list of str, optional (default=None)
        Columns to impute missing values using the median imputer.
    knn_cols : list of str, optional (default=None)
        Columns to impute missing values using the KNN imputer.
    freq_cols : list of str, optional (default=None)
        Columns to impute missing values using the most frequent imputer.
    const_cols : dict of {column_name: constant_value} pairs, optional (default=None)
        Columns to impute missing values using a constant value.

    Returns:
    --------
    X_imputed : pandas.DataFrame
        A DataFrame with imputed missing values.
    """
    def __init__(self, median_cols=None, knn_cols=None, freq_cols=None, const_cols=None, fill_const=0):
        self.median_cols = median_cols
        self.knn_cols = knn_cols
        self.freq_cols = freq_cols
        self.const_cols = const_cols
        self.fill_const = fill_const
    
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer()
        self.freq_imputer = SimpleImputer(strategy='most_frequent')
        self.const_imputer = SimpleImputer(strategy='constant', fill_value=self.fill_const)

        if self.median_cols is not None:
            self.median_imputer.fit(X[self.median_cols])
        if self.knn_cols is not None:
            self.knn_imputer.fit(X[self.knn_cols])
        if self.freq_cols is not None:
            self.freq_imputer.fit(X[self.freq_cols])
        if self.const_cols is not None:
            self.const_imputer.fit(X[self.const_cols])

        return self
    
    def transform(self, X):
        X_imputed = X.copy()
        if self.median_cols is not None:
            X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]), 
                                    columns=self.median_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.median_cols, axis=1), X_median], axis=1)
        if self.knn_cols is not None:
            X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]), 
                                 columns=self.knn_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.knn_cols, axis=1), X_knn], axis=1)
        if self.freq_cols is not None:
            X_freq = pd.DataFrame(self.freq_imputer.transform(X[self.freq_cols]), 
                                  columns=self.freq_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.freq_cols, axis=1), X_freq], axis=1)
        if self.const_cols is not None:
            X_const = pd.DataFrame(self.const_imputer.transform(X[self.const_cols]), 
                                  columns=self.const_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.const_cols, axis=1), X_const], axis=1)
        return X_imputed
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that drops specified columns from a DataFrame.

    Parameters
    ----------
    columns : list
        A list of column names to be dropped.
    return
    ------
        dataframe with dropped columns
    """
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.columns is None:
            return X
        else:
            return X.drop(self.columns,axis=1)
        
        
class WinsorizationImpute(BaseEstimator, TransformerMixin):
    """
    A transformer that performs winsorization imputation on specified columns in a Pandas DataFrame.

    Parameters:
    -----------
    p : float, default=0.05
        The percentile value representing the lower bound for winsorization.
    q : float, default=0.95
        The percentile value representing the upper bound for winsorization.
    random_state : int, default=42
        Seed for the random number generator used for imputing missing values.
    columns : list
        The list of names of columns to be winsorized.
    outlier_handling : str, default='random_in_distribution'
        The method used for imputing missing values. Valid options are:
        - 'random_in_distribution_std': impute using random values generated from a normal distribution
        - 'median': impute using the median value of the column
        - 'lower_bound_upper_bound': impute using the lower bound for lower outliers and the upper bound for upper outliers

    Returns:
    --------
    A new Pandas DataFrame with the specified columns winsorized.

    """
    def __init__(self, columns, p=0.05, q=0.95, random_state=42, outlier_handling='lower_bound_upper_bound'):
        self.p = p
        self.q = q
        self.random_state = random_state
        self.columns = columns
        self.outlier_handling = outlier_handling
        
    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.median_ = {}
        for col in self.columns:
            lower_bound = np.percentile(X[col], self.p * 100)
            upper_bound = np.percentile(X[col], self.q * 100)
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
            self.median_[col] = X[col].median()
        return self
    
    def transform(self, X):
        X_winsorized = X.copy()
        for col in self.columns:
            lower_bound = self.lower_bounds_[col]
            upper_bound = self.upper_bounds_[col]
            outliers_mask = (X_winsorized[col] < lower_bound) | (X_winsorized[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                if self.outlier_handling == 'random_in_distribution':
                    random_values = np.random.normal(loc=X_winsorized[col].mean(), scale=X_winsorized[col].std(), size=outliers_count)
                    random_values = np.clip(random_values, lower_bound, upper_bound)
                    X_winsorized.loc[outliers_mask, col] = random_values
                elif self.outlier_handling == 'median':
                    X_winsorized.loc[outliers_mask, col] = self.median_[col]
                elif self.outlier_handling == 'lower_bound_upper_bound':
                    X_winsorized.loc[X_winsorized[col] < lower_bound, col] = lower_bound
                    X_winsorized.loc[X_winsorized[col] > upper_bound, col] = upper_bound
        return X_winsorized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


    


class LogTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a log transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    columns : str
        The name of the column to apply the log transform to.
    domain_shift : float
        The value to be added to the column before applying the log transform.
        
    return
    ------
        transformed feature
    """
    def __init__(self, columns, domain_shift=0):
        self.columns = columns
        self.domain_shift = domain_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = np.log(X_copy[self.columns] + self.domain_shift)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    



class BoxCoxTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a Box-Cox transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    columns : str or list of str
        The name(s) of the column(s) to apply the transform to.
    domain_shift : float
        The value to be added to the column before applying the transform.
        
    Returns
    ------
        Transformed feature
    """
    def __init__(self, columns, domain_shift=0):
        self.columns = columns
        self.domain_shift = domain_shift
        self.lambdas_ = {}

    def fit(self, X, y=None):
        if isinstance(self.columns, str):
            self.columns = [self.columns]
        for col in self.columns:
            _, self.lambdas_[col] = boxcox(X[col] + self.domain_shift)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if isinstance(self.columns, str):
            self.columns = [self.columns]
        for col in self.columns:
            X_copy[col] = boxcox(X_copy[col] + self.domain_shift, lmbda=self.lambdas_[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


    

class YeoJohnsonTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a Yeo-Johnson transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    columns : list
        The name of the columns to apply the transform to.
    domain_shift : float
        The value to be added to the column before applying the transform.
    """
    def __init__(self, columns, domain_shift=0):
        self.columns = columns
        self.domain_shift = domain_shift
        self.transformer_ = None

    def fit(self, X, y=None):
        self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=False)
        self.transformer_.fit(X.loc[:, self.columns].values.reshape(-1, len(self.columns)) + self.domain_shift)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.columns] = self.transformer_.transform(X.loc[:, self.columns].values.reshape(-1, len(self.columns)) + self.domain_shift)
        return X_copy

    def fit_transform(self, X, y=None):
        self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=False)
        X_copy = X.copy()
        X_copy.loc[:, self.columns] = self.transformer_.fit_transform(X.loc[:, self.columns].values.reshape(-1, len(self.columns)) + self.domain_shift)
        return X_copy


    

    
class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to encode categorical columns using LabelEncoder.

    Parameters
    ----------
    columns : list of str
        The names of the columns to be encoded.

    return
    ------
        encoded feature
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

    
class OneHotEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply one-hot encoding to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    columns : list
        A list of column names to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified columns one-hot encoded.
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoder_ = None

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder_.fit(X.loc[:, self.columns])
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(self.encoder_.transform(X.loc[:, self.columns]), 
                                 index=X.index)
        X_copy = X.drop(columns=self.columns)
        return pd.concat([X_copy, X_encoded], axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


    
class OrdinalEncodeColumns(BaseEstimator, TransformerMixin):
    """
    Transformer class to perform ordinal encoding on specified columns of a Pandas DataFrame.

    Parameters
    ----------
    columns : list of str
        The names of the ordinal columns to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the ordinal columns encoded.
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoder = None
    
    def fit(self, X, y=None):
        ordinal_data = X[self.columns].values
        self.encoder = OrdinalEncoder()
        self.encoder.fit(ordinal_data)
        return self
    
    def transform(self, X):
        X_new = X.copy()
        ordinal_data = X_new[self.columns].values
        encoded_data = self.encoder.transform(ordinal_data)
        X_new[self.columns] = encoded_data
        return X_new
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    


class BinaryEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to encode categorical columns using BinaryEncoder.

    Parameters
    ----------
    columns : list of str
        The names of the columns to be encoded.

    return
    ------
        encoded feature
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            encoder = BinaryEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            encoded_cols = encoder.transform(X_copy[col])
            encoded_cols.columns = [f'{col}_binary_{i}' for i in range(encoded_cols.shape[1])]
            X_copy = pd.concat([X_copy, encoded_cols], axis=1)
            X_copy = X_copy.drop(col, axis=1)
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class StandardScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply standard scaling to.
    """
    def __init__(self, columns):
        self.columns = columns
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.columns] = self.scaler_.transform(X_copy.loc[:, self.columns])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.columns])
        return self.transform(X)
    
    
    
class MinMaxScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply min-max scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply min-max scaling to.
    """
    def __init__(self, columns):
        self.columns = columns
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler().fit(X.loc[:, self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.columns] = self.scaler_.transform(X_copy.loc[:, self.columns])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = MinMaxScaler().fit(X.loc[:, self.columns])
        return self.transform(X)