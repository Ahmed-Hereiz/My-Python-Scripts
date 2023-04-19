# My-Python-Scripts

## In this repository Iam making many python scripts which helps me while making any Machine Learning project

## Requirements :
> - Python 3.6 or later
> - `scikit-learn`
> - `numpy`
> - `pandas`
> - `SciPy`
> - `matplotlib`
> - `seaborn`
> - `Tensorflow`

## Scripts:
> - EDA
> - Test
> - Preprocessing
> - MachineLearning
> - ImageDataProcessor


<br><br>

<b><hr style="border-top: 4px solid black;"/></b>

# EDA :

##### Contains three classes for now which are :  (ColorPalette, DataVisualizer, DataExplorer)

# ColorPalette Class

A class for creating color palettes.

- `create_sequential_palette(num_colors)`: Creates a sequential color palette with the specified number of colors, using a base color with the specified hue, saturation, and value_start.
- `create_diverging_palette(num_colors, value)`: Creates a diverging color palette with the specified number of colors, ranging from start_hue to end_hue.
- `get_color()`: Given a list of colors, returns the last color in the list.


# DataVisualizer

A class for Visualizing data.

- `plot_distribution`: Generates distribution plots (boxplot and KDE or countplot) of selected columns of a data frame based on the number of unique values in the selected column.

- `plot_feature_by_target`: Generates scatterplot or barplot for selected columns of a data frame with respect to a target variable.

- `plot_bar`: Generates barplot and countplot for selected categorical columns of a data frame with respect to a target variable.

- `plot_correlation`: Generates a correlation matrix heatmap plot for selected numeric columns of a data frame.

- `plot_missing`: Generates missing data sum heatmap plot.

- `plot_skewness`: Generates a heatmap of skewness values for selected columns of a data frame.

- `plot_pie`: Generates pie chart plots for categorical features with up to 6 unique values.

- `plot_time_series` : Generates time series plot of all features.


# DataExplorer

A class for Exploring data.

- `explore_unique_number`: Prints the number of unique values in the specified columns of the data.


<br><br>

<b><hr style="border-top: 4px solid black;"/></b>


# Test :

## Contains one class for now which is :<br>DataChecker

# DataChecker 

A class to check the shapes of the training and test sets, and to check for negative values and NaN values in the target and feature variables.

- check_shapes(): Checks if the shapes of the training and test sets match and if the number of features and targets match.

- check_negative_values_y(): Checks if there are negative values in the target variables of the training and test sets.

- check_nan_values_X(): Checks if there are NaN values in the feature variables of the training and test sets.

- check_nan_values_y(): Checks if there are NaN values in the target variable of the training and test sets.

<br><br>

<b><hr style="border-top: 4px solid black;"/></b>

# Preprocessing :

## this script is usefull for making sklearn pipelines it makes it too much easier when doing the pipelines, you just use the classes you need and columns you need in it then put this classes in only 1 sklearn pipeline and in 1 step data will change into raw data ready for applying the machine learning model on it !!!

(This project contains a collection of Python classes designed to simplify the creation of data preprocessing pipelines in `scikit-learn`. These transformers can be used to perform common data preprocessing tasks such as selecting columns, imputing missing values, encoding categorical variables, and scaling numeric features)

## Transformer Classes

### ColumnSelector

A transformer class to select specified columns from a Pandas DataFrame.

### ArithmeticColumnsTransformer

A transformer class to perform arithmetic operations on specified columns in a Pandas DataFrame.

### DataFrameImputer

A transformer class to impute missing values in a Pandas DataFrame.

### DropColumnsTransformer

A transformer class to drop specified columns from a Pandas DataFrame.

### WinsorizationImpute

A transformer class to impute missing values in a Pandas DataFrame using Winsorization.

### LogTransform

A transformer class to apply a log transform to a specified column in a Pandas DataFrame.

### BoxCoxTransform

A transformer class to apply a Box-Cox transform to a specified column in a Pandas DataFrame.

### YeoJohnsonTransform

A transformer class to apply a Yeo-Johnson transform to a specified column in a Pandas DataFrame.

### LabelEncodeColumns

A transformer class to label encode specified columns in a Pandas DataFrame.

### OneHotEncodeColumns

A transformer class to one-hot encode specified columns in a Pandas DataFrame.

### OrdinalEncodeColumns

A transformer class to ordinal encode specified columns in a Pandas DataFrame.

### BinaryEncodeColumns

A transformer class to binary encode specified columns in a Pandas DataFrame.

### StandardScaleTransform

A transformer class to standardize numeric features in a Pandas DataFrame.

### MinMaxScaleTransform

A transformer class to scale numeric features to a specified range in a Pandas DataFrame.

### DateTimeTranformer

A transformer class for extracting days, months, years from a time serires


<b><hr style="border-top: 4px solid black;"/></b>


# ImageDataProcessor :
