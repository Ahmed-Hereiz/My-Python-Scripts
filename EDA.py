import colorsys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd


class ColorPalette:
    """
    A class for creating color palettes.

    Attributes:
        hue (float): The hue of the base color to use for the palette.
        saturation (float): The saturation of the base color to use for the palette.
        value_start (float): The starting value of the base color to use for the palette.

    Methods:
        create_sequential_palette(num_colors):
            Creates a sequential color palette with the specified number of colors,
            using a base color with the specified hue, saturation, and value_start.
            
        create_diverging_palette(num_colors,value):
            Creates a diverging color palette with the specified number of colors,
            using a base color with the start hue, end hue, saturation, and value.
        
        get_color():
            Given a list of colors, returns the indexed color in the list.
    """
    
    def __init__(self, hue=0.5, saturation=0.8, value_start=0.4, start_hue=0.6, end_hue=0.1):
        self.hue = hue
        self.saturation = saturation
        self.value_start = value_start
        self.start_hue = start_hue
        self.end_hue = end_hue
        
    def create_sequential_palette(self, num_colors):
        """
        Creates a sequential color palette with the specified number of colors,
        using a base color with the specified hue, saturation, and value_start.
        """
        colors = [colorsys.hsv_to_rgb(self.hue,
                                      self.saturation,
                                      self.value_start + (i/num_colors)*(1-self.value_start)) for i in range(num_colors)]
        
        return sns.color_palette(colors)
    
    def create_diverging_palette(self, num_colors, value=0.9):
        """
        Creates a diverging color palette with the specified number of colors,
        ranging from start_hue to end_hue.
        """
        colors = []
        for i in range(num_colors):
            hue = self.start_hue + (i / (num_colors - 1)) * (self.end_hue - self.start_hue)
            r, g, b = colorsys.hsv_to_rgb(hue, self.saturation, value)
            colors.append((r, g, b))

        return sns.color_palette(colors)

    def get_color(self, palette, color_index=-2):
        """
        Given a list of colors, returns the last color in the list.
        """
        return palette[color_index]
        

class DataVisualizer:
    """
    The DataVisualizer class is designed to help visualize and explore data.
    It contains Eight methods : 
    plot_distribution, plot_feature_by_target, plot_bar, plot_correlation,
    plot_missing, plot_skewness, plot_pie, plot_time_series
    that can be used to generate various types of plots to understand the data better.
    
    Attributes:

        data (pd.DataFrame): The input data frame to be visualized
        
    Methods:

        plot_distribution(cols=None, palette='Blues', color='b', hue=None, feature_type='both'):
        
            Generates distribution plots of the selected columns of data frame.
            It plots either a boxplot and KDE plot or a countplot depending
            on the number of unique values in the selected column. 
            
        Parameters:   
        - cols (list, str or None): Columns to be plotted, default is None, which plots all columns.
        - palette (seaborn color palette): color palette to be used in the plot, default is 'Blues'
        - color (seaborn color): color for the KDE plot or countplot, default is 'b'
        - hue (str): Column to split the plot into subplots based on the unique values, default is None
        - feature_type (str): Column type to be selected for plotting, default is 'both', other values can be 'numeric' or 'categorical'
    
    
    
        plot_feature_by_target(target, feature_type='both', color='b', height=5, width=5, cols=None, plotdim=None):
        
            Generates scatterplot or barplot for the selected columns of the data frame
            with respect to a target variable. 
            
        Parameters: 
        - target (str): Target variable to be plotted against the selected columns.
        - feature_type (str): Column type to be selected for plotting, default is 'both', other values can be 'numeric' or 'categorical'
        - color (seaborn color): color to be used in the plot, default is 'b'
        - height (float): height of each subplot, default is 5
        - width (float): width of each subplot, default is 5
        - cols (list, str or None): Columns to be plotted, default is None, which plots all columns.
        - plotdim (list or tuple): shape of the plot, default is None,
        which generates a square-shaped plot. Allowed values are list or tuple with shape = (,2).
        
        
        
        plot_bar(self, target, cols=None, palette='Blues', hue=None):
        
            Generate barplot and countplot for the selected categorical columns of the data frame
            with respect to a target variable.

        Parameters:
        - target (str): Target variable to be plotted against the selected columns.
        - cols (list, str or None): Columns to be plotted, default is None, which plots all categorical columns.
        - palette (seaborn color palette or str): color palette to be used in the plot, default is 'Blues'
        - hue (str): Column to group by when plotting, default is None.
        
        
        
        plot_correlation(cols=None, palette='Blues', width=16, height=18):
        
            Generates a correlation matrix heatmap plot for the selected numeric columns
            of the data frame.
        
        Parameters:  
        - cols (list, str or None): Columns to be plotted, default is None, which plots all numeric columns.
        - target(list, str or None): Columns as target, target columns for correlation, default None.
        - palette (seaborn color palette): color palette to be used in the plot, default is 'Blues'
        - figsize (tuple): size of plot.
        - cbar(bool): adding cbar, default is False
        - fmt(str): format string for the annotations, default is '.2g'.
        - ascending : to sort corr matrix only of target is not None, defult is False
        
        
        
        plot_missing(palette='Blues', figsize=(16, 16), cbar=False, fmt='.5g'):
        
            Generates missing data sum heatmap plot. 
            
        Parameters:
        - palette (seaborn color palette): color palette to be used in the plot, default is 'Blues'
        - figsize (tuple): size of the plot, default is (16, 16)
        - cbar (bool): whether to add a colorbar to the plot, default is False
        - fmt (str): format string for the annotations, default is '.5g'    
        
        
        
        plot_skewness(cols, palette='Blues', figsize=(18, 8), cbar=False, fmt='.2g',sort_ascending=True):
        
            Generates a heatmap of skewness values for the selected columns of the data frame.
            
        Parameters:
        - cols (list or str): Columns to be plotted.
        - palette (seaborn color palette): color palette to be used in the plot, default is 'Blues'.
        - figsize (tuple): size of plot, default is (18,8).
        - cbar(bool): adding color bar, default is False.
        - fmt(str): number of decimal places, default is '.2g'.
        - sort_ascending(bool): sorting the skewness values in ascending order, default is True.
        
        
        
        plot_pie(self, palette='Blues', cols=None, fig_size=(15, None), plotdim=None):
        
            Generates pie chart plots for categorical features with up to 6 unique values.
    
        Parameters:
        - palette (str or seaborn color palette): color palette to be used in the plots. Default is 'Blues'.
        - cols (list, str, or None): columns to be plotted. Default is None, which plots all columns.
        - fig_size (tuple of float): size of the figure. Default is (15, None).
    
    
    
       plot_time_series(self, time_col, target_cols, format="%Y-%m-%d %H:%M", color="b", height=5, width=5)
        
           Generates time series plot for selected columns of a data frame.
        
       Parameters:
       - data (pandas DataFrame): Input data frame containing the data to be plotted.
       - time_col (str): Name of the column containing the time stamps.
       - target_cols (list or str): Columns to be plotted.
       - format (str): Format of the time stamps in the data, default is '%Y-%m-%d %H:%M'.
       - color (str): Color of the plotted line, default is 'b'.
       - height (float): Height of the plot in inches, default is 5.
       - width (float): Width of the plot in inches, default is 5. 
    """
    
    def __init__(self, data):
        self.data = data
    
    def plot_distribution(self, cols=None, palette='Blues', color='b', hue=None, feature_type='both'):
        """
        How to use : 
        ------------
        > dv = DataVisualizer(data=df)
        > dv.plot_distribution()
        """
        if cols is None:
            cols = self.data.columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")

        if feature_type == 'numeric':
            cols = self.data[cols].select_dtypes(include=np.number).columns
        elif feature_type == 'categorical':
            cols = self.data[cols].select_dtypes(include=['category', 'object']).columns
        elif feature_type != 'both':
            raise ValueError("Invalid value for 'feature_type'. Allowed values are 'numeric', 'categorical', and 'both'.")

        for feature in cols:
            if len(self.data[feature].unique()) > 20:
                fig, (ax_box, ax_kde) = plt.subplots(ncols=2, sharex=True, figsize=(20, 3))
                if type(palette) == str:
                    cp = palette
                else:
                    cp = palette.create_sequential_palette(num_colors=len(self.data[hue].unique())) if hue else palette.create_sequential_palette(num_colors=len(self.data[feature].unique()))
                sns.boxplot(x=feature, data=self.data, ax=ax_box, linewidth=1.0, palette=cp, hue=hue)
                sns.kdeplot(x=feature, data=self.data, ax=ax_kde, fill=True, palette=cp, hue=hue, color=color)
            else:
                fig, ax_count = plt.subplots(ncols=1, sharex=True, figsize=(20, 4))
                if type(palette) == str:
                    cp = palette
                else:
                    cp = palette.create_sequential_palette(num_colors=len(self.data[feature].unique()))
                sns.countplot(x=feature, data=self.data, ax=ax_count, linewidth=1.0, palette=cp, hue=hue)
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.show()


    def plot_feature_by_target(self, target, feature_type='both', color='b', height=5, width=5, cols=None, plotdim=None):
        """
        How to use : 
        ------------
        > dv = DataVisualizer(data=df)
        > dv.plot_feature_by_target(target=target_col)
        """
        if cols is None:
            cols = self.data.columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")

        if feature_type == 'numeric':
            cols = self.data[cols].select_dtypes(include=np.number).columns
        elif feature_type == 'categorical':
            cols = self.data[cols].select_dtypes(include=['category', 'object']).columns
        elif feature_type != 'both':
            raise ValueError("Invalid value for 'feature_type'. Allowed values are 'numeric', 'categorical', and 'both'.")

        n = len(cols)
        if plotdim == None:
            nrows = int(np.ceil(np.sqrt(n)))
            ncols = int(np.ceil(n / nrows))
        elif plotdim != None and len(plotdim) == 2:
            nrows = plotdim[0]
            ncols = plotdim[1]
        else:
            raise ValueError("Invalid value for 'plotdims'. Allowed values are None,list,tuple with shape = (,2).")
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*width, nrows*height))
        counter = 0
        for i in cols:
            sub = fig.add_subplot(nrows, ncols, counter+1)
            if feature_type == 'numeric':
                g = sns.scatterplot(x=i, y=target, data=self.data, color=color)
            elif feature_type == 'categorical':
                g = sns.barplot(x=i, y=target, data=self.data, color=color)
            else:
                if self.data[i].dtype != 'object':
                    g = sns.scatterplot(x=i, y=target, data=self.data, color=color)
                elif pd.api.types.is_numeric_dtype(self.data[target].dtype):
                    g = sns.barplot(x=i, y=target, data=self.data, color=color)
                else:
                    continue
            counter = counter + 1
            if counter >= n:
                break

        for i in range(nrows * ncols):
            ax.flatten()[i].set_visible(False)

        plt.tight_layout()
        
        
    def plot_bar(self, target, cols=None, palette='Blues', hue=None):
        """
        How to use : 
        ------------
        > dv = DataVisualizer(data=df)
        > dv.plot_bar(target=target_col)
        """
        if cols is None:
            cols = self.data.columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")

        cols = self.data[cols].select_dtypes(include=['category', 'object']).columns
        
        if self.data[target].dtype not in ['category', 'object']:
            for feature in cols:
                fig, (ax_bar, ax_count) = plt.subplots(ncols=2, sharex=True, figsize=(20, 3))
                if type(palette) == str:
                    cp = palette
                else:
                    cp = palette.create_sequential_palette(num_colors=len(self.data[hue].unique())) if hue else palette.create_sequential_palette(num_colors=len(self.data[feature].unique()))

                sns.barplot(x=feature, y=target, data=self.data, ax=ax_bar, linewidth=1.0, palette=cp, hue=hue)
                sns.countplot(x=feature, data=self.data, ax=ax_count, fill=True, palette=cp, hue=hue)

            plt.xlabel(feature)


    def plot_correlation(self, cols=None, target=None, palette='Blues', figsize=(16, 16), cbar=False, fmt='.2g', ascending=False):
        """
        How to use : 
        ------------
        > dv = DataVisualizer(data=df)
        > dv.plot_correlation()
        """
        if cols is None:
            cols = self.data.columns
            cols = self.data[cols].select_dtypes(include=np.number).columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")
        
        if target == None:
            cols_corr = self.data[cols].corr()
        else:
            if isinstance(target, str):
                target = [target]
            cols_corr = pd.DataFrame(self.data.corr()[target].loc[cols])
        if target is not None:
            cols_corr = cols_corr.sort_values(by=target[0], ascending=ascending)
        if type(palette) == str:
            cp = palette
        else:
            cp = palette.create_sequential_palette(num_colors=len(cols))
        plt.figure(figsize=figsize)
        sns.heatmap(cols_corr,annot=True,cmap=cp,cbar=cbar,fmt=fmt)
        plt.title('Data Correlation')
    

    def plot_missing(self, palette='Blues', figsize=(16, 16), cbar=False, fmt='.5g'):
        """
        How to use : 
        ------------
        > dv = DataVisualizer(data=df)
        > dv.plot_missing()
        """
        if type(palette) == str:
            cp = palette
        else:
            cp = palette.create_sequential_palette(num_colors=len(self.data.columns))
        plt.figure(figsize=figsize)
        sns.heatmap(pd.DataFrame(self.data.isna().sum()), cmap=cp, cbar=cbar, annot=True, fmt=fmt)
        plt.title("Missing Values")
        
        
    def plot_skewness(self, cols, palette='Blues', figsize=(18, 8), cbar=False, fmt='.2g',sort_ascending=True):
        if isinstance(cols, str):
            cols = [cols]
        plt.figure(figsize=figsize)
        
        if type(palette) == str:
            cp = palette
        else:
            cp = palette.create_sequential_palette(num_colors=len(cols))
        sns.heatmap(pd.DataFrame(self.data[cols].skew().sort_values(ascending=sort_ascending)),
                    cmap=cp, cbar=cbar, annot=True, fmt=fmt)
        plt.title("Skewness")
        
        
        
    def plot_pie(self, palette='Blues', cols=None, fig_size=(15, None)):
        if cols is None:
            cols = self.data.columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")
        num_cols = len([x for x in cols if len(self.data[x].unique()) <= 6])
        if num_cols == 0:
            print("No categorical features with <= 6 unique values found!")
            return
        nrows = int(np.ceil(num_cols/3))
        if fig_size[1] is None:
            fig_size = (fig_size[0], 5 * nrows)
        fig, ax = plt.subplots(nrows=nrows, ncols=min(num_cols, 3), figsize=fig_size)
        counter = 0
        for feature in cols:
            if len(self.data[feature].unique()) <= 6:
                if isinstance(palette, str):
                    cp = sns.color_palette(palette, n_colors=len(self.data[feature].unique()))
                else:
                    cp = palette.create_sequential_palette(num_colors=len(self.data[feature].unique()))
                if num_cols > 3:
                    ax[counter//3][counter%3].pie(self.data[feature].value_counts(), labels=self.data[feature].unique(), autopct='%1.1f%%', colors=cp)
                else:
                    ax[counter].pie(self.data[feature].value_counts(), labels=self.data[feature].unique(), autopct='%1.1f%%', colors=cp)
                if num_cols > 3:
                    ax[counter//3][counter%3].set_title(feature)
                else:
                    ax[counter].set_title(feature)
                counter += 1
                if counter >= num_cols:
                    break

        for i in range(counter, nrows * min(num_cols, 3)):
            if num_cols > 3:
                fig.delaxes(ax[i // 3][i % 3])
            else:
                fig.delaxes(ax[i])

        plt.show()
        

    def plot_time_series(self, time_col, target_cols, format="%Y-%m-%d %H:%M", color="b", height=5, width=5):        
        if not isinstance(self.data[time_col], pd.core.series.Series):
            self.data[time_col] = pd.to_datetime(self.data[time_col], format=format)
        self.data.set_index(time_col)
        for col in target_cols:
            if col == time_col:
                continue
            plt.figure(figsize=(width, height))
            plt.plot(self.data[col], color=color)
            plt.xlabel(time_col)
            plt.ylabel(col)
            plt.title("Time series plot of {}".format(col))
            plt.show()
        
        
        
class DataExplorer:
    """
    A class for exploring the data and generating summary statistics.

    Attributes
    ----------
    data : pandas.DataFrame
        The input data to be explored.

    Methods
    -------
    explore_unique_number(cols=None, feature_type='both')
        Prints the number of unique values in the specified columns of the data.
    """
    def __init__(self, data):
        self.data = data
        
    def explore_unique_number(self, cols=None, feature_type='both'):
        if cols is None:
            cols = self.data.columns
        elif isinstance(cols, str):
            cols = [cols]
        elif not isinstance(cols, list):
            raise ValueError("Invalid value for 'cols'. Allowed values are None, str, and list.")
            
        if feature_type == 'numeric':
            cols = self.data[cols].select_dtypes(include=np.number).columns
        elif feature_type == 'categorical':
            cols = self.data[cols].select_dtypes(include=['category', 'object']).columns
        elif feature_type != 'both':
            raise ValueError("Invalid value for 'feature_type'. Allowed values are 'numeric', 'categorical', and 'both'.")
            
        for feature in cols:
            print('\nNumber of unique data in the "{}" is : '.format(feature),len(self.data[feature].unique()))
