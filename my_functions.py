import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

def format_column_names(df: pd.DataFrame, column_renames: Dict[str, str] = {} ) -> pd.DataFrame:
    '''
    This function takes a DataFrame and 
    (1) formats column names to lower case and removes white spaces and
    (2) renames columns according to the input dictionary.
    
    Inputs:
    df: input DataFrame
    column_renames: Dictionary with column renaming
    
    Outputs:
    formatted DataFrame
    '''
    df_formatted = df.copy()
    df_formatted.columns = [col.lower().replace(' ','_') for col in df.columns] # remove white spaces & lower cased
    df_formatted.rename(columns = column_renames, inplace = True) # rename columns according to dictionary
    
    return df_formatted


def clean_column_by_replacing_string(df: pd.DataFrame, column:str, replacements: list) -> pd.DataFrame:
    '''
    This function takes a Dataframe and replaces the strings 
    in the input replacements to the specified column.
    
    Inputs:
    df: input DataFrame
    column: column to apply transformations
    replacements: list of lists with replacements 
        [[old_value1, new_value1],[old_value2, new_value2],...]
        
    Output:
    pandas DataFrame with the clean column
    '''
    df1 = df.copy()
    
    for item in replacements:
        df1[column] = df1[column].str.replace(item[0],item[1]) # replace items in column
        
    return df1


def reassign_column_data_type(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    '''
    This function takes a DataFrame and reassigns data types as specified in the columns parameter.
    
    Input: 
    df: pandas DataFrame
    columns: Dictionary with column and data type assignment
    
    Output:
    Dataframe with data type reassign columns
    '''
    
    for key, value in columns.items():
        df[key] = df[key].astype(value)

    return df


def remove_duplicate_and_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function removes duplicate rows and rows with all the columns empty.
    
    Input:
    df: input DataFrame
    
    Output:
    df: output DataFrame
    '''
    df1 = df.copy()
    
    df1.drop_duplicates(inplace = True) # drop all duplicate rows
    df1.dropna(inplace = True) # drop all empty rows
    
    return df1


def clean_and_format_data(df: pd.DataFrame, 
                          cols_to_rename: Dict[str, str], 
                          cols_to_replace: Dict[str, list],
                          cols_to_reassign_datatype: Dict[str,str]) -> pd.DataFrame:
    '''
    This function executes all the cleaning functions on the input df in 4 steps:
    (1) formats and renames column names
    (2) removes duplicate and empty rows
    (3) cleans the gender column
    (4) cleans number_of_complaints column
    (5) applies character replacements
    (6) reassigns data types
    
    Inputs:
    df: input Dataframe
    cols_to_rename: Dictionary with columns to rename
    cols_to_replace: Dictionary with columns to apply replacements
    cols_to_reassign_datatype: Dictionary with columns to reassign datatype
    
    Output:
    pandas DataFrame with clean and formatted data
    '''
    df1 = df.copy()
    
    df1 = format_columns(df1,cols_to_rename)# format & rename columns
    df1 = remove_duplicate_and_empty_rows(df1)# remove duplicate and empty rows
    df1 = clean_gender_column(df1)# clean gender column
    df1 = clean_number_of_complains_column(df1)# clean number_of_complain_column
    for key, value in cols_to_replace.items():
        df1 = clean_column_by_replacing_string(df1, key, value)# replace cleaning
    df1 = reassign_column_data_type(df1, cols_to_reassign_datatype)# reassign data types
        
    return df1


def plot_numeric_columns(df, plot_type='histogram'):
    '''
    This function generates plots for all numerical columns of the input DataFrame.

    Inputs:
    df: input DataFrame
    plot_type: type of plot, 'histogram' or 'boxplot'

    Output:
    A plot for each column

    '''
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    num_plots = len(numerical_columns)
    num_rows = int(np.ceil(num_plots / 2))
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)

    for i, column in enumerate(numerical_columns):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        if plot_type == 'histogram':
            ax.hist(df[column], bins=20, edgecolor='black')
            ax.set_ylabel('Frequency')
        elif plot_type == 'boxplot':
            ax.boxplot(df[column], vert = False)

        ax.set_xlabel(column)
        ax.set_title(f'{plot_type.capitalize()} of {column}')

    # Remove empty subplots
    if num_plots < num_rows * num_cols:
        if num_rows > 1:
            for i in range(num_plots, num_rows * num_cols):
                fig.delaxes(axes[i // num_cols, i % num_cols])
        else:
            for i in range(num_plots, num_rows * num_cols):
                fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def calculate_skew(df, interval=(-2, 2)):
    '''
    This function generates a table  to analyse the skewness of the numerical columns of a DataFrame.

    Inputs:
    df: input DataFrame
    interval: (min,max) skewness interval

    Output:
    DataFrame with 3 columns:
    (1) column_name
    (2) skew value
    (3) Boolean for skew value out of interval
    '''
    numerical_cols = df.select_dtypes(include='number').columns
    skew_values = []
    for col in numerical_cols:
        skew = df[col].skew()
        is_outside_interval = not (interval[0] <= skew <= interval[1])
        skew_values.append((col, skew, is_outside_interval))

    result_df = pd.DataFrame(skew_values, columns=['Column', 'Skew', 'Outside Interval'])
    return result_df


def select_features_for_linear_models_based_on_correlation(df: pd.DataFrame, y: str, threshold=0.75) -> list:
    '''
    Input
    df: pd.DataFrame
    y: column to be predicted

    Output
    list: list of strings
    '''

    df2 = df.copy()

    correlation_matrix = df2.corr(numeric_only=True)#.reset_index()

    list_of_selected_columns = list(correlation_matrix[y].loc[abs(correlation_matrix[y])>=threshold].index)

    return list_of_selected_columns


def plot_corr_matrix(df):
    # Calculate the correlation matrix
    correlation_matrix = df.corr(numeric_only = True)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the correlation matrix as a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='magma', linewidths=0.5, linecolor='lightgray',
                cbar=True, square=True, xticklabels=True, yticklabels=True, ax=ax)

    # Customize the plot
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def remove_outliers(df: pd.DataFrame, columns: list, lower_percentile: float=25, upper_percentile: float =75):
    '''
    This function removes the outliers of specified columns of a 
    dataset that lie out of the limits provided as inputs.
    
    Input:
    df: input DataFrame
    columns: list of columns to remove outliers
    lower_percentile: lower limit percentile to remove outliers
    upper_percentile: upper limit percentile to remove outliers
    
    Output:
    DataFrame with removed outliers
    
    '''
    
    filtered_data = df.copy()
    
    for column in columns:
        lower_limit = np.percentile(filtered_data[column], lower_percentile)
        upper_limit = np.percentile(filtered_data[column], upper_percentile)
        iqr = upper_limit - lower_limit
        lower_whisker = lower_limit - 1.5 * iqr
        upper_whisker = upper_limit + 1.5 * iqr

        filtered_data = filtered_data[(filtered_data[column] > lower_whisker) & (filtered_data[column] < upper_whisker)]

    return filtered_data