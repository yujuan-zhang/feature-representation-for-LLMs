#### 1. Summary of Data Processing and Visualization Functions

We defines a set of functions for processing and visualizing data from CSV files. These functions include:

**process_data(keyword)**: This function filters and loads CSV files containing a specific keyword, setting the 'cluster_name' column as the index.

**df_col_choose(df_input, pattern_input)**: This function selects columns from the input specified subcellular localization type.

**extract_index(dict_input, extract)**: It extracts specific rows from specified subcellular localization type.

**adjust_scales_with_log(x)**: A function for scaling values using a log transformation.

**compare_with_groups_and_log_scale(df, compare_group, group_indices, filename, save_path, figure_size)**: This function performs group comparisons, log-scales values, and generates visualizations, including box plots and p-values.

These functions are designed to facilitate data processing and visualization tasks.

```python
import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# This function processes data by filtering filenames containing a specific keyword and loading CSV files.
def process_data(keyword):
    # Find filenames containing the specific keyword and remove the '.csv' suffix
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_importance', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assuming there is only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Use pandas to read the CSV file
    df = pd.read_csv(file_path)

    # Set the 'cluster_name' column as the index and remove it after setting the index
    df.set_index('cluster_name', inplace=True)

    return df


# This function chooses columns from the input DataFrame based on a pattern and returns a dictionary of separated DataFrames.
def df_col_choose(df_input, pattern_input):
    # Create a dictionary to store split DataFrames
    dict_col = {}
    for pattern in pattern_input:
        # Get all column names that match the pattern
        columns = [col for col in df_input.columns if re.match(pattern, col)]

        # If there are matching column names
        if columns:
            # Extract these columns from df_input and create a new DataFrame
            dict_col[re.sub(r'\\\d\+', '', pattern)] = df_input[columns]
    return dict_col


# This function extracts specific rows from DataFrames in the input dictionary and combines them into a single DataFrame.
def extract_index(dict_input, extract):
    # Create an empty list of DataFrames
    dfs = []

    # Iterate through each item in the dictionary
    for key, value in dict_input.items():
        # Extract the desired part
        extracted = value[value.index.str.contains(extract)].T

        # Add a new column to store the key
        extracted['group'] = key

        # Add the result to the list
        dfs.append(extracted)

    # Concatenate all the DataFrames together using the concat function
    df = pd.concat(dfs, axis=0)

    # Rename the column to the desired name
    df.columns = ['value', 'group']

    # Ensure that the 'group' column is a categorical variable for correct identification during plotting
    df['group'] = df['group'].astype('category')
    return df


# This function scales values using log transformation.
def adjust_scales_with_log(x):
    return np.log(x)


# This function compares groups and log-scales the values for visualization.
def compare_with_groups_and_log_scale(df, compare_group, group_indices, filename, save_path, figure_size):
    # Preprocess: apply log transformation to 'value' and sort 'group'
    df['log_value'] = df['value'].apply(adjust_scales_with_log)
    df['group'] = df['group'].astype(pd.CategoricalDtype(categories=group_indices, ordered=True))

    # Set the figure size
    plt.figure(figsize=figure_size)

    # Create a box plot
    palette = {group: color for group, color in zip(df['group'].unique(), sns.color_palette("colorblind", len(df['group'].unique())))}
    ax=sns.swarmplot(x='group', y='log_value', data=df, order=group_indices, palette=palette,size=2)

    # Calculate and add mean values to the chart
    group_means = df.groupby('group')['value'].mean()
    for i, mean in enumerate(group_means):
        ax.text(i, np.log(max(mean, 1)), f"Mean: {mean:.6f}", 
                     horizontalalignment='center', color='black', weight='semibold')

    # Calculate and add comparison lines and corresponding p-values
    values_1 = df[df['group'] == compare_group]['value']
    y_max = df['log_value'].max()
    offset = (y_max - df['log_value'].min()) * 0.1  
    for i, group in enumerate(group_indices):
        if group != compare_group:
            values_2 = df[df['group'] == group]['value']
            _, p_value = mannwhitneyu(values_1, values_2)
            y = max(np.log(max(group_means[[compare_group, group]])), 1) + i * offset  
            line_x = [i, group_indices.index(compare_group)]
            line_x.sort()
            plt.plot(line_x, [y, y], color='black')
            plt.text((line_x[0] + line_x[1]) / 2, y, f"p = {p_value:.2e}",
                     horizontalalignment='center', color='black', weight='semibold')
    plt.tight_layout()

    # If specified, save the plot as a PDF file
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename + '.pdf'), format='pdf')
```

#### 2. Data Processing and Visualization for ESM2 Features

This code performs data processing and visualization for ESM2 features using SHAP values. It loads data from "shap_feature_swiss_importance_T.csv" data produced by the "Summary of Data Processing and Visualization Functions" step, extracts `"Mitochondrion"`, and generates box plots with log-scaled values to compare different groups. The results are saved in the output directory.  For other datasets (TrEMBL datasets), please follow the steps(`SHAP` or `IG` step -> `Summary of Data Processing and Visualization Functions` step).

```python
##Import necessary libraries and classes
import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

##Global Hyperparameter Tuning
open_path=r"<The path to the results of the Average Feature Importance Calculation steps>"
save_path=r"<your save path>"
names=os.listdir(open_path)

##Work space
df_train = process_data('swiss')

# Define Column Name Pattern List
patterns = ['ESM2_cls\d+','ESM2_segment0_mean\d+', 'ESM2_mean\d+','ESM2_eos\d+', ]
dict_train=df_col_choose(df_input=df_train ,pattern_input=patterns)   
extract = "Mitochondrion" # The required subcellular localization type.

df_train=extract_index(dict_train,extract)

compare_with_groups_and_log_scale(df_train, 'ESM2_eos', 
        ['ESM2_cls', 'ESM2_segment0_mean', 'ESM2_mean','ESM2_eos' ],f'swiss_{extract}_boxplot',save_path,(10,10))
```
