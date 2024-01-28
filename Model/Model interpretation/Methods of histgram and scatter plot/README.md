

## Methods of histgram and scatter plot

1.  Define functions `process_data`, `plot_histogram`, and `plot_scatter_chart`.

```python
## Global parameter tuning
import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum
import matplotlib.pyplot as plt

def process_data(keyword):
    # Find file names containing a specific keyword and remove the '.csv' extension
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_importance', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assume there's only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Set the 'cluster_name' column as the index and remove it after setting the index
    df.set_index('cluster_name', inplace=True)

    # Add a row and calculate the sum of each column
    df.loc['feature importance', :] = df.sum(axis=0)

    return df

def plot_histogram(df, title, bins_num=50):
    plt.figure(figsize=(10, 10))

    # Select data from the 'feature importance' row
    data = df.loc['feature importance', :]

    # Plot the histogram
    plt.hist(data, bins=bins_num, alpha=1, color='m')  # Set color to magenta
    plt.title(title)
    plt.xlabel('Total Feature Importance')
    plt.ylabel('Number of Features')

    # Save the image
    plt.savefig(f'{save_path}/{title}_histogram.png', dpi=1000)  # Set dpi to 1000 for png
    plt.savefig(f'{save_path}/{title}_histogram.pdf')  # Save as pdf, size 10x10 inches
    plt.close()  # Close the plot window

def plot_scatter_chart(df, title):
    plt.figure(figsize=(10, 10))

    # Sort the values of the 'feature importance' row and get the ranks
    data = df.loc['feature importance', :].sort_values(ascending=False)
    ranks = np.arange(1, len(data) + 1)

    # Calculate log10-transformed values
    log_values = np.log10(data)

    # Plot the scatter chart
    plt.scatter(ranks, log_values, color='m')  # Set color to magenta
    plt.title(title)
    plt.xlabel('Feature Rank')
    plt.ylabel('log10(Feature Importance)')

    # Save the image
    plt.savefig(f'{save_path}/{title}_scatter_chart.png', dpi=1000)  # Set dpi to 1000 for png
    plt.savefig(f'{save_path}/{title}_scatter_chart.pdf')  # Save as pdf, size 10x10 inches
    plt.close()  # Close the plot window
```

2. The input data file after processed by `Average feature importance calculation` step are located in each folders: RF, DNN, and DNN_ig. Subsequently, the data will be processed using the functions defined in step 1 to obtain histograms and scatter charts plot for both `feature_all` features and `eos` features. other individual feature type can also refer to  this processed methods to get corresponding output.

```python
# This section processes the results of Random Forest (RF). If you need to process results for DNN or DNN_ig,
# simply change the 'open_path' variable to the corresponding directory.
open_path = r"..\feature-representation-for-LLMs\Model\Model interpretation\Methods of histgram and scatter plot\data\DNN"
save_path = r"<your path to output>"
names = os.listdir(open_path)

# Read the target file based on the file keyword
df_train = process_data('train')

# Plot histograms and scatter charts for df_train
plot_histogram(df_train, 'Train Feature all')
plot_scatter_chart(df_train, 'Train Feature all')

# Plot histograms and scatter charts for eos features separately
df_train_eos = df_train[[col for col in df_train.columns if re.match(r'ESM2_eos\d+', col)]]  #If you need to view other features, simply modify 'ESM2_eos\d+' to the corresponding feature, such as 'ESM2_cls\d+'
plot_histogram(df_train_eos, 'Train Feature eos')
plot_scatter_chart(df_train_eos, 'Train Feature eos')
```
