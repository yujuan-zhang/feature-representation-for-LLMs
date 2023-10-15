

1. Global Configuration and Workspace Setup for DL Model Interpretation (Deep SHAP Analysis)

```python
import os
import pandas as pd
import numpy as np
import re
from functools import reduce
from protloc_mex1.SHAP_plus import SHAP_importance_sum
from sklearn.preprocessing import MinMaxScaler
from protloc_mex1.SHAP_plus import FeaturePlot

## Global Configuration
open_path = r"D:\Working\Team\Luo_Zeyu\Sub_articles\BIB\Revision_processing\11_shap_interpretable\2_dl_importance\data\in"
save_path = r"D:\Working\Team\Luo_Zeyu\Sub_articles\BIB\Revision_processing\11_shap_interpretable\2_dl_importance\output\swiss"

type_name = ['swiss']
gene_ID = "Entry"
depleted_ID_len = 2  ## Discard columns only in the last few columns
data_input_name = ['train', 'test']
feature_single_import_c = False

## Function Area

## Workspace

if os.path.isdir(save_path + "/" + type_name[0]):
    pass
else:
    os.makedirs(save_path + "/" + type_name[0])
names = os.listdir(open_path)
pattern = re.compile('.csv')
names = list(map(lambda x: pattern.sub('', x), names))
```

2. Calculating and Saving Deep SHAP Importance Summaries

```python
# Read the 'SHAP_base_probablity.xlsx' file
type_data_all = list(map(lambda x: pd.read_csv(open_path + "/" + x + ".csv", header=0, sep=",", index_col=gene_ID), names))
type_data_all = dict(zip(names, type_data_all))

# Define an 'all' class
all_data = SHAP_importance_sum()
all_data.shapley_data_all = type_data_all

# Calculate for all_data
SHAP_importance_sum_claulate_outcome = all_data.IG_importance_sum_claulate_process(depleted_ID_len=2, file_name=type_name[0] + '_')

for value in type_data_all.keys():
    SHAP_importance_sum_claulate_outcome['type_data_shap_sum_all_outcome'][value].to_csv(
        save_path + "/" + type_name[0] + "/" + value + ".csv")

SHAP_importance_sum_claulate_outcome['shap_feature_importance'].to_csv(
    save_path + "/" + type_name[0] + "/shap_feature_importance.csv", index_label='feature_name')

# Save feature weights one by one
i_data_name = str(type_name)

data_shap_importance_T = SHAP_importance_sum_claulate_outcome['shap_feature_importance_T'].loc[
    all_data.str_contain(i_data_name, list(SHAP_importance_sum_claulate_outcome['shap_feature_importance_T'].index))
]
data_shap_importance_T.to_csv(
    save_path + "/" + type_name[0] + "/shap_feature_" + i_data_name + "_importance_T.csv", index_label='cluster_name')
```
