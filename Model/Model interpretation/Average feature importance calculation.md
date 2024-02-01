

1. Global Configuration and Workspace Setup for Model Interpretation. 

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
open_path = r"<The data storage path for SHAP or IG execution results>"
save_path = r"<your save path>"
type_name = ['swiss']
gene_ID = "Entry"
depleted_ID_len = 2  ## Discard columns only in the last few columns

## Workspace
if os.path.isdir(save_path + "/" + type_name[0]):
    pass
else:
    os.makedirs(save_path + "/" + type_name[0])
names = os.listdir(open_path)
pattern = re.compile('.csv')
names = list(map(lambda x: pattern.sub('', x), names))
```

2. Calculating and Saving feature Importance Summaries. and below we use `SHAP_importance_sum` API for calculate feature importance based on SHAP value or IG value, the core process contain two calculation.  Firstly, take the absolute values of the original SHAP or IG values, where the original values are considered as the contribution of features, and the sign indicates their contribution direction. Taking the absolute value can then be viewed as the importance of the feature. For the underlying theory, please refer to SHAP-related literature. Secondly, average the importance values of each feature across all data instances to measure single feature's average contribution effect. Finally, these values can be further averaged across all features to assess the overall average contribution of the feature set (corresponding to the Mean in a swarm plot). 

```python
# Reading the data of ID or SHAP results.
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

data_shap_importance_T = SHAP_importance_sum_claulate_outcome['shap_feature_importance_T'].loc[all_data.str_contain(i_data_name,list(SHAP_importance_sum_claulate_outcome['shap_feature_importance_T'].index))]
data_shap_importance_T.to_csv(
    save_path + "/" + type_name[0] + "/shap_feature_" + i_data_name + "_importance_T.csv", index_label='cluster_name')
```

Note we not discuss detail in above outcome, but with example in next analysis. besides `FeaturePlot`are not example in this file, please refer to the source code of the `protloc_mex1.SHAP_plus` directly, or you can based on original SHAP or IG value using SHAP package directly to draw `summary plot` or `feature dependence plot`.
