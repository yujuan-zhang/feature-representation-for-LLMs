Note, the below demo is conduct in Swiss train set. other datasets like the Swiss test and independent datasets are treated in the same manner, and it's important to note that the background datasets in this trial are the respective datasets that need to be explained. For example, when explaining the Swiss train, the background dataset is the Swiss train set; when explaining the Swiss test, the background dataset is the Swiss test.

1. Loading package and preparing for loading model (load path is `'.Model/ESM2_feature_all/RF_model_param'`)  .

```python

import os
import pandas as pd
import numpy as np
from joblib import load
import shap
from protloc_mex1.SHAP_conduct import SHAPValueConduct
from protloc_mex1.SHAP_plus import FeaturePlot

# Define global parameters and file paths
open_path=r".Model/ESM2_feature_all/RF_model_param"
save_path=r"<your save path>"
up_down_name="label"
type_name=['human']
gene_ID="Entry"

# Define file names
names=os.listdir(open_path)

# Define RF model name
model_parameters_name = 'ESM2_feature_allhuman1.2.2.pkl'

# Set up working directories
if os.path.isdir(save_path+"/"+type_name[0]):
    pass
else:
    os.makedirs(save_path+"/"+type_name[0])

if os.path.isdir(save_path+"/"+type_name[0]+"/shap_global"):
    pass
else:
    os.makedirs(save_path+"/"+type_name[0]+"/shap_global")
    
```

2. Loading Model Parameters and Data for SHAP Analysis. specifically, the `swiss_feature_all.xlsx` is placed in figshare，other TrEMBL data can also processing and be inferred  in the same way.

```python
# Load RF model pkl
model = load(os.path.join(open_path, model_parameters_name))

# Load training data
swiss_data = pd.read_excel(os.path.join(open_path, 'swiss_feature_all.xlsx'),index_col=gene_ID)

```

3. Preprocessing  Datasets and Model Setup for SHAP Analysis.

```python

# Preprocess training data 
'''
caution feature in your inference data must equal to the RF_train data, 
detail in model training section. 
'''
X_train = swiss_data.drop(columns=up_down_name)
y_train = swiss_data[up_down_name]

# Predict using the model
X_train_predict=pd.DataFrame(model.predict(np.array(X_train)),columns=["predict"],
                   index=X_train.index)
```

4. Perform Tree SHAP Analysis on datasets and Save SHAP Values. Note, in this way `SHAPValueConduct` API are designed for easily save shap value in excel form. if you want to save as npy form or other, please using instantiation of `SHAPValueConduct` function `.shap_value_conduct()` directly, which will return shap value in a array form.

```python
# Initialize RF TreeExplainer

explainer = shap.TreeExplainer(model)

# Analyze the training dataset using Deep SHAP
train = SHAPValueConduct(explainer,X_train,y_train)
shap_train_values= train.shap_value_conduct()

X_shap_train_save=train.Shapley_value_save(X_train_predict,type_class=model.classes_,
                                    save_path=save_path+"/"+type_name[0]+"/"+"shap_global/", file_name="swiss",gene_ID=gene_ID)

```

5. Calculate Baseline Prediction Probability and Save Results as an Excel File.

```python
##Calculate base prediction probabilities
base_value = explainer.expected_value

result_df = pd.DataFrame({'base_value': base_value,
                          'classes': model.classes_.tolist()})

# Save the base probabilities to an Excel file
result_df.to_excel(save_path + "/" + type_name[0] + "/" + "shap_global/SHAP_base_probablity.xlsx", index=False)

```

At last we thanks for developer for `SHAP` package, we are only integrate different interpretation technique include `SHAP`  in our pipeline, and for promote `SHAP` use in bioinformatics.  Our RF model based SHAP analysis can directly come true based on `shap.TreeExplainer`, So if our package install or process is not satisfy your system and experiments, you can choose `shap.TreeExplainer` from `SHAP`  directly and reproduce the experiment.
