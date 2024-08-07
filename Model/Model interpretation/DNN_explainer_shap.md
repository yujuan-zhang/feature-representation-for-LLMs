Note, the below demo is conduct in Swiss train set. other datasets like the Swiss test and independent datasets are treated in the same manner, and it's important to note that the background datasets in this trial are the respective datasets that need to be explained. For example, when explaining the Swiss train, the background dataset is the Swiss train set; when explaining the Swiss test, the background dataset is the Swiss test.

1. Define the DNN Model , note the model is the classification DNN model (MLP+RF_filter).

```python
## Define the DNN model
##DNN_MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc5 = nn.Linear(hidden_dim // 8, num_classes)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8)
        # Loss function
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        
        return F.log_softmax(x, dim=1)  # Apply Log Softmax for multi-class classification

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)
    
    def model_infer(self, X_data, device):
        self.eval()

        input_data = torch.Tensor(X_data.values).to(device) # or your test data

        with torch.no_grad():
            predictions = self(input_data)
            
        predictions = predictions.exp()
        _, predicted_labels = torch.max(predictions, 1)

        predicted_labels = predicted_labels.cpu().numpy()
        probabilities = predictions.cpu().numpy()
        return predicted_labels, probabilities
```

2. Loading package and preparing for loading model parameter (which is the same as the DNN trained model and the load path is `'./Model/ESM2_feature_all/DNN_model_param'`)  .

```python
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import shap
from protloc_mex1.SHAP_conduct import DeepSHAPValueConduct

# Define global parameters and file paths
open_path=r"./Model/ESM2_feature_all/DNN_model_param"
save_path=r"<your save path>"
up_down_name="label"
type_name=['human']
gene_ID="Entry"
shap_plot_figure_size=(15,10)

# Define file names
names=os.listdir(open_path)
model_parameters_name = 'model_parameters.pt'
model_optimization_results_name = 'model_optimization_results.xlsx'
label2number_name = 'label2number.xlsx'

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

3. Loading Model Parameters and Data for SHAP Analysis. specifically, the `swiss_feature_all.xlsx` is placed in figshare，other TrEMBL data can also processing and be inferred  in the same way.

```python
# Load PyTorch model parameters
model_parameters = torch.load(os.path.join(open_path, model_parameters_name))

# Load model optimization results
model_optimization_results = pd.read_excel(os.path.join(open_path, model_optimization_results_name))

# Load training data
swiss_data = pd.read_excel(os.path.join(open_path, 'swiss_feature_all.xlsx'),index_col=gene_ID)

# Load label mapping dictionary
label2number = pd.read_excel(os.path.join(open_path, label2number_name))
```

4. Preprocessing  Datasets and Model Setup for SHAP Analysis.

```python
# Preprocess training data
'''
caution feature in your inference data must equal to the model train data, 
detail in model training section. 
'''
X_train = swiss_data.drop(columns=up_down_name)
y_train = swiss_data[up_down_name]


# Create a PyTorch model
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model=ClassificationDNN(input_dim=input_dim, hidden_dim=model_optimization_results.loc[0,'Best hidden_dim'],
                             num_classes=len(label2number.index)).to(device)
load_model.load_state_dict(torch.load(os.path.join(open_path,model_parameters_name)))
load_model.eval()  

# Predict using the model
X_train_scale_hat, X_train_scale_probabilities = load_model.model_infer(X_train, device)

# Convert predicted labels
label_dict = dict(zip(label2number['EncodedLabel'], label2number['OriginalLabel']))
X_train_scale_hat = [label_dict[i] for i in X_train_scale_hat]
X_train_scale_hat_df = pd.DataFrame(X_train_scale_hat, columns=["predict"],index=X_train.index)
```

5. Perform Deep SHAP Analysis on datasets and Save SHAP Values. Note, in this way `DeepSHAPValueConduct` API are designed for easily save shap value in excel form. if you want to save as npy form or other, please using instantiation of `DeepSHAPValueConduct` function `.shap_value_conduct()` directly, which will return shap value in a array form.

```python
# Initialize SHAP DeepExplainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor=torch.Tensor(X_train.values).to(device)

# Set a random seed
np.random.seed(0)

background_index = np.random.choice(X_train.shape[0], size=200, replace=False)

# Reset the random seed
np.random.seed(None)

background_data = X_train.iloc[background_index].values  # Convert DataFrame to numpy array
background_data_tensor=torch.Tensor(background_data).to(device)

explainer = shap.DeepExplainer(load_model, background_data_tensor)

# Analyze the training dataset using Deep SHAP
train = DeepSHAPValueConduct(explainer,X_train, X_train_tensor,y_train)
shap_train_values= train.shap_value_conduct()

X_shap_train_save=train.Shapley_value_save(X_train_scale_hat_df,type_class=list(label2number.loc[:,'OriginalLabel']),
                                    save_path=save_path+"/"+type_name[0]+"/"+"shap_global/", file_name="swiss",gene_ID=gene_ID)
```

6. Calculate Baseline Prediction Probability and Save Results as an Excel File.

```python
# Calculate base prediction probabilities
base_value = explainer.expected_value
result_df = pd.DataFrame({'base_value': base_value,
                          'classes': list(label2number.loc[:,'OriginalLabel'])})

# Save the base probabilities to an Excel file
result_df.to_excel(save_path + "/" + type_name[0] + "/" + "shap_global/SHAP_base_probablity.xlsx", index=False)
```

At last we thanks for developer for `SHAP` package, we are only integrate different interpretation technique include `SHAP`  in our pipeline, and for promote `SHAP` use in bioinformatics.  Our DNN model based SHAP analysis can directly come true based on `shap.DeepExplainer`, So if our package install or process is not satisfy your system and experiments, you can choose `shap.DeepExplainer` from `SHAP`  directly and reproduce the experiment.
