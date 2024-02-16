Global Interpretation of DNN Results Using SHAP for Multi-class Classification with the IntegratedGradients Class

1. Define the DNN Model (The model is the same as the previous DNN model). 

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

2.  Setting up Global Configuration and Workspace Directories for Integrated Gradients analysis.

```python
import os
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients
from protloc_mex1.IG_calculator import IntegratedGradientsCalculator

# Define global parameters and file paths
open_path=r"<the path>"
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

# Create a directory for the specified cancer type
if os.path.isdir(save_path+"/"+type_name[0]):
    pass
else:
    os.makedirs(save_path+"/"+type_name[0])

if os.path.isdir(save_path+"/"+type_name[0]+"/IntegratedGradients"):
    pass
else:
    os.makedirs(save_path+"/"+type_name[0]+"/IntegratedGradients")
```

3. Loading Model Parameters and Data for Integrated Gradients analysis. specifically, the `swiss_feature_all.xlsx` is placed in figshare，other TrEMBL data can also processing and be inferred  in the same way.

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

4. Preprocessing Datasets and Model Setup for Integrated Gradients analysis.

```python

# Preprocess the training and testing datasets
X_train = swiss_data.drop(columns=up_down_name)
y_train = swiss_data[up_down_name]

# Model
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
load_model = ClassificationDNN(input_dim=input_dim, hidden_dim=model_optimization_results.loc[0,'Best hidden_dim'],
                             num_classes=len(label2number.index)).to(device)

load_model.load_state_dict(torch.load(os.path.join(open_path,model_parameters_name)))

                           
load_model.eval()  # Set the model to evaluation mode

# Get model predictions (predicted labels)
X_train_scale_hat, X_train_scale_probabilities = load_model.model_infer(X_train, device)

# Convert predicted labels using a label mapping dictionary
label_dict = dict(zip(label2number['EncodedLabel'], label2number['OriginalLabel']))
X_train_scale_hat = [label_dict[i] for i in X_train_scale_hat]

# Convert the predicted results to DataFrames
X_train_scale_hat_df = pd.DataFrame(X_train_scale_hat, columns=["predict"], index=X_train.index)

```

5. Perform Integrated Gradients analysis on datasets and Save result.

```python

# Perform local analysis of the PyTorch-based DNN model on the training set using Deep SHAP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.Tensor(X_train.values).to(device)

train =IntegratedGradientsCalculator(load_model,X_train, X_train_tensor,y_train,batch_size=100, n_steps=50)
IG_train_values= train.compute_integrated_gradients(list(label2number.loc[:,'OriginalLabel']))
IG_train_values_save=train.integrated_gradients_save(X_train_scale_hat_df,save_path=save_path+"/"+type_name[0]+"/IntegratedGradients/swiss_")
```
