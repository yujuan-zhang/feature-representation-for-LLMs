# Feature Representation for LLMs

## Introduction

### **Author Contact Information:**

- Author 1: Zeyu Luo
     Email: 1024226968@qq.com
- Author 2: Rui Wang
     Email: 2219312248@qq.com

This repository presents the implementation of "Feature Representation for Latent Language Models (LLMs)" and includes two Python libraries, namely protloc-mex1 ([https://pypi.org/project/protloc-mex1/](https://pypi.org/project/protloc-mex1/)) and protloc-mex-x ([https://pypi.org/project/protloc-mex-x/](https://pypi.org/project/protloc-mex-x/)).

For detailed usage instructions regarding these two Python libraries, please refer to the documentation available on PyPI.

Please note that the article is currently under review, and as a result, some code examples have not been publicly disclosed yet. Once the review process is complete, we will make the relevant code examples available for public access.

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the authors via the provided email addresses. Thank you for your interest in our work!

## Datasets

The raw data regarding train, test, and independent sets have been placed in the "source_data" folder. To transform these raw sequences into corresponding feature representations, and to perform the train and test split, we will refer to the instructions provided in the mentioned Python toolkits (protloc-mex-x). 

For final processed data and feature representations generated during the process, you can contact the "Author 1: Zeyu Luo Email: [1024226968@qq.com]" for access.

## Model

### Feature representation model

The feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t6_8M_UR50D. Besides, we develop  protloc-mex-x which containing detail for 'cls','mean', 'eos','segment 0-9','pho' feature representation from ESM2.



### VAE dimensional reduction model 

VAE model includes the model weights file (model_parameters.pt), the architecture parameters file (model_optimization_results.xlsx), the model architecture file (VAE_original_architecture). 

For using VAE model you can follow this instructions,

1. First, we need to import the required Python libraries, and define the architecture for the VAE (Variational Autoencoder) model.

```

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import os
# Set the device to GPU (if available) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VAE model
class ContinuousResidualVAE(torch.nn.Module):
# Model architecture is defined in the model architecture file, please copy it to replace this.

```

2. Loading a pre-trained VAE (Variational Autoencoder) model, and then use the model to reduce the ESM21280-dimensional features to 18 dimensions.

```
# Randomly set parameters
input_dim = 1280  # the dimensionality of ESM2 feature representation is 1280
hidden_dim = 859  
z_dim = 18  
## hidden_dim and z_dim are sourced from the architecture parameters file.
save_dir = './Model/VAE model'  # Directory to save the model parameters

# # Randomly generate a dataset

X_train = pd.DataFrame(np.random.randn(100, input_dim))
X_val = pd.DataFrame(np.random.randn(100, input_dim))


# Convert to PyTorch datasets
train_dataset = TensorDataset(torch.Tensor(X_train.values))
val_dataset = TensorDataset(torch.Tensor(X_val.values))

# Define data loaders
batch_size = 32  # This is arbitrary; adjust as needed



# Loading the model
# Load parameters
load_model = ContinuousResidualVAE(input_dim, hidden_dim, z_dim, loss_type='MSE',reduction='sum').to(device)
load_model.load_state_dict(torch.load(os.path.join(save_dir, 'model_parameters.pt')))



```

3. Execute inference using the model, where each protein variable is individually processed through the model, extracting the reduced 18-dimensional features.

```
# perform inference directly on the dataset
latent_vectors = []
load_model.eval()  # switch to evaluation mode
with torch.no_grad():  # disable gradient computation
    for data in train_dataset:
        data = data[0].unsqueeze(0).to(device)  # Unsqueeze to add the batch dimension
        z = load_model.get_model_inference_z(data)
        latent_vectors.append(z.cpu().detach().numpy())

    for data in val_dataset:
        data = data[0].unsqueeze(0).to(device)  # Unsqueeze to add the batch dimension
        z = load_model.get_model_inference_z(data)
        latent_vectors.append(z.cpu().detach().numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)

# Convert the latent vectors to DataFrame and reorder it according to the original index
latent_vectors_df = pd.DataFrame(latent_vectors, index=np.concatenate([X_train.index, X_val.index]), columns=[f"latent_{i}" for i in range(latent_vectors.shape[1])])
```



### DNN/RF classification model

For using downstream prediction model based on feature representation, we develop several DNN and RF model for different feature representation construction and demonstrate how to use DNN model based on combined feature to inference and evaluate outcome .

1. Define DNN model. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from protloc_mex1.classifier_evalute import ClassifierEvaluator
import numpy as np

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

2. Load model parameter and fit model architecture.

```python
# Configuration
input_dim = 3152 ##the dim of combined feature 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes=10

save_dir = "./Model/ESM2_feature_all/DNN_model_param"

save_path = "<YOUR_PATH_HERE>"
# Load the model
# Load parameters
load_model=ClassificationDNN(input_dim=input_dim, hidden_dim=802,num_classes=num_classes).to(device)

load_model.load_state_dict(torch.load(os.path.join(save_dir,'model_parameters.pt')))
load_model.eval()  # 设置模型为评估模式
```

3. 读入模型的训练集以确保后面的推断数据的特征与模型训练用到的数据特征一样, conduct inference and evaluate. Caution, this dataset is only a small subset of the original data. To access the complete dataset, please either follow the previous steps for generation or contact the author.

```python
train_data = pd.read_excel(os.path.join(save_dir, 'train_ESM2_feature_all_DNN.xlsx'))

inference_data=pd.read_excel(os.path.join(save_dir,'ESM2_combined_feature_inference_test.xlsx'))

inference_data.set_index('ID',inplace=True)

X_inference_data= inference_data.drop('label',axis=1)
y_inference_data= inference_data.loc[:,'label']

#检查 merged_df 是否包含所有 train_scale_data 的列
if set(train_data.columns) == set(X_inference_data.columns):
    print("All columns from X_train are in X_test.")
else:
    raise ValueError("The columns of 'X_train' do not match the columns of 'X_test'.")

#调整列的顺序，与训练集相同
X_inference_data = X_inference_data.reindex(columns=train_data.columns)
```

4. 模型推断以得到分类结果，最后还执行模型的分类效果评价（混淆矩阵、精准率、MCC）

```python
label_mapping=pd.read_excel(os.path.join(save_dir,'label2number.xlsx'))

# Convert DataFrame to Dictionary
label_dict = dict(zip(label_mapping['EncodedLabel'], label_mapping['OriginalLabel']))

X_inference_data_hat,X_inference_data_probabilities=load_model.model_infer(X_inference_data,device=device)

X_inference_data_hat = [label_dict[i] for i in X_inference_data_hat]


# Build classifier and perform evaluation

# Convert the prediction results to DataFrame

X_inference_data_hat = pd.DataFrame(X_inference_data_hat, columns=["predict"],index=X_inference_data.index)

classes=label_mapping.loc[:,'OriginalLabel'].values

# Create ClassifierEvaluator object

test_classification = ClassifierEvaluator(X_inference_data_probabilities, y_inference_data, X_inference_data_hat, classes)

# Save evaluation results

test_classification.classification_report_conduct(save_path,'/file_name')

# Plot evaluation charts
test_classification.classification_evaluate_plot(save_path,'/file_name',(10,10))
```

## model interpretation 

```
## we are under review, and this example code and file regard this model will realse soon.
```



## Citation

For those interested in citing our work related to this project, please note that our article is currently under review. We appreciate your interest and will provide citation details once the review process is complete.

If you are using the ESM-2 model in your project or research, please cite the original work by the authors:
Lin, Z., et al., Evolutionary-scale prediction of atomic level protein structure with a language model. bioRxiv, 2022: p. 2022.07.20.500902.



## Acknowledgments

we are acknowledge the contributions of the open-source community and the

developers of the Python libraries used in this study.
