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

### UMAP Model

```
## we are under review, and this example code and file regard this model will realse soon.
```

### Feature representation model

The feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t6_8M_UR50D. Besides, we develop  protloc-mex-x which containing detail for 'cls','mean', 'eos','segment 0-9','pho' feature representation from ESM2.



### VAE dimensional reduction model 

VAE model includes the model weights file (model_parameters.pt), the architecture parameters file (model_optimization_results.xlsx), the model architecture file (VAE_original_architecture). 

For using VAE model you can follow this instructions,

1. First, we need to import the required Python libraries, and define the architecture for the VAE (Variational Autoencoder) model.

```python
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

2. Loading a pre-trained VAE (Variational Autoencoder) model, and then use the model to reduce the ESM2 'cls' 1280-dimensional features to 18 dimensions. For inference on additional features (eos, pho), refer to the preceding method and independently train the VAE model.

```python
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

```python
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

<<<<<<< HEAD
### Stand Scaler Model

由于数据增强阶段需要用到基于距离的算法来生成数据，因此我们对训练集进行z-score标准化，再进行数据增强，这样会导致亚细胞定位模型（DNN，RF）需要对标准化后的数据进行拟合。因此我们保留了训练集的标准化特征即平均值和方差，并将其用于测试集和推断数据的对应特征进行标准化。值得注意的是使用训练集的标准差和平均值来对测试集或推断数据进行标准化是合理且标准的做法。这种做法不会导致数据泄露。理由如下，

1. 在机器学习中，重要的一点是防止从测试集中泄露任何信息到训练过程中。如果你使用测试集的统计数据（例如平均值和标准差）来标准化训练集，那么就会产生数据泄露。但是，如果你使用训练集的统计数据来标准化测试集和推断数据，那么不会发生数据泄露。
2. 由于模型是在训练集的分布下训练的，因此使用相同的参数（即训练集的平均值和标准差）来标准化测试集和推断数据是合理的。这确保了模型在处理类似分布的数据时能够表现良好。
3. 在实际情况中，当新数据到来时，你通常不会有机会重新计算标准化参数。因此，使用训练集的参数来标准化新数据是模拟实际操作的一种方式。

基于此，我们对不同特征划分的训练集（与后面的DNN/RF模型对应）都进行了自己的标准化，并将进行标准化的模型放入了`./Model/Stand_scaler_model`中，下面我们将以cls_scaler举例说明如何使用我们训练好的标准化模型对cls特征的测试集或推断数据集进行拟合操作（注：标准化模型训练就是对训练集进行z-score并保留特征的平均值和标准差，故不再介绍训练过程）。需要注意的是所有数据包括新的拟合数据在将其放入训练好的DNN或RF进行推断时都需要根据对应的标准化模型进行拟合。

1. 读入定义的SimpleScaler标准化模型框架和对应的训练集数据，注意这里的训练集只是feature all的特征训练集的一部分蛋白，主要是作为记录数据的特征列顺序的信息。

   ```python
   import os
   import pandas as pd
   import numpy as np
   from <> import SimpleScaler
   save_dir = "./Model/Stand_scaler_model"
   
   save_path = "<YOUR_PATH_HERE>"
   train_data = pd.read_csv(os.path.join(save_dir, 'feature_all_train.csv'))‘
   
   scaler_filepath =save_dir+ '/feature_all_scaler.joblib'  
   feature_all_scaler = SimpleScaler.load(scaler_filepath)
   ```

2. 读入待标准化的新的推理数据集，需要确保其特征和训练集特征一模一样。

   ```python
   inference_data=pd.read_excel(os.path.join(save_dir,'your_data.xlsx'))
   
   inference_data.set_index('ID',inplace=True)##if your data have ID ,you need do this step.
   
   X_inference_data= inference_data.drop('label',axis=1)##if you are using our test data need this step.
   y_inference_data= inference_data.loc[:,'label'] ##if you are using our test data need this step.
   
   #Verify if merged_df encompasses all columns present in train_scale_data
   if set(train_data.columns) == set(X_inference_data.columns):
       print("All columns from X_train are in X_test.")
   else:
       raise ValueError("The columns of 'X_train' do not match the columns of 'X_test'.")
   
   #Adjust the order of columns to match that of the training set
   X_inference_data = X_inference_data.reindex(columns=train_data.columns)
   ```

3. 对`X_inference_data`按照训练好的标准化模型进行标准化操作,在进行前可设置numpy的输出精度为15位小数

   ```python
   # 设置numpy的输出精度为15位小数
   np.set_printoptions(precision=15)
   normalized_test_data = feature_all_scaler.transform(X_inference_data)
   ```

注意这里只演示了feature all特征数据集的标准化，如果数据集是使用ESM2提取的其它特征如'cls','eos','pho'等需要在`./Model/Stand_scaler_model`中使用对应的标准化模型和特征训练集（用以确保数据集的特征与训练模型的特征一致。）后面的DNN和RF也是类似的过程即需要预训练的模型和对应的特征输入。当然我们并不想加大使用我们模型的难度，这样处理主要是为了更系统化的对ESM2的不同特征提取进行详细分析，我们的项目致力于提供底层的分段式模块，用于大家自由组合搭建需要的任务流程。

=======
>>>>>>> fef861451b3241180015b0c41981e05b0bfe51cd
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
load_model.eval()  # Set the model to evaluation mode
```

3. Load the training set of the model to ensure that the features of the inference data match the features of the data used for model training, conduct inference and evaluate. Caution, this dataset is only a small subset of the original data. To access the complete dataset, please either follow the previous steps for generation or contact the author.

```python
train_data = pd.read_excel(os.path.join(save_dir, 'train_ESM2_feature_all_DNN.xlsx'))

inference_data=pd.read_excel(os.path.join(save_dir,'ESM2_combined_feature_inference_test.xlsx'))

inference_data.set_index('ID',inplace=True)

X_inference_data= inference_data.drop('label',axis=1)
y_inference_data= inference_data.loc[:,'label']

#Verify if merged_df encompasses all columns present in train_scale_data
if set(train_data.columns) == set(X_inference_data.columns):
    print("All columns from X_train are in X_test.")
else:
    raise ValueError("The columns of 'X_train' do not match the columns of 'X_test'.")

#Adjust the order of columns to match that of the training set
X_inference_data = X_inference_data.reindex(columns=train_data.columns)
```

4. Perform model inference to obtain classification results, and finally execute the evaluation of the model's classification performance (confusion matrix, precision, MCC).

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
