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

the final processed data and feature representations generated during the process are already placed on figshare(DOI:10.6084/m9.figshare.24312292), If you have any questions, please contact the "Author 1: Zeyu Luo Email: [1024226968@qq.com]" for access.

## Model

### UMAP Model

```
## we are under review, and this example code and file regard this model will realse soon.
```

### Feature representation model

The feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t6_8M_UR50D. Besides, we develop  protloc-mex-x which containing detail for 'cls','mean', 'eos','segment 0-9','pho' feature representation from ESM2.

### Res-VAE dimensional reduction model 

For the detail in training the Res-VAE model refer to <VAE训练说明>

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
# Model architecture is defined in the <VAE训练说明> step3, please copy it to replace this.
```

2. Loading a pre-trained VAE (Variational Autoencoder) model, and then use the model to reduce the ESM2 'cls' 1280-dimensional features to 18 dimensions. For inference on additional features (eos, pho), refer to the preceding method and independently train the VAE model.

```python
# Randomly set parameters
input_dim = 1280  # the dimensionality of ESM2 'cls' feature representation is 1280
hidden_dim = 859  
z_dim = 18  
## hidden_dim and z_dim are sourced from the architecture parameters file.
save_dir = './Model/VAE model'  # Directory to save the model parameters

# # Randomly generate a dataset for inference

X_inference = pd.DataFrame(np.random.randn(100, input_dim))

# Convert to PyTorch datasets
inference_dataset = TensorDataset(torch.Tensor(X_inference.values))

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
    for data in inference_dataset:
        data = data[0].unsqueeze(0).to(device)  # Unsqueeze to add the batch dimension
        z = load_model.get_model_inference_z(data)
        latent_vectors.append(z.cpu().detach().numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)

# Convert the latent vectors to DataFrame and reorder it according to the original index
latent_vectors_df = pd.DataFrame(latent_vectors, index=X_inference.index, columns=[f"latent_{i}" for i in range(latent_vectors.shape[1])])
```

### Stand Scaler Model

During the data augmentation phase, distance-based algorithms are employed to generate data, which necessitates the normalization of the training set using the z-score method before augmentation. As a result, subcellular localization models such as Deep Neural Networks (DNN) and Random Forests (RF) are required to fit the normalized data. Consequently, we retain the normalization features of the training set, namely the mean and standard deviation, and apply them for normalizing the corresponding features of the test set and inference data. It's worth noting that using the training set's standard deviation and mean to normalize the test or inference data is a standard and reasonable practice, which does not lead to data leakage, for the following reasons:

1. One key principle in machine learning is to prevent any leakage of information from the test set into the training process. If the statistical metrics (such as mean and standard deviation) of the test set are used to normalize the training set, data leakage occurs. However, utilizing the training set's statistics to normalize the test set and inference data prevents such leakage.

2. Given that the model is trained under the distribution of the training set, it is logical to use the same parameters (i.e., the mean and standard deviation of the training set) to normalize the test set and inference data. This ensures that the model performs well when handling data with similar distributions.

3. In practical scenarios, when new data arrives, there is usually no opportunity to recalculate normalization parameters. Hence, employing the training set's parameters to normalize new data simulates real-world operations effectively.

Based on this, we have performed individual normalizations for different feature divisions of the training set (corresponding to the subsequent DNN/RF models) and stored the normalization models in `./Model/Stand_scaler_model`. Below, we exemplify how to use our trained normalization model to fit the 'feature all' features of the test set or inference dataset, using `feature_all_scaler` as an example (Note: the training of the normalization model entails performing z-score on the training set while retaining the features' mean and standard deviation, thus the training process will not be elaborated further). It is important to note that all data, including new fitting data, need to be normalized according to the corresponding normalization model before being fed into the trained DNN or RF for inference.

1. Load the defined `SimpleScaler` normalization model  along with the corresponding training set data. Note that the training set here comprises only a portion of the protein in the 'feature all' feature training set, primarily serving as a record of the order of feature columns.

   ```python
   import os
   import pandas as pd
   import numpy as np
   import joblib
   ##define SimpleScaler model
   class SimpleScaler:
       def __init__(self):
           self.mean = None
           self.std = None
       
       def fit(self, data):
           self.mean = np.mean(data, axis=0)
           self.std = np.std(data, axis=0)
       
       def transform(self, data):
           return (data - self.mean) / self.std
       
       def fit_transform(self, data):
           self.fit(data)
           return self.transform(data)
           
       def save(self, filepath):
           joblib.dump((self.mean, self.std), filepath)
       
       @classmethod
       def load(cls, filepath):
           scaler = cls()
           scaler.mean, scaler.std = joblib.load(filepath)
           return scaler
   
   save_dir = "./Model/Stand_scaler_model"
   
   save_path = "<YOUR_PATH_HERE>"
   train_data = pd.read_csv(os.path.join(save_dir, 'feature_all_train.csv'))‘
   
   scaler_filepath =save_dir+ '/feature_all_scaler.joblib'
   ##load model pt (mean and std)
   feature_all_scaler = SimpleScaler.load(scaler_filepath)
   ```

2. Load the new inference dataset to be normalized, ensuring that its features exactly match those of the training set.

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

3. Normalize `X_inference_data` according to the trained normalization model. Before proceeding, you may set the output precision of `numpy` to 15 decimal places.

   ```python
   # Set the output precision to 15 decimal places
   np.set_printoptions(precision=15)
   normalized_test_data = feature_all_scaler.transform(X_inference_data)
   ```

Please note that here only the normalization of the 'feature all' feature dataset is demonstrated. If the dataset employs other features extracted using ESM2 such as 'cls', 'eos', 'pho', etc., it is necessary to use the corresponding normalization models and feature training sets in `./Model/Stand_scaler_model` (to ensure consistency of features between the dataset and training model). The subsequent processes with DNN and RF are similar, requiring pre-trained models and corresponding feature inputs. We do not aim to increase the complexity of utilizing our models; this approach is primarily adopted for a more systematic analysis of different feature extractions from ESM2. Our project is committed to providing foundational modular segments, allowing for flexible assembly and configuration of required task workflows by users.

### DNN/RF classification model

For using downstream prediction model based on feature representation, we develop several DNN and RF model for different feature representation construction and demonstrate how to use DNN model based on combined feature to inference and evaluate outcome.

For the detail in training the DNN model and RF model refer to <DNN训练说明> and <RF训练说明>.

For inference using the trained DNN and RF models, please refer to the following guidelines.

**RF model for inference**

1. load RF model,`train_data` (also used for check if the inference data feature match corresponding model) and `inference_data`, we also choose 'feature all' model and corresponding train_data for demonstrate and place in `./Model/ESM2_feature_all/RF_model_param`, other type of feature train data with their model can communicate to author for acquired . Please note that the RF model is a complete `scikit-learn` model. It is crucial to ensure that your `scikit-learn` version is compatible with ours. The version of the `scikit-learn` package is indicated in the model filename as `1.2.2`.

   ```python
   import os
   import pandas as pd
   import numpy as np
   from joblib import load
   
   save_dir = './Model/ESM2_feature_all/RF_model_param'
   save_path = "<YOUR_PATH_HERE>"
   
   model=load(os.path.join(save_dir,'ESM2_feature_allhuman1.2.2.pkl'))
   
   train_data = pd.read_csv(os.path.join(save_dir, 'train_ESM2_feature_all_RF.csv'))
   
   inference_data=pd.read_excel(os.path.join("./Model/ESM2_feature_all/DNN_model_param",'ESM2_combined_feature_inference_test.xlsx'))
   
   inference_data.set_index('ID',inplace=True)
   
   X_inference_data= inference_data.drop('label',axis=1)
   y_inference_data= inference_data.loc[:,'label'] #if using test data you need this step or not when in inference.
   
   #Verify if merged_df encompasses all columns present in train_scale_data
   if set(train_data.columns) == set(X_inference_data.columns):
       print("All columns from X_train are in X_test.")
   else:
       raise ValueError("The columns of 'X_train' do not match the columns of 'X_test'.")
   
   #Adjust the order of columns to match that of the training set
   X_inference_data = X_inference_data.reindex(columns=train_data.columns)
   
   ```

2. Model Prediction and Classification Result Evaluation

   ```python
   # model predict
   X_inference_data_hat = pd.DataFrame(model.predict(np.array(X_inference_data)), columns=["predict"], index=X_inference_data.index)
   
   # Create ClassifierEvaluator object
   
   test_classification = ClassifierEvaluator(model.predict_proba(X_inference_data), y_inference_data, X_inference_data_hat, model.classes_)
   # Save evaluation results
   
   test_classification.classification_report_conduct(save_path,'/your_file_name')
   
   # Plot evaluation charts
   test_classification.classification_evaluate_plot(save_path,'/your_file_name',(10,10))
   ```

   

**DNN model for inference**

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
input_dim = 3152 ##the dim of 'feature all' 

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
y_inference_data= inference_data.loc[:,'label'] #if using test data you need this step or not when in inference.

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

test_classification.classification_report_conduct(save_path,'/your_file_name')

# Plot evaluation charts
test_classification.classification_evaluate_plot(save_path,'/your_file_name',(10,10))

```



## model interpretation 

```
## we are under review, and this example code and file regard this model will realse soon.
```

## Dataset Available



## Citation

For those interested in citing our work related to this project, please note that our article is currently under review. We appreciate your interest and will provide citation details once the review process is complete.

If you are using the ESM-2 model in your project or research, please cite the original work by the authors:
Lin, Z., et al., Evolutionary-scale prediction of atomic level protein structure with a language model. bioRxiv, 2022: p. 2022.07.20.500902.



## Acknowledgments

we are acknowledge the contributions of the open-source community and the

developers of the Python libraries used in this study.
