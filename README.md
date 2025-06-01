# Feature Representation for LLMs

## Introduction

### **Author Contact Information:**

- Author 1: Zeyu Luo, Email: 1024226968@qq.com, ORCID: 0000-0001-6650-9975
  
- Author 2: Rui Wang, Email: 2219312248@qq.com
  
- Author 3: Yawen Sun, Email: 2108437154@qq.com


This repository presents the implementation of "Feature Representation for Latent Language Models (LLMs)" and includes two Python libraries, namely protloc-mex1 ([https://pypi.org/project/protloc-mex1/](https://pypi.org/project/protloc-mex1/)) and protloc-mex-x ([https://pypi.org/project/protloc-mex-x/](https://pypi.org/project/protloc-mex-x/)).

For detailed usage instructions regarding these two Python libraries, please refer to the documentation available on PyPI. 

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the authors via the provided email addresses. Thank you for your interest in our work!

### Update new 
For new release and update please refer to this [document](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/update_new.md)

## Dataset Available

The raw data regarding train, test and independent sets have been placed in the "source_data" folder (**Note**: all protein in this experiment are belong to human protein). To transform these raw sequences into corresponding feature representations, we will refer to the instructions provided in the mentioned Python toolkits (protloc-mex-x). 

the final processed data and feature representations generated during the process are already placed on [figshare](https://figshare.com/articles/dataset/feature-representation-for-LLMs/24312292) (DOI:10.6084/m9.figshare.24312292), If you have any questions, please contact the "Author 1: Zeyu Luo Email: [1024226968@qq.com]" for access.

Additional, the average importance feature data (Table S10) related to this project are available on figshare(DOI:10.6084/m9.figshare.24312292).

Figure S4 (eos) and Figure S5 (eos) are supplement for the Histogram plots and Scatter plots of feature eos in corresponding Figure S4 and Figure S5.

Figure S6-8 are also available on figshare(DOI:10.6084/m9.figshare.24312292).

## Work Environment Setup

To ensure that you can replicate our work from the paper accurately, we recommend using the following work environment:

- Python version: 3.9.7
- protloc-mex-x version: 0.0.17
- protloc-mex1 version: 0.0.21

Here are the [details](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Work%20Environment%20Setup/Setting%20Up%20the%20Work%20Environment.md).
### PyPI Open Source

We have made the source code for `protloc-mex-x` and `protloc-mex1` available [here](https://github.com/yujuan-zhang/feature-representation-for-LLMs/tree/main/package%20source%20code%20backup). This is especially useful for those who are not familiar with PyPI and prefer to reference the source code directly. However, please note that this serves as a backup, and the latest updates will continue to be released directly on PyPI (protloc-mex1 ([https://pypi.org/project/protloc-mex1/](https://pypi.org/project/protloc-mex1/)) and protloc-mex-x ([https://pypi.org/project/protloc-mex-x/](https://pypi.org/project/protloc-mex-x/))).

## Non-homologous division process

We performed a non-homologous operation, which lead to create non-homologous independent datasets, you can follow this [methods](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/Instructions%20for%20creating%20a%20dataset%20based%20on%20non-homologous%20division.md).

## Model

### Feature representation model

The feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t6_8M_UR50D. Besides, we develop  [protloc-mex-x](https://pypi.org/project/protloc_mex_X/) which containing detail for 'cls','mean', 'eos','segment 0-9','pho' feature representation from ESM2.

**Segmentation Formula Explanation**

The amino acid sequence is also divided into 10 equal-length segments, and the mean of the representations of the amino acid characters in each segment is calculated, yielding the ‘segment0–9’ mean features. The specific segmentation and calculation methods are as follows:

![](https://github.com/yujuan-zhang/libocell-file/blob/master/feature-representation-for-LLMs/Segmentation%20Formula.png)

where `L` represents the sequence length, `N` is the number of segments, set to 10 in this study, `S` represents the size of each segment, `R` is the remainder, `Ei`is the ending position, `H` symbolizes the hidden layer feature representations corresponding to the amino acid sequence, `Subi` is the result of the feature representation for residue in each segment, and by further averaging to get each segment mean features, `i` represents the specific segment within the given `N`.

Specifically, for `i = 0`, which is the first segment, `Sub0` represents the position from the start 0 to the end `E0`. 

For example, if the ordinal number `i` is less than the remainder `R`, then `E0 = S+1`, `E1 = 2E0`, `E2 = 3E0`. Additionally, if the end position `Ei` of a segment is greater than the previous end position `E(i-1)`, then `Sub0 = H[0:E0]`, `Sub0 = H[E0:E1]`. In summary, the purpose of the formula design is to ensure that when the sequence cannot be divided evenly, the remainder is distributed one by one to the segments at the front. If the number of amino acid residues in the sequence to be divided is less than the number of divisions `N`, then the subsequent segments will be zero vectors. Our design takes into account the rules of Python slicing. For more details, please refer to our source code on protloc-mex-x (https://pypi.org/project/protloc-mex-x/).

### Res-VAE dimensional reduction model 

For the detail in training the Res-VAE model refer to [VAE training detail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/VAE%20model/Res_VAE%20training%20detail.md)

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
# Model architecture is defined in the VAE training detail step3, please copy it to replace this.
```

2. Loading a pre-trained VAE (Variational Autoencoder) model, and then use the model to reduce the ESM2 'cls' 1280-dimensional features to 18 dimensions. For inference on additional features (eos, pho), refer to the preceding method and independently train the VAE model.

```python
# Randomly set parameters
input_dim = 1280  # the dimensionality of ESM2 'cls' feature representation is 1280
hidden_dim = 859  
z_dim = 18  
## hidden_dim and z_dim are sourced from the architecture parameters file.
save_dir = './Model/VAE model'  # Directory to the model parameters saved location.

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

During the data augmentation phase, distance-based algorithms are employed to generate data, which necessitates the normalization of the training set using the z-score method before augmentation. As a result, subcellular localization models such as Deep Neural Networks (DNN) and Random Forests (RF) are required to fit the normalized data (except DNN_Liner). Consequently, we retain the normalization features of the training set, namely the mean and standard deviation, and apply them for normalizing the corresponding features of the test set and inference data. It's worth noting that using the training set's standard deviation and mean to normalize the test or inference data is a standard and reasonable practice, which does not lead to data leakage, for the following reasons:

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

For the detail in training the RF_filter model，please refer to the fourth step in [RF training detrail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/ESM2_feature_all/RF_model_param/RF%20training%20detail.md).

For the detail in training the DNN model and RF model refer to [DNN training detrail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/ESM2_feature_all/DNN_model_param/DNN%20MLP.md) and [RF training detrail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/ESM2_feature_all/RF_model_param/RF%20training%20detail.md).

For inference using the trained DNN and RF models, please refer to the following guidelines.

**RF model for inference**

1. load RF model,`train_data` (also used for check if the inference data feature match corresponding model) and `inference_data`, we also choose 'feature all' model and corresponding train_data for demonstrate and place in `./Model/ESM2_feature_all/RF_model_param`, other type of feature train data with their model can be found in figshare(DOI: 10.6084/m9.figshare.24312292), if you have any problem, communicate to author for acquired . Please note that the RF model is a complete `scikit-learn` model. It is crucial to ensure that your `scikit-learn` version is compatible with ours. The version of the `scikit-learn` package is indicated in the model filename as `1.2.2`. Specifically, for RF model training and inference data (include Swiss_normalized and original_TrEMBL_normalized) are placed on figshare(DOI:10.6084/m9.figshare.24312292). Note that the non_homology version can divided from original_TrEMBL_normalized based on protein ID.

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

Specifically, for DNN (MLP+RF_filter) model training and inference data (include Swiss_normalized and original_TrEMBL_normalized) are placed on figshare(DOI:10.6084/m9.figshare.24312292). Note that the non_homology version can divided from original_TrEMBL_normalized based on protein ID.The DNN  (MLP+RF_filter) model using feature_all as input is saved in './Model/ESM2_feature_all/DNN_model_param'.

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

### DNN_Liner model

DNN_Liner model training detail in [methods](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/DNN%20Liner/DNN_Line%20model%20training.md) ,and completed trained model are placed in corresponding file. The DNN_Liner model does not conduct feature normalization and has not used data augmentation. This is a common practice when fine-tuning the output layer of large models. Specifically, for DNN_Liner model training and inference data (include Swiss, original_TrEMBL and non_homology_TrEMBL) are placed on figshare(DOI:10.6084/m9.figshare.24312292).

### MCC five-fold validation

In order to evaluate the performance of the independent test set more accurately and comprehensively, we employed `StratifiedKFold` for 5-fold stratified cross-validation, and calculated the average and sample standard deviation (unbiased estimate) of the MCC (Matthews Correlation Coefficient) scores from the cross-validation. Note each fold's training data is not utilized for model training but set aside, whereas the testing portion is employed to compute the MCC score. Hence, our approach more closely aligns with the external validation phase of Nested Cross-Validation. The steps are as follows:

1. Import the necessary packages, where cross-validation is performed using `StratifiedKFold` from the `sklearn` library.

```python
import pandas as pd
import os
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
```

2. Define a function to calculate the MCC score, and execute a 5-fold cross-validation. If a fold used for calculating the MCC score has fewer than two types of classification labels in the test set, skip this fold.

```python
def calculate_mcc_for_class(y_true, y_pred, protein_class):
    type_mapping = {value: 1 if value == protein_class else 0 for value in y_true.unique()}
    y_true_mapped = y_true.map(type_mapping)
    y_pred_mapped = y_pred.map(type_mapping)
    return matthews_corrcoef(y_true_mapped, y_pred_mapped)

def process_dict_for_mcc(data_dict, label_col, pred_col):
    rows_for_each_label = []

    for pattern, df in data_dict.items():
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        all_indices = list(skf.split(df, df[label_col]))  # Store all indices

        for label in df[label_col].unique():
            mcc_scores_label = []

            for train_index, test_index in all_indices:  # Use the stored indices
                _, test = df.iloc[train_index], df.iloc[test_index]

                # Inspect the number of categories
                if len(test[label_col].unique()) < 2 or len(test[pred_col].unique()) < 2:
                    continue  # skip this fold

                mcc = calculate_mcc_for_class(test[label_col], test[pred_col], label)
                mcc_scores_label.append(mcc)

            mean_mcc_label = np.mean(mcc_scores_label)
            std_mcc_label = np.std(mcc_scores_label, ddof=1)  # get sample standard deviation

            row_for_each_label = pd.DataFrame({
                'Pattern': [pattern],
                'Label': [label],
                'MCC': [mean_mcc_label],
                'MCC_Std': [std_mcc_label]  

            })
            rows_for_each_label.append(row_for_each_label)

    df_each_label = pd.concat(rows_for_each_label, ignore_index=True)

    return df_each_label  # Return each categorie label Mcc 5 fold outcome
```

3. Execute batch processing for 5-fold cross-validation to calculate the MCC scores, where `dfs_test` is a dictionary with different feature names as keys (e.g., 'cls', 'eos', 'pho', etc.), and values are datasets containing their respective features, true subcellular localization labels of proteins, and prediction labels from DNN or RF models, respectively. By running our algorithm, a comprehensive evaluation of different feature types and model predictions can be conducted in batch.

```python
dfs_test = {key: value for key, value in dfs_test.items()}  

df_each_label = process_dict_for_mcc(
        dfs_test,
        label_col="label",  # your data True label column
        pred_col="predict"  # model predict label column
    )

# save_path = '.'  # please define your save path
with pd.ExcelWriter(os.path.join(save_path, "mcc_results.xlsx")) as writer:
    if df_each_label is not None:
        df_each_label.to_excel(writer, sheet_name="each_label", i
                               ndex=False)
```

## Model usage demo (end-to-end prediction deployment)

This section provides a convenient example for efficiently accomplishing the protein sequence feature extraction task in this project. Specifically, it leverages the ESM2 model to perform deep analysis and feature extraction on protein sequences, followed by utilizing the extracted features for subcellular localization prediction of proteins. For detailed steps and methods, please refer to the method [details](https://github.com/yujuan-zhang/feature-representation-for-LLMs/tree/main/Model/mean_DNN_linear_inference).

## Model Interpretation 

![](https://github.com/yujuan-zhang/libocell-file/blob/master/feature-representation-for-LLMs/Work_procedure.png)

We employed three interpretability methods, DeepExplainer, Integrated Gradient, and Tree SHAP. Using these interpretability methods, we calculated feature importance. For details on the calculation methods and further feature importance visualization, please refer to the [methods](https://github.com/yujuan-zhang/feature-representation-for-LLMs/tree/main/Model/Model%20interpretation). The running steps for these methods are: `RF_Tree_shap` , `DNN_explainer_shap`or `IG` step -> `Average feature importance calculation` step -> `Swarm plot visualization` step.

Additionally, we presented Histogram plots and Scatter plots based on feature importance to measure the distribution of feature significance for overall prediction (summing across all subcellular types). This was specifically implemented in each feature by summing the average importance of feature for the prediction of all subcellular localization types, as represented in the [Methods of histgram and scatter plot](https://github.com/yujuan-zhang/feature-representation-for-LLMs/tree/main/Model/Model%20interpretation/Methods%20of%20histgram%20and%20scatter%20plot#methods-of-histgram-and-scatter-plot), to assess as each feature’s overall predictive importance. This differs from the swarm plot and the methodology in Table S8 (https://doi.org/10.1093/bib/bbad534), which calculates the average importance of all features for individual subcellular localization type. this represents a different approach and direction in explanatory analysis based on feature importance.

Additionally, differing from the analysis involved in the swarm plot, DNN model use in this analysis only includes RF_filter+MLP (with an input feature dimension of 3152), and we have also not considered conducting this analysis on non-homologous datasets and with different feature dimensions input-based model. This is primarily due to constraints related to the publication cycle and other time considerations. We encourage researchers to build upon our work and further explore interpretation studies on features extracted from different positional intervals or sites using large protein language models. We aim to maintain this Github project long-term, including the [figshare dataset](https://figshare.com/articles/dataset/feature-representation-for-LLMs/24312292), as a supplement to our research presented in the paper.

## Model Interpretation supplementary

In the main text and supplementary materials of our journal article, we discussed the similarities and differences between Tree SHAP, Deep Explainer, and Integrated Gradients (IG). The essence is to understand that they are all methods for feature attribution analysis. Moreover, they are not entirely model-agnostic; instead, they are applicable only to specific models. Additionally, their mathematical foundations lead to variations in how they calculate the contribution of different features and, subsequently, their importance analysis. Therefore, when using feature interpretability methods for studying feature representation capabilities, one should either cautiously select a feature attribution algorithm based on mathematics and statistics or, as we did in our paper, simulate a computation process that is not dependent on a specific model or feature attribution algorithms using different feature interpretability techniques and models. Furthermore, due to the publication cycle, the supplementary document 1 on Tree SHAP, Deep Explainer, and IG value might lack detailed descriptions. For more comprehensive information on these methods, please refer to the following resources:

* [SHAP API reference](https://shap-lrjball.readthedocs.io/en/latest/api.html)

* [SAGE introduction website](https://iancovert.com/blog/understanding-shap-sage/)

* [Original IG paper](https://arxiv.org/abs/1703.01365v2)

* "[Interpreting Machine Learning Models With SHAP](https://leanpub.com/shap)" book

## Go enrich for pho feature

To investigate whether the features extracted by a ESM2 model embed latent biological attributes and functions, this study examining whether phosphorylation features potentially reflect phosphorylation functions. Initially, proteins in the dataset are divided into different groups based on the distribution intervals of feature attribution values. Subsequently, Gene Ontology (GO) enrichment analysis is conducted on the proteins within these groups. This method achieves clustering of proteins based on the contribution of features and explores whether the GO enrichment results reflect some fundamental attributes of phosphorylation features, particularly those associated with phosphorylation function. This also reflects the potential biological representation mechanisms of phosphorylation embedded in the ESM2 model. Additionally, to ensure the robustness of the experiment, the candidate phosphorylation features are selected based on the top 10% of important feature(calculated by various feature importance measures) to ensure that the feature attribution interval division is representative.

Additionally, the protein UniProt ID needs to be converted to the ENTREZID (Gene ID) in GO enrich experiment.

The algorithm implementation idea refers to the work by Rui Qin et al. ( https://doi.org/10.1016/j.isci.2022.105163), which is based on using SHAP to select key predictive genes for glycosylation occurrence and conducting GO enrichment analysis on related genes. The assumption is that the features with core impacts on prediction are often the key factors influencing model decisions, and hence, these features might contain richer representational information. For a protein dataset, when key phosphorylation features are selected, based on the properties of SHAP and IG algorithms, the greater the feature attribution values, the more significant is the protein's contribution to the prediction, indicating stronger decision-making influence of the model. In other words, these proteins might possess richer representational information. Conducting GO enrichment for proteins with high feature attribution values can reflect the key biological function representational information of the features, which is highly related to model predictions.

However, this result requires a cautious interpretation. On one hand, drawing conclusions directly from GO enrichment results is limited, as noted by Kaumadi Wijesooriya (PMCID: PMC8936487 DOI: 10.1371/journal.pcbi.1009935) and James A Timmons (PMCID: PMC4561415 DOI: 10.1186/s13059-015-0761-7). On the other hand, this study does not perform tasks directly predicting phosphorylation sites or functions, which is a direction for future work. Therefore, these results should only be viewed as potential signals indicating that ESM2 captures biological functional representational information, with the primary aim being to provide a potential method and approach.

## UMAP enviroment set

For the current task, the UMAP library version used is 0.5.3. The parameters employed for UMAP visualization are as follows:

- `random_state`: 0
- `min_dist`: 0.5
- `n_neighbors`: 15

## Comparison model (UDSMProt, Doc2vec model, Deeploc2.0)

In this study, we employed UDSMProt, Doc2vec models (sequence2_doc2vec and sequence3_doc2vec), as well as Deeploc2.0 for comparative purposes with ESM2's DNN_Liner, MLP, and RF. For the construction methods of these models, please refer to Supplementary Document 1 of the article. If you require access to the relevant source code, please contact the authors.

## Citation

If our work has contributed to your research, we would greatly appreciate it if you could cite our work as follows. 

Zeyu Luo, Rui Wang, Yawen Sun, Junhao Liu, Zongqing Chen, Yu-Juan Zhang, Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction, *Briefings in Bioinformatics*, Volume 25, Issue 2, March 2024, bbad534, https://doi.org/10.1093/bib/bbad534.

If you are using the ESM-2 model in your project or research,  please refer to original work completed by the authors: Lin, Z., et al., Evolutionary-scale prediction of atomic level protein structure with a language model. bioRxiv, 2022: p. 2022.07.20.500902.

## Acknowledgments

we are acknowledge the contributions of the open-source community and the developers of the Python libraries used in this study.

## Related Works
If you are interested in feature extraction and model interpretation for RNA-seq foundation models (like scFoundation and geneformer), you may find our new work helpful:

scATD: [GitHub Repository](https://github.com/doriszmr/scATD)
