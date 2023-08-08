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

The feature representation used the pre-trained protein model ESM2 developed by Meta company and placed on Hugging Face. For more details, please search in https://huggingface.co/facebook/esm2_t6_8M_UR50D.  

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
