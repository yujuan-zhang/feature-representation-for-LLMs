### DNN model training

In our paper, we have built a subcellular localization classification model based on 'RF_filter + feature all' (dim = 3152) using Deep Neural Networks (DNN). In the following, we provide detailed explanations on data augmentation techniques (using the `imblearn` package), model training hyperparameter optimization (with the `optuna` package) and saving the model weight, classification label mapping dictionary. It's important to note that data augmentation is only applied to the training set.

1. Load `train_data`. Due to GitHub's storage limitations, the `train_data` presented here consists of input features identical to those used in the DNN model and has already been normalized. However, it only contains 20 protein sequences and does not retain the label column, thereby not representing the complete training set for the model. For reproducing the full 'feature all' feature training set, one may refer to the previous steps, or contact the authors for further information.

   

   ```python
   import os
   import pandas as pd
   import numpy as np
   
   save_dir = "./Model/ESM2_feature_all/DNN_model_param"
   
   save_path = "<YOUR_PATH_HERE>"
   train_data = pd.read_excel(os.path.join(save_dir, 'train_ESM2_feature_all_DNN.xlsx'))
   
   X_train_scale= train_data.drop('label',axis=1)
   y_train=train_data.loc[:,'label']
   ```

   

2. Define the DNN model, this framework should be consistent with [the DNN model used for inference](https://github.com/yujuan-zhang/feature-representation-for-LLMs).

   ```python
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

3. Data augmentation is performed through the `imblearn` library, specifically utilizing the Synthetic Minority Over-sampling Technique (SMOTE) to address the issue of class imbalance in classification problems. This method synthesizes new minority class samples to bolster the sample size for underrepresented classes. The SMOTE algorithm functions by randomly selecting a sample from the minority class and identifying its k-nearest neighbors (with a default value of `k_neighbors=5`). Subsequently, one neighbor is randomly chosen, and a new sample is generated on the line segment joining the selected sample and its neighbor. This process is repeated until the desired number of minority class samples are produced to alleviate the issue of class imbalance. Implementation is achieved through the `imblearn.over_sampling` module, where the `sampling_strategy` parameter specifies the target number of samples for each class. Setting `random_state = 42` ensures the repeatability of the results.

   ```python
   from imblearn.over_sampling import SMOTE
   
   sampling_strategy = {"Nucleus": 1847, "Cytoplasm": 1500, "Cell membrane": 1500,
                       "Secreted": 1500, "Mitochondrion": 1500, "Endoplasmic reticulum": 1500,
                       "Golgi apparatus": 1500, "Cell projection": 1500, "Lysosome" : 1500,
                        "Cell junction" : 1500 }
   
   smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
   
   # Oversampling is performed on the training data 
   X_train_res, y_train_res = smote.fit_resample(X_train_scale, y_train)
   
   ```



4. For training the DNN model, the `Adam` optimizer is used to adjust the learning weights. The batch size is set to 32, and the model is trained for 80 epochs. The model is designed to classify data into 10 different labels. Additionally, the augmented training set `X_train_res` is further partitioned into training and validation subsets. These subsets are utilized for the model's training and for hyperparameter optimization using the `optuna` package.

   ```python
   ##model inference and training
   
   from torch.utils.data import DataLoader, TensorDataset
   import torch.optim as optim
   
   import optuna
   
   from torch.optim import Adam
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   # Configuration
   input_dim = X_train_res.shape[1] ##the dim of 'feature all' = 3152 
   
   batch_size = 32
   epochs = 80
   num_classes=10
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Train-Test Split
   X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_train_res, y_train_res,test_size=0.2, random_state=0)
   
   # Create a label encoder
   le = LabelEncoder()
   
   # Fit the label encoder and transform the labels
   y_train_res = le.fit_transform(y_train_res)
   y_val_res = le.transform(y_val_res)
   
   # Create a mapping of encoded classes to original labels
   label_mapping = pd.DataFrame({
       'EncodedLabel': range(len(le.classes_)),
       'OriginalLabel': le.classes_
   })
   ##save label_mapping for next inference step
   label_mapping.to_excel(os.path.join(save_path,'label2number.xlsx'),index=False)
   
   # Convert to PyTorch datasets
   train_dataset = TensorDataset(torch.Tensor(X_train_res.values), torch.LongTensor(y_train_res))
   
   val_dataset = TensorDataset(torch.Tensor(X_val_res.values), torch.LongTensor(y_val_res))
   
   # Define data loaders
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
   ```

5. This step is the model training process and the hyperparameter optimization. The hyperparameters to be learned are the hidden dimension `hidden_dim` and the learning rate `lr`. Due to the similarity of the process and optimization objectives with [VAE training detail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/VAE%20model/Res_VAE%20training%20detail.md
   ), further details are not elaborated upon here.
   
   ```python
   # Define an objective function to be minimized.
   
   min_loss = float('inf')
   best_model = None
   best_hidden_dim = None
   best_lr = None
   def objective(trial):
       global best_model, min_loss, best_lr, best_hidden_dim
       # Suggest values for the hyperparameters:
       hidden_dim = trial.suggest_int('hidden_dim', 250, 1000)
       lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
   
       model = ClassificationDNN(input_dim, hidden_dim, num_classes).to(device)
       
       # Optimizer
       optimizer = Adam(model.parameters(), lr=lr)
   
       #criterion = nn.NLLLoss() # Negative log likelihood loss
   
       
   
       # Training loop
       for epoch in range(epochs):
           model.train()
   
           for inputs, labels in train_loader:
               inputs, labels = inputs.to(device), labels.to(device)
   
               # Forward pass
               outputs = model(inputs)
   
               # Compute loss
               loss = model.compute_loss(outputs, labels)
   
               # Backward pass and optimization
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   
           # Validation
           model.eval()
           val_loss = 0
   
           for inputs, labels in val_loader:
               inputs, labels = inputs.to(device), labels.to(device)
   
               with torch.no_grad():
                   outputs = model(inputs)
               
               loss = model.compute_loss(outputs, labels)
               val_loss += loss.item()
   
           # Calculate average validation loss per sample
           val_loss = val_loss / len(val_loader.dataset)
           
           # Prune unpromising trials
           trial.report(val_loss, epoch)
           if trial.should_prune():
               raise optuna.exceptions.TrialPruned()
   
   
           # Save the best model and parameters
           if val_loss < min_loss:
              min_loss = val_loss
              best_model = model
              best_hidden_dim = hidden_dim
              best_lr = lr
   
   
       # Return the best validation loss
       return val_loss
   
   # Create a study object and optimize the objective function.
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=30)
   
   # Print the result
   trial = study.best_trial
   ```
   
6. The optimal model, denoted as `best_model`, has its weight parameters saved. Additionally, some information about the optimal hyperparameters is also preserved. The optimal hyperparameters information for the 'feature all' DNN model are included at the end.

   ```python
   
   ##Save Model Parameters 
   torch.save(best_model.state_dict(), os.path.join(save_path,'model_parameters.pt'))
   
   ##Save Model Hyperparameters
   # Create a dictionary to map parameter names to their values
   
   results = {
       'Best hidden_dim': [best_hidden_dim],
       'Best learning rate': [best_lr],
       'Best eval loss': [min_loss]
   }
   
   # Create a DataFrame
   df = pd.DataFrame(results)
   
   # Write the DataFrame to an Excel file
   df.to_excel(os.path.join(save_path,'model_optimization_results.xlsx'), index=False)
   
   ```
   
   | model           | Best hidden_dim | Best learning rate | Best eval loss | epoch | input_dim |
   | :-------------- | :-------------- | ------------------ | :------------- | :---- | --------- |
   | feature_all_DNN | 802             | 0.000012801        | 0.00393        | 80    | 3152      |