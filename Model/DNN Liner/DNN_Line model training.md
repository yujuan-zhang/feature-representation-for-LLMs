### DNN_Line model training

In the context of our research, we have developed a subcellular localization classification model. This model is based on a feature set that includes 'feature all' (with a dimension of 6418), 'cls,' 'eos,' 'mean,' 'segment0-9,' 'pho,' and other relevant features. The core architecture of the model is a deep neural network (DNN_Line) with the inclusion of a linear adaptation layer.

1. This step is similar to the first step of the DNN model, so we won't delve into it in great detail.

   ```python
   import os
   import pandas as pd
   import numpy as np
   
   save_path = "<YOUR_PATH_HERE>"
   train_data = "Please use our Swiss data available on Figshare"
   
   train_data = data.set_index("ID")
   X_train_scale= train_data.drop('label',axis=1)
   y_train=train_data.loc[:,'label']
   ```
   
2. Define the DNN_Line model.

   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   
   class DNNLine(nn.Module):
       def __init__(self, input_dim, num_classes):
           super().__init__()
           
           # Fully connected layers
           self.fc1 = nn.Linear(input_dim, num_classes)
           
           # Loss function
           self.criterion = nn.NLLLoss()
   
       def forward(self, x):
           x = self.fc1(x)
           
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

3. For training the DNN_Line model, the `Adam` optimizer is used to adjust the learning weights. This step is similar to the fourth step of the DNN model, so we won't go into extensive detail.

   ```python
   ##model inference and training
   
   from torch.utils.data import DataLoader, TensorDataset
   import torch.optim as optim
   
   import optuna
   
   from torch.optim import Adam
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   # Configuration
   input_dim = X_train_scale.shape[1] ##the dim of 'feature all' = 3152 
   
   batch_size = 32
   epochs = 80
   num_classes=10
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Train-Test Split
   X_train, X_val, y_train, y_val = train_test_split(X_train_scale, y_train,test_size=0.2, random_state=0)
   
   # Create a label encoder
   le = LabelEncoder()
   
   # Fit the label encoder and transform the labels
   y_train = le.fit_transform(y_train)
   y_val = le.transform(y_val)
   
   # Create a mapping of encoded classes to original labels
   label_mapping = pd.DataFrame({
       'EncodedLabel': range(len(le.classes_)),
       'OriginalLabel': le.classes_
   })
   ##save label_mapping for next inference step
   label_mapping.to_excel(os.path.join(save_path,'label2number.xlsx'),index=False)
   
   # Convert to PyTorch datasets
   train_dataset = TensorDataset(torch.Tensor(X_train.values), torch.LongTensor(y_train))
   val_dataset = TensorDataset(torch.Tensor(X_val.values), torch.LongTensor(y_val))
   
   # Define data loaders
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   ```

4. This step is the model training process and the hyperparameter optimization. The hyperparameters to be learned are the hidden dimension `hidden_dim` and the learning rate `lr`. Due to the similarity of the process and optimization objectives with the step 5 of the DNN model, further details are not elaborated upon here.

   ```python
   # Define an objective function to be minimized.
   
   def objective(trial):
       global best_model, min_loss, best_lr#, best_hidden_dim
       # Suggest values for the hyperparameters:
       #hidden_dim = trial.suggest_int('hidden_dim', 250, 1000)
       lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
   
       model = DNNLine(input_dim, num_classes).to(device)
       
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
   
           trial.set_user_attr('best_model_state_dict', model.state_dict())
           trial.set_user_attr('best_epoch', epoch)
   
   
       # Return the best validation loss
       return val_loss
   
   # Create a study object and optimize the objective function.
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=30)
   
   # Print the result
   trial = study.best_trial
   
   best_lr = trial.params['lr']  # Get the best learning rate
   best_model_state_dict = trial.user_attrs.get('best_model_state_dict')  # Get the best model state dict
   
   best_loss = trial.value
   ```

5. The optimal model, denoted as `best_model`, has its weight parameters saved. Additionally, some information about the optimal hyperparameters is also preserved. The optimal hyperparameters information for the DNN_Line model are included at the end.

   ```python
   ##Save Model Parameters 
   torch.save(best_model_state_dict, os.path.join(save_path,'model_parameters.pt'))
   
   ##Save Model Hyperparameters
   # Create a dictionary to map parameter names to their values
   
   results = {
       #'Best hidden_dim': [best_hidden_dim],
       'Best learning rate': [best_lr],
       'Best eval loss': [best_loss]
   }
   
   # Create a DataFrame
   df = pd.DataFrame(results)
   
   # Write the DataFrame to an Excel file
   df.to_excel(os.path.join(save_path,'model_optimization_results.xlsx'), index=False)
   ```
