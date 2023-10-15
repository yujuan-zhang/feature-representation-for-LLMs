### Residual Variational Autoencoder (Res-VAE) Training Process

1. Preparing the Training Dataset: The dataset for this experiment is generated using cls features represented by ESM2 through the protloc-mex-x package. Other features can also be subjected to similar experiments. Additionally, it is crucial to ensure that the training dataset does not overlap with the test set used for downstream classification tasks to prevent data leakage. Finally, the training dataset should be partitioned into a training set and a validation set for the training and validation of the Residual Variational Autoencoder (Res-VAE) model.

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   
   data = 'your ESM2 cls data'
   # Train-Test Split
   X_train, X_val = train_test_split(data, test_size=0.2, random_state=0)
   ```

2. Divide the data into batches using DataLoader and prepare it for input into the model for computation, with the batch size set to 32.

   ```python
   from torch.utils.data import DataLoader, TensorDataset
   import torch.optim as optim
   import torch
   
   input_dim = X_train.shape[1]
   batch_size = 32
   epochs = 80
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Convert to PyTorch datasets
   train_dataset = TensorDataset(torch.Tensor(X_train.values))
   val_dataset = TensorDataset(torch.Tensor(X_val.values))
   
   # Define data loaders
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
   ```

3. Defining Res_VAE model and get ready for training. 

   ```python
   
   ##vision2
   import torch
   from torch import nn
   from torch.nn import functional as F
   
   class ContinuousResidualVAE(nn.Module):
       class ResBlock(nn.Module):
           def __init__(self, in_dim, out_dim):
               super().__init__()
               self.fc = nn.Linear(in_dim, out_dim)
               self.bn = nn.BatchNorm1d(out_dim)
               self.dropout = nn.Dropout(0.3)
               if in_dim != out_dim:
                   self.downsample = nn.Linear(in_dim, out_dim)
               else:
                   self.downsample = None
   
           def forward(self, x):
               out = F.leaky_relu(self.bn(self.fc(x)))
               out = self.dropout(out)
               if self.downsample is not None:
                   x = self.downsample(x)
               return out + x
       
   
       def __init__(self, input_dim, hidden_dim, z_dim,loss_type='RMSE',reduction='sum'):
           super().__init__()
           # Encoder
           self.fc1 = nn.Linear(input_dim, hidden_dim)
           self.bn1 = nn.BatchNorm1d(hidden_dim)
           self.dropout1 = nn.Dropout(0.3)
           self.resblock1 = self.ResBlock(hidden_dim, hidden_dim // 2)
           self.resblock2 = self.ResBlock(hidden_dim // 2, hidden_dim // 4)
           # Latent space
           self.fc21 = nn.Linear(hidden_dim // 4, z_dim)  # mu layer
           self.fc22 = nn.Linear(hidden_dim // 4, z_dim)  # logvariance layer
           # Decoder
           self.fc3 = nn.Linear(z_dim, hidden_dim // 4)
           self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
           self.dropout3 = nn.Dropout(0.3)
           self.resblock3 = self.ResBlock(hidden_dim // 4, hidden_dim // 2)
           self.resblock4 = self.ResBlock(hidden_dim // 2, hidden_dim)
           self.fc4 = nn.Linear(hidden_dim, input_dim)
           
           # Add attributes for loss type and reduction type
           self.loss_type = loss_type
           self.reduction = reduction
           
           if reduction not in ['mean', 'sum']:
               raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)
   
   
       def encode(self, x):
           h = F.leaky_relu(self.bn1(self.fc1(x)))
           h = self.dropout1(h)
           h = self.resblock1(h)
           h = self.resblock2(h)
           return self.fc21(h), self.fc22(h)  # mu, logvariance
   
       def reparameterize(self, mu, logvar):
           std = torch.exp(0.5 * logvar)
           eps = torch.randn_like(std)##取形状与std相同，且平均值为0，标准差为1的正态分布中进行采样
           return mu + eps * std
   
       def decode(self, z):
           h = F.leaky_relu(self.bn3(self.fc3(z)))
           h = self.dropout3(h)
           h = self.resblock3(h)
           h = self.resblock4(h)
           return self.fc4(h) # No sigmoid here
   
       def forward(self, x):
           mu, logvar = self.encode(x.view(-1, x.shape[1]))
           z = self.reparameterize(mu, logvar)
           return self.decode(z), mu, logvar
   
       def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
           
           if self.loss_type == 'MSE':
               self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction)
           elif self.loss_type == 'RMSE':
               self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction))
           else:
               raise ValueError(f'Invalid loss type: {self.loss_type}')
   
           if self.reduction == 'mean':
               self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
           else: 
               self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   
           return beta * self.REC + self.KLD
       
   
   
       def print_neurons(self):
           print("Encoder neurons:")
           print(f"Input: {self.fc1.in_features}, Output: {self.fc1.out_features}")
           print(f"ResBlock1 - Input: {self.resblock1.fc.in_features}, Output: {self.resblock1.fc.out_features}")
           print(f"ResBlock2 - Input: {self.resblock2.fc.in_features}, Output: {self.resblock2.fc.out_features}")
           
           print("Latent neurons:")
           print(f"mu - Input: {self.fc21.in_features}, Output: {self.fc21.out_features}")
           print(f"logvar - Input: {self.fc22.in_features}, Output: {self.fc22.out_features}")
   
           print("Decoder neurons:")
           print(f"Input: {self.fc3.in_features}, Output: {self.fc3.out_features}")
           print(f"ResBlock3 - Input: {self.resblock3.fc.in_features}, Output: {self.resblock3.fc.out_features}")
           print(f"ResBlock4 - Input: {self.resblock4.fc.in_features}, Output: {self.resblock4.fc.out_features}")
           print(f"Output: {self.fc4.in_features}, Output: {self.fc4.out_features}")
   
       def get_model_inference_z(self, x, seed=None):
           """
           This function takes input x and returns the corresponding latent vectors z.
           If a seed is provided, it is used to make the random number generator deterministic.
           """
           self.eval()  # switch to evaluation mode
           if seed is not None:
               torch.manual_seed(seed)
           with torch.no_grad():  # disable gradient computation
               mu, logvar = self.encode(x.view(-1, x.shape[1]))
               z = self.reparameterize(mu, logvar)
           return z
   
   ```

   4. Begin model training. In our experiment, the `beta` value controlling the `REC` loss is fixed at 1000. This value was determined through various experiments, where we found that a `beta` of 1000 yielded the lowest loss and the hidden layer `z` features performed best when used with UMAP. Additionally, the settings for `loss_type='MSE'` and `reduction='sum'` are also based on experimental results, not model auto-tuning. The number of hidden units `z_dim`, the size of the first layer in the Encoder and the last layer in the Decoder `hidden_dim`, and the initial learning rate `lr` will be optimized by the `Optuna` package. Details on the training scenarios for `ESM2_650m_cls`, <model1>, and <model2> will be presented later. The  optimization methods used by `Optuna` are as follows,

      **ESM2_650m_cls：**

      The goal of `Optuna` hyperparameter optimization is to minimize the loss on the validation set `X_val`.

      Search Space: The hyperparameters `z_dim` (ranging from 10 to 50), `hidden_dim` (ranging from 250 to 1000), and `lr` (ranging from 1e-5 to 1e-3) are being optimized.

      Optimization Algorithm: The TPE (Tree-structured Parzen Estimator) algorithm is used for hyperparameter optimization.

      Pruning: Early stopping is implemented to conserve computational resources. If a trial performs poorly in the early stages, it will be terminated prematurely.

      Epochs: Each experiment runs for a maximum of 80 epochs.

      Number of Trials: Optuna will conduct 50 rounds of trials to find the optimal hyperparameter combination within the specified range.

      ```python
      min_loss = float('inf')
      best_model = None
      best_dim = None
      best_hidden_dim = None
      best_lr = None
      
      def objective(trial):
          global best_model, min_loss, best_dim, best_lr, best_hidden_dim #, best_gamma, best_warmup
      
          # Define hyperparameters
          z_dim = trial.suggest_int('z_dim', 10, 50)
          hidden_dim = trial.suggest_int('hidden_dim', 250, 1000)
          lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
          
          # Create model
          model = ContinuousResidualVAE(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim, loss_type='MSE',reduction='sum').to(device)
      
          # Optimizer
          optimizer = Adam(model.parameters(), lr=lr)
      
         
          # Training loop
          for epoch in range(epochs):
              model.train()
              for batch_idx, batch in enumerate(train_loader):
                  data = batch[0].to(device)
                  optimizer.zero_grad()           
                  recon_batch, mu, logvar = model(data)
                  loss = model.loss_function(recon_batch, data, mu, logvar, beta=1000)
                  loss.backward()
                  optimizer.step()
      
                  # # Step the scheduler
                  # scheduler.step()
      
              # Validation loop
              model.eval()
              val_loss = 0
              with torch.no_grad():  # disable gradient computation
                  for batch_idx, batch in enumerate(val_loader):
                      data = batch[0].to(device)
                      recon_batch, mu, logvar = model(data)
                      val_loss += model.loss_function(recon_batch, data, mu, logvar, beta=1000).item() * data.size(0)
      
              # Get the average validation loss
              val_loss /= len(val_loader.dataset)
      
              # Prune unpromising trials
              trial.report(val_loss, epoch)
              if trial.should_prune():
                  raise optuna.exceptions.TrialPruned()
      
              # Save the best model and parameters
              if val_loss < min_loss:
                  min_loss = val_loss
                  best_model = model
                  best_dim = z_dim
                  best_hidden_dim = hidden_dim
                  best_lr = lr
                  # best_gamma = gamma
                  # best_warmup = warmup
      
          return val_loss
      
      study = optuna.create_study(direction='minimize')
      study.optimize(objective, n_trials=50)
      ```

   5. The trained model from above can be saved and used to extract the hidden layer `z` features for both the training data and new data such as the test set. For more details, see the main [VAE documentation page](https://github.com/yujuan-zhang/feature-representation-for-LLMs).

   6. Training performance and optimal hyperparameters for ESM2_650m_cls, 

      | model     | Best z_dim | Best hidden_dim | Best learning rate | Best eval loss | lose_type | epoch | input_dim |
      | :-------- | ---------- | :-------------- | ------------------ | :------------- | :-------- | :---- | --------- |
      | ESM2_650M | 18         | 859             | 0.000469           | 22509.06       | MSE       | 80    | 1280      |

It is worth noting that higher loss on the validation set is expected due to the use of Mean Squared Error (MSE) with a multiplicative factor of `beta` set to 1000. Additionally, we employ a `sum` reduction strategy, summing up the Reconstruction (REC) and Kullback-Leibler Divergence (KLD) losses across both the batch and feature dimensions. With a batch size of 32 and an input dimension of 1280 for ESM2_650M, elevated loss values are anticipated.

Moreover, it's important to highlight that we did not normalize our training data. This decision was made considering the relatively small scale variance in the features extracted and represented by ESM2. Our Res_VAE model does not constrain the output range, as the final output layer does not employ an activation function, allowing for a match with the numerical range of ESM2 features. Given concerns for model interpretability and the adaptability of downstream predictive models to normalized data, we chose not to standardize the data here (In fact, we will use z-score in the data augmentation stage， for detail see DNN/RF model training). This approach also avoids the necessity to normalize all the data when using the model for hidden feature extraction, which would have been required if we had normalized the data during model training.

​     



