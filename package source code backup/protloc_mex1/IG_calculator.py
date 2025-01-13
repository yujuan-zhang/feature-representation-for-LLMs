


import numpy as np
import pandas as pd

try:
    import torch
    from captum.attr import IntegratedGradients
    from captum.attr import DeepLift
except ImportError:
    raise ImportError("Missing dependencies: Please install 'torch' and 'captum' manually.")




class IntegratedGradientsCalculator:

    def __init__(self, model, X_input, X_input_tensor,y_input,batch_size=100, n_steps=50):
        self.model = model
        self.X_input = X_input 
        self.X_input_tensor=X_input_tensor
        self.y_input = y_input
        self.batch_size=batch_size
        self.n_steps=n_steps
    def _compute_integrated_gradients_define(self,target_type):
        
        device = self.X_input_tensor.device  
        baseline = torch.zeros(self.X_input_tensor.shape).to(device)
        
        ig = IntegratedGradients(self.model)

        all_attributions = []
        all_deltas = []

        num_batches = (self.X_input_tensor.shape[0] + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.X_input_tensor.shape[0])
            X_batch = self.X_input_tensor[start_idx:end_idx]

            attributions, delta = ig.attribute(X_batch, baseline[start_idx:end_idx], 
                            target=target_type, return_convergence_delta=True,
                            n_steps=self.n_steps)

            attributions_np = attributions.cpu().detach().numpy()
            delta_np = delta.cpu().detach().numpy()

            all_attributions.append(attributions_np)
            all_deltas.append(delta_np)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        all_attributions = np.concatenate(all_attributions, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)
        
        return all_attributions, all_deltas
    
    def compute_integrated_gradients(self,type_class):
        
        self.all_attributions_values=[]
        self.all_deltas=[]
        for i in range(len(type_class)):
            all_attributions_values_inn, all_deltas_inn=self._compute_integrated_gradients_define(i)
            self.all_attributions_values.append(all_attributions_values_inn)
            self.all_deltas.append(all_deltas_inn)
        # self.all_attributions_values, self.all_deltas=list(map
        #                     (lambda x: self._compute_integrated_gradients_define(x),range(len(type_class))))
        
        self.all_attributions_values=list(map(lambda x: pd.DataFrame(x,index=self.X_input.index,columns=self.X_input.columns),self.all_attributions_values))
        self.all_deltas=list(map(lambda x: pd.DataFrame(x,index=self.X_input.index,columns=['approximation error']),self.all_deltas))
       
        self.all_attributions_values=dict(zip(type_class,self.all_attributions_values))
        self.all_deltas=dict(zip(type_class,self.all_deltas))
        
        return self.all_attributions_values, self.all_deltas
    def integrated_gradients_save(self,X_predict_input,save_path):
        all_attributions_value_save = {key: value.join(self.y_input, how="inner") for key,value in self.all_attributions_values.items()}
        all_attributions_value_save = {key: value.join(X_predict_input, how="inner") for key, value in all_attributions_value_save.items()}
        for key,value in all_attributions_value_save.items():
            file_path = f"{save_path}{key}_integrated_gradients_value.csv"
            value.to_csv(file_path, index=True)
        for key,value in self.all_deltas.items():
            file_path = f"{save_path}{key}_approximation_error_value.csv"
            value.to_csv(file_path, index=True)
            
        return all_attributions_value_save
        
        
        
####-----不应该计算输入数据的中位数，需要先用训练集的中位数，然后再按输入数据进行填充
class DeepliftCalculator:

    def __init__(self, model, X_input, X_input_tensor,y_input,batch_size=100):
        self.model = model
        self.X_input = X_input 
        self.X_input_tensor=X_input_tensor
        self.y_input = y_input
        self.batch_size=batch_size
        
    def _compute_deepliftvalue_define(self,target_type):
        
        device = self.X_input_tensor.device  
        # 在样本维度上计算 X_input 的均值作为基线
        baseline = np.mean(self.X_input.values, axis=0,keepdims=True)

        # 使用 np.broadcast_to 将基线扩展为与 X_input 相同的形状
        baseline = np.broadcast_to(baseline, self.X_input.shape)

        # 将 numpy array 转换为 PyTorch 张量
        baseline = torch.Tensor(baseline).to(device)

        
        dl = DeepLift(self.model)

        all_attributions = []
        all_deltas = []

        num_batches = (self.X_input_tensor.shape[0] + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.X_input_tensor.shape[0])
            X_batch = self.X_input_tensor[start_idx:end_idx]

            attributions, delta = dl.attribute(X_batch, baseline[start_idx:end_idx], 
                            target=target_type, return_convergence_delta=True)

            attributions_np = attributions.cpu().detach().numpy()
            delta_np = delta.cpu().detach().numpy()

            all_attributions.append(attributions_np)
            all_deltas.append(delta_np)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        all_attributions = np.concatenate(all_attributions, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)
        
        return all_attributions, all_deltas
    
    def compute_deepliftvalue(self,type_class):
        
        self.all_attributions_values=[]
        self.all_deltas=[]
        for i in range(len(type_class)):
            all_attributions_values_inn, all_deltas_inn=self._compute_deepliftvalue_define(i)
            self.all_attributions_values.append(all_attributions_values_inn)
            self.all_deltas.append(all_deltas_inn)
        # self.all_attributions_values, self.all_deltas=list(map
        #                     (lambda x: self._compute_integrated_gradients_define(x),range(len(type_class))))
        
        self.all_attributions_values=list(map(lambda x: pd.DataFrame(x,index=self.X_input.index,columns=self.X_input.columns),self.all_attributions_values))
        self.all_deltas=list(map(lambda x: pd.DataFrame(x,index=self.X_input.index,columns=['approximation error']),self.all_deltas))
       
        self.all_attributions_values=dict(zip(type_class,self.all_attributions_values))
        self.all_deltas=dict(zip(type_class,self.all_deltas))
        
        return self.all_attributions_values, self.all_deltas
    def deepliftvalue_save(self,X_predict_input,save_path):
        all_attributions_value_save = {key: value.join(self.y_input, how="inner") for key,value in self.all_attributions_values.items()}
        all_attributions_value_save = {key: value.join(X_predict_input, how="inner") for key, value in all_attributions_value_save.items()}
        for key,value in all_attributions_value_save.items():
            file_path = f"{save_path}{key}_deeplift_value.csv"
            value.to_csv(file_path, index=True)
        for key,value in self.all_deltas.items():
            file_path = f"{save_path}{key}_approximation_error_value.csv"
            value.to_csv(file_path, index=True)
            
        return all_attributions_value_save
        
             
        
        
        
        
        
        
        
        
        
        
        
        
        