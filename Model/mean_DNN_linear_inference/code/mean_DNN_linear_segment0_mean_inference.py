# 
# # Import necessary libraries and modules
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch
# import pandas as pd
# from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor
# import os
# import re
# 
# # 获取open_path目录下的所有文件名
# 
# open_path = './data/in'
# save_path = './output'
# local_model_file ='./data/local_model'
# local_DNN_linear_model = './data/DNN_linear_segment0_mean_model'
# 
# files = os.listdir(open_path)
# 
# # 假设目录中只有一个文件，且这个文件是我们感兴趣的文件
# # 提取文件名（去除.xlsx后缀）
# species_name = files[0].replace('.xlsx', '')
# 
# # 让用户选择是使用本地模型还是下载模型
# choice = input("Do you want to use a local model or download one? Type 'local' or 'download': ").strip().lower()
# 
# # 基于用户的选择，使用相应的模型
# if choice == 'local':
#     # 用户选择使用本地模型
#     base_path = local_model_file
#     tokenizer = AutoTokenizer.from_pretrained(base_path)
#     model = AutoModelForMaskedLM.from_pretrained(base_path, output_hidden_states=True)
# elif choice == 'download':
#     # 用户选择下载模型
#     model_name = "facebook/esm2_t33_650M_UR50D"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
# else:
#     print("Invalid choice. Please type 'local' or 'download'.")
# 
# 
# 
# 
# # Create a DataFrame containing protein sequences
# protein_sequence_df= pd.read_excel(open_path+'/'+species_name+'.xlsx')
# 
# # Initialize the feature extractor. If you wish to directly utilize 'cuda'/'cpu', simply replace 'auto' with 'cuda'/'cpu'
# feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
#                                                    compute_cls=False, compute_eos=False, compute_mean=False, compute_segments=True,
#                                                    #device_choose = 'auto'
#                                                    )
# 
# # Perform feature extraction on the protein sequences
# protein_sequence_df_represent = feature_extractor.get_last_hidden_features_combine(protein_sequence_df, sequence_name='Sequence', batch_size= 1)
# 
# 
# # Extract column names that match the pattern 'ESM2_clsX'
# cols = [col for col in protein_sequence_df_represent.columns if re.match(r'ESM2_segment0_mean\d+', col)]
# 
# # Set the DataFrame index to 'Entry' for human representation and create a sub-DataFrame for cls columns
# protein_sequence_df_represent = protein_sequence_df_represent['Entry', 'Sequence', cols]
# 
# 
# ##保存特征结果
# protein_sequence_df_represent.to_excel(save_path+'/'+species_name+'_feature_rep.xlsx',index=False)
# 
# # print('run mean_DNN_linear_inference.py success, feature representation outcome are deploted in ./mean_DNN_linear_inference/output')
# 
# ####定义DNN linear的框架
# 
# # import torch
# import torch.nn as nn
# import torch.nn.functional as F
# 
# class DNNLine(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         
#         # Fully connected layers
#         self.fc1 = nn.Linear(input_dim, num_classes)
#         
#         # Loss function
#         self.criterion = nn.NLLLoss()
# 
#     def forward(self, x):
#         x = self.fc1(x)
#         
#         return F.log_softmax(x, dim=1)  # Apply Log Softmax for multi-class classification
# 
#     def compute_loss(self, outputs, targets):
#         return self.criterion(outputs, targets)
#     
#     def model_infer(self, X_data, device):
#         self.eval()
# 
#         input_data = torch.Tensor(X_data.values).to(device) # or your test data
# 
#         with torch.no_grad():
#             predictions = self(input_data)
#             
#         predictions = predictions.exp()
#         _, predicted_labels = torch.max(predictions, 1)
# 
#         predicted_labels = predicted_labels.cpu().numpy()
#         probabilities = predictions.cpu().numpy()
#         return predicted_labels, probabilities
# 
# ##进行DNN linear推断
# # Configuration
# input_dim = 1280 ##the dim of 'feature all' 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes=10
# 
# 
# # Load parameters
# load_model=DNNLine(input_dim=input_dim,num_classes=num_classes).to(device)
# load_model.load_state_dict(torch.load(os.path.join(local_DNN_linear_model,'model_parameters.pt')))
# load_model.eval()  # Set the model to evaluation mode
# 
# 
# # 将 'Entry' 设置为索引
# protein_sequence_df_represent.set_index('Entry', inplace=True)
# 
# # 删除 'Sequence' 列
# protein_sequence_df_represent.drop('Sequence', axis=1, inplace=True)
# 
# label_mapping=pd.read_excel(os.path.join(local_DNN_linear_model,'label2number.xlsx'))
# 
# # Convert DataFrame to Dictionary
# label_dict = dict(zip(label_mapping['EncodedLabel'], label_mapping['OriginalLabel']))
# 
# X_inference_data_hat,X_inference_data_probabilities=load_model.model_infer(protein_sequence_df_represent,device=device)
# 
# X_inference_data_hat = [label_dict[i] for i in X_inference_data_hat]
# 
# 
# # Build classifier and perform evaluation
# # Convert the prediction results to DataFrame
# X_inference_data_hat = pd.DataFrame(X_inference_data_hat, columns=["predict_topic"],index = protein_sequence_df_represent.index)
# 
# # 将X_inference_data_probabilities列表转换为DataFrame
# X_inference_data_probabilities_max = [max(probs) for probs in X_inference_data_probabilities]
# 
# X_inference_data_probabilities_df = pd.DataFrame(X_inference_data_probabilities_max, columns=['predict_probability'],index = protein_sequence_df_represent.index)
# 
# X_inference_data_hat = pd.concat([X_inference_data_hat, X_inference_data_probabilities_df], axis=1)
# 
# X_inference_data_hat.to_excel(save_path+'/'+species_name+'_prediction.xlsx')
# # classes=label_mapping.loc[:,'OriginalLabel'].values
# 
# print('run mean_DNN_linear_inference.py success, feature representation and prediction outcome are deploted in ./mean_DNN_linear_inference/output')

# Import necessary libraries and modules
import os
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor

# Set directories and file paths
open_path = './data/in'
save_path = './output'
local_model_file = './data/local_model'
local_DNN_linear_model = './data/DNN_linear_segment0_mean_model'

# Get the list of files in the 'open_path' directory
files = [file for file in os.listdir(open_path) if file.endswith('.xlsx')]

# Assume there is only one .xlsx file in the directory and that it's the file we are interested in
species_name = files[0].replace('.xlsx', '')

# Allow user to choose between using a local model or downloading one
choice = input("Do you want to use a local model or download one? Type 'local' or 'download': ").strip().lower()

# Initialize the tokenizer and model based on user's choice
if choice == 'local':
    base_path = local_model_file
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForMaskedLM.from_pretrained(base_path, output_hidden_states=True)
elif choice == 'download':
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
else:
    print("Invalid choice. Please type 'local' or 'download'.")

# Load protein sequences into DataFrame
protein_sequence_df = pd.read_excel(open_path + '/' + species_name + '.xlsx')

# Initialize feature extractor with specified options
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=False, compute_eos=False, compute_mean=False,
                                                   compute_segments=True)

# Extract features from protein sequences
protein_sequence_df_represent = feature_extractor.get_last_hidden_features_combine(protein_sequence_df,
                                                                                   sequence_name='Sequence',
                                                                                   batch_size=1)

# Filter columns containing 'ESM2_segment0_mean' in their names
cols = [col for col in protein_sequence_df_represent.columns if re.match(r'ESM2_segment0_mean\d+', col)]


# Select specific columns and set 'Entry' as index for the representation DataFrame
protein_sequence_df_represent = protein_sequence_df_represent[['Entry', 'Sequence'] + cols]

# Save extracted features to an Excel file
protein_sequence_df_represent.to_excel(save_path + '/' + species_name + '_feature_rep.xlsx', index=False)
print("Now, you have successfully obtained the 'segment0_mean' feature, which has a dimension of 1280.")

#### Define DNN linear model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNLine(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, num_classes)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)  # Apply Log Softmax for multi-class classification

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def model_infer(self, X_data, device):
        self.eval()

        input_data = torch.Tensor(X_data.values).to(device)

        with torch.no_grad():
            predictions = self(input_data)

        predictions = predictions.exp()
        _, predicted_labels = torch.max(predictions, 1)

        predicted_labels, probabilities = predicted_labels.cpu().numpy(), predictions.cpu().numpy()
        return predicted_labels, probabilities


## Perform DNN linear inference
input_dim = 1280  # The dimension of 'feature all'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

# Load model parameters
load_model = DNNLine(input_dim=input_dim, num_classes=num_classes).to(device)
load_model.load_state_dict(torch.load(os.path.join(local_DNN_linear_model, 'model_parameters.pt')))
load_model.eval()

# Prepare data for inference
protein_sequence_df_represent.set_index('Entry', inplace=True)
protein_sequence_df_represent.drop('Sequence', axis=1, inplace=True)

# Load label mapping
label_mapping = pd.read_excel(os.path.join(local_DNN_linear_model, 'label2number.xlsx'))

# Convert label mapping to dictionary
label_dict = dict(zip(label_mapping['EncodedLabel'], label_mapping['OriginalLabel']))

# Perform inference
X_inference_data_hat, X_inference_data_probabilities = load_model.model_infer(protein_sequence_df_represent,
                                                                              device=device)

# Map encoded labels to original labels
X_inference_data_hat = [label_dict[i] for i in X_inference_data_hat]

# Convert prediction results to DataFrames
X_inference_data_hat_df = pd.DataFrame(X_inference_data_hat, columns=["predict_topic"],
                                       index=protein_sequence_df_represent.index)

X_inference_data_probabilities_max = [max(probs) for probs in X_inference_data_probabilities]
X_inference_data_probabilities_df = pd.DataFrame(X_inference_data_probabilities_max, columns=['predict_probability'],
                                                 index=protein_sequence_df_represent.index)

# Concatenate the two DataFrames
X_inference_data_hat = pd.concat([X_inference_data_hat_df, X_inference_data_probabilities_df], axis=1)

# Save the predictions to an Excel file
X_inference_data_hat.to_excel(save_path + '/' + species_name + '_prediction.xlsx')

print('run mean_DNN_linear_inference.py success, feature representation and prediction outcome are deploted in ./mean_DNN_linear_inference/output')