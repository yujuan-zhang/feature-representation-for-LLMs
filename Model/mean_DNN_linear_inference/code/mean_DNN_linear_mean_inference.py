
# Import necessary libraries and modules
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor

# Set the input and output directories
open_path = './data/in'
save_path = './output'
local_model_file = './data/local_model'
local_DNN_linear_model = './data/DNN_linear_mean_model'

# Get the list of files in the 'open_path' directory
files = [file for file in os.listdir(open_path) if file.endswith('.xlsx')]

# Assume there is only one .xlsx file in the directory and that it's the file we are interested in
species_name = files[0].replace('.xlsx', '')

# Initialize the tokenizer and model based on the user's choice (local or download)
choice = input("Do you want to use a local model or download one? Type 'local' or 'download': ").strip().lower()

if choice == 'local':
    # Use the local model
    base_path = local_model_file
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForMaskedLM.from_pretrained(base_path, output_hidden_states=True)
elif choice == 'download':
    # Download the model
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
else:
    print("Invalid choice. Please type 'local' or 'download'.")

# Load protein sequences into a DataFrame
protein_sequence_df = pd.read_excel(open_path + '/' + species_name + '.xlsx')

# Initialize the feature extractor with options
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=False, compute_eos=False, compute_mean=True,
                                                   compute_segments=False)

# Extract features from protein sequences
protein_sequence_df_represent = feature_extractor.get_last_hidden_features_combine(protein_sequence_df,
                                                                                   sequence_name='Sequence',
                                                                                   batch_size=1)

# Save the extracted features to an Excel file
protein_sequence_df_represent.to_excel(save_path + '/' + species_name + '_feature_rep.xlsx', index=False)
print("Now, you have successfully obtained the 'mean' feature, which has a dimension of 1280.")

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
        return F.log_softmax(self.fc1(x), dim=1)

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
# Configuration
input_dim = 1280  # The dimension of 'feature all'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

# Load the model parameters
load_model = DNNLine(input_dim=input_dim, num_classes=num_classes).to(device)
load_model.load_state_dict(torch.load(os.path.join(local_DNN_linear_model, 'model_parameters.pt')))
load_model.eval()

# Prepare data for inference
protein_sequence_df_represent.set_index('Entry', inplace=True)
protein_sequence_df_represent.drop('Sequence', axis=1, inplace=True)

# Convert labels mapping and perform inference
label_mapping = pd.read_excel(os.path.join(local_DNN_linear_model, 'label2number.xlsx'))
label_dict = dict(zip(label_mapping['EncodedLabel'], label_mapping['OriginalLabel']))

X_inference_data_hat, X_inference_data_probabilities = load_model.model_infer(protein_sequence_df_represent,
                                                                              device=device)

X_inference_data_hat = [label_dict[i] for i in X_inference_data_hat]

# Convert prediction results to DataFrames and save them
X_inference_data_hat_df = pd.DataFrame(X_inference_data_hat, columns=["predict_topic"],
                                       index=protein_sequence_df_represent.index)

X_inference_data_probabilities_max = [max(probs) for probs in X_inference_data_probabilities]
X_inference_data_probabilities_df = pd.DataFrame(X_inference_data_probabilities_max, columns=['predict_probability'],
                                                 index=protein_sequence_df_represent.index)

X_inference_data_hat = pd.concat([X_inference_data_hat_df, X_inference_data_probabilities_df], axis=1)
X_inference_data_hat.to_excel(save_path + '/' + species_name + '_prediction.xlsx')

print('run mean_DNN_linear_inference.py success, feature representation and prediction outcome are deploted in ./mean_DNN_linear_inference/output')