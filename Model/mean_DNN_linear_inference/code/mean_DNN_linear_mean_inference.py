import os
import torch
import pandas as pd
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
open_path = str(base_dir / 'data/in')
save_path = str(base_dir / 'output')
local_model_file = str(base_dir / 'data/local_model')
local_DNN_linear_model = str(base_dir / 'data/DNN_linear_mean_model')

import argparse
parser = argparse.ArgumentParser(description='Predict localization from one protein Excel file.')
source = parser.add_mutually_exclusive_group()
source.add_argument('--input', type=Path, help='Excel file with Entry and Sequence columns')
source.add_argument('--features', type=Path, help='Excel file with Entry, Sequence and ESM2_mean0..1279 columns')
parser.add_argument('--output-dir', type=Path, help='Separate output directory; defaults to the bundled output directory')
args = parser.parse_args()
files = sorted(Path(open_path).glob('*.xlsx'))
if args.features or args.input:
    input_file = (args.features or args.input).expanduser().resolve()
elif len(files) == 1:
    input_file = files[0]
else:
    parser.error('Expected one Excel file in the input directory; specify --input explicitly.')
if not input_file.is_file():
    parser.error('Input file does not exist: ' + str(input_file))
species_name = input_file.stem
if args.output_dir:
    save_path = str(args.output_dir.expanduser().resolve())
Path(save_path).mkdir(parents=True, exist_ok=True)

if args.features:
    import numpy as np
    protein_sequence_df_represent = pd.read_excel(input_file)
    required = ['Entry', 'Sequence'] + [f'ESM2_mean{i}' for i in range(1280)]
    if list(protein_sequence_df_represent.columns) != required:
        parser.error('Expected Entry, Sequence, then ESM2_mean0 through ESM2_mean1279 in order.')
    if protein_sequence_df_represent.empty or protein_sequence_df_represent['Entry'].isna().any():
        parser.error('Feature input must contain at least one row and non-empty Entry IDs.')
    try:
        values = protein_sequence_df_represent.iloc[:, 2:].to_numpy(dtype=float)
    except (ValueError, TypeError):
        parser.error('Mean features must be numeric.')
    if not np.isfinite(values).all():
        parser.error('Mean features must be finite.')
    species_name = species_name.removesuffix('_feature_rep')
else:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor
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
        raise SystemExit("Invalid choice. Please type 'local' or 'download'.")
    
    # Load protein sequences into a DataFrame
    protein_sequence_df = pd.read_excel(input_file)
    
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
load_model.load_state_dict(torch.load(os.path.join(local_DNN_linear_model, 'model_parameters.pt'), map_location=device))
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

print('Prediction saved to: ' + str(Path(save_path) / (species_name + '_prediction.xlsx')))
