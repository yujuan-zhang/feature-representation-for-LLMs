


import pandas as pd
from packaging import version
import warnings
import numpy as np
from itertools import islice


try:
    import torch
    if version.parse(torch.__version__) < version.parse('1.12.1'):
        warnings.warn("Your torch version is older than 1.12.1 and may not operate correctly.")
except ImportError:
    warnings.warn("Torch not found. Some functions will not be available.")

# Check if tqdm is installed and its version
try:
    import tqdm
    if version.parse(tqdm.__version__) < version.parse('4.63.0'):
        warnings.warn("Your tqdm version is older than 4.63.0 and may not operate correctly.")
    from tqdm import tqdm
except ImportError:
    warnings.warn("tqdm is not installed. Some features may not work as expected.")

# Check if re is installed and its version
try:
    import re
    if version.parse(re.__version__) < version.parse('2.2.1'):
        warnings.warn("Your re version is older than 2.2.1 and may not operate correctly.")
except ImportError:
    warnings.warn("re is not installed. Some features may not work as expected.")

try:
    import sklearn
    if version.parse(sklearn.__version__) < version.parse('1.0.2'):
        warnings.warn("Your sklearn version is older than 1.0.2 and may not operate correctly.")
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    warnings.warn("Sklearn not found. Some functions will not be available.")

##一个蛋白一个蛋白计算cls, eos, 氨基酸平均表征
def get_last_hidden_features_single(X_input, tokenizer, model, sequence_name='sequence',device_choose = 'auto'):
    X_input = X_input.reset_index(drop=True)
    X_outcome = pd.DataFrame()
    
    if device_choose == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choose == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise TypeError("CUDA is not available. Please check your GPU settings.")

    elif device_choose == 'cpu':
        device = torch.device("cpu")
    
    model.to(device)
    with torch.no_grad():
        for index, sequence in tqdm(enumerate(X_input[sequence_name]),
                                    desc='one batch for infer time',
                                    total=len(X_input[sequence_name])):
            inputs = tokenizer(sequence, return_tensors="pt").to(device)
            outputs = model(**inputs)

            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
            eos_position = tokens.index(tokenizer.eos_token) if tokenizer.eos_token in tokens else len(tokens) - 1

            last_hidden_state = outputs.hidden_states[-1]

            last_cls_token = last_hidden_state[:, 0, :]
            last_eos_token = last_hidden_state[:, eos_position, :]
            last_mean_token = last_hidden_state[:, 1:eos_position, :].mean(dim=1)

            features = {}

            cls_features = last_cls_token.squeeze().tolist()
            for i, feature in enumerate(cls_features):
                features[f"ESM2_cls{i}"] = feature

            eos_features = last_eos_token.squeeze().tolist()
            for i, feature in enumerate(eos_features):
                features[f"ESM2_eos{i}"] = feature

            mean_features = last_mean_token.squeeze().tolist()
            for i, feature in enumerate(mean_features):
                features[f"ESM2_mean{i}"] = feature

            result = pd.DataFrame.from_dict(features, orient='index').T
            result.index = [index]
            X_outcome = pd.concat([X_outcome, result], axis=0)
            
            del inputs, outputs
            # 仅在使用CUDA时清理和同步
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
     
    return X_outcome






   
class Esm2LastHiddenFeatureExtractor:
    def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False,num_segments=10,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        
    def get_last_hidden_states(self, outputs):
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def get_last_cls_token(self, last_hidden_state):
        return last_hidden_state[:, 0, :]

    def get_last_eos_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, eos_position, :]

    def get_last_mean_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, last_hidden_state, eos_position):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
        
    
    
    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        
        
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    last_hidden_state = self.get_last_hidden_states(outputs)
                    last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
                    last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if last_cls_token is not None:
                        cls_features = last_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if last_eos_token is not None:
                        eos_features = last_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if last_mean_token is not None:
                        mean_features = last_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    


    
    ##计算磷酸化表征
    def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', 
                                                         phosphorylation_positions='phosphorylation_positions', batch_size=32):
        
        X_input = X_input.reset_index(drop=True)
        
        
            
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = last_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    ##计算磷酸化表征(fast)
    def get_last_hidden_phosphorylation_position_feature_fast(self, X_input, sequence_name='sequence',
                                                              phosphorylation_positions='phosphorylation_positions',
                                                              batch_size=32):

        X_input = X_input.reset_index(drop=True)

        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j + 1]

                    sequence_positions = np.array(sequence_positions).astype(int)
                    # 预先分配空间以存储特征向量
                    features = []

                    # 使用高级索引一次性从last_hidden_state中提取所有特征
                    # 这里假设last_hidden_state的shape为(1, sequence_length, hidden_size)
                    position_features = last_hidden_state[:, sequence_positions, :]

                    # 如果last_hidden_state的第一个维度为1，移除它以简化操作
                    if last_hidden_state.shape[0] == 1:
                        position_features = position_features.squeeze(0)

                    # 将每个特征向量转换为列表并存储
                    for feature in position_features:
                        features.append(feature.tolist())

                    # 将features_batch中的特征向量对应到DataFrame中正确的索引
                    # 将Int64Index转换为常规Python列表
                    for idx, feature in zip(sequence_indices.tolist(), features):
                        X_outcome.loc[idx] = feature

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result


    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = self.get_last_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df




class Esm2LayerHiddenFeatureExtractor:
    def __init__(self, tokenizer, model, layer_indicat, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False, num_segments=10,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.layer_indicat = layer_indicat
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
    
    
    def get_layer_hidden_states(self, outputs):
        layer_hidden_state = outputs.hidden_states[self.layer_indicat]
        return layer_hidden_state

    def get_layer_cls_token(self, layer_hidden_state):
        return layer_hidden_state[:, 0, :]

    def get_layer_eos_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, eos_position, :]

    def get_layer_mean_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, layer_hidden_state, eos_position):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = layer_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(layer_hidden_state[:, start:start+1, :].shape, device=layer_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
    
    
    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_layer_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    layer_hidden_state = self.get_layer_hidden_states(outputs)
                    layer_cls_token = self.get_layer_cls_token(layer_hidden_state[j:j+1]) if self.compute_cls else None
                    layer_eos_token = self.get_layer_eos_token(layer_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    layer_mean_token = self.get_layer_mean_token(layer_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(layer_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if layer_cls_token is not None:
                        cls_features = layer_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if layer_eos_token is not None:
                        eos_features = layer_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if layer_mean_token is not None:
                        mean_features = layer_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    


    
    ##计算磷酸化表征
    def get_layer_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    layer_hidden_state = self.get_layer_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = layer_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    
 
    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the layer hidden state
        layer_hidden_state = self.get_layer_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df







def NetPhos_classic_txt_DataFrame(pattern, data):
    '''
    This function takes a pattern and data as input and returns a DataFrame containing
    the parsed information.

    Parameters
    ----------
    pattern : str
        A regular expression pattern used to match lines in the input data.
    data : str
        The input data containing the information to be parsed.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the parsed information.

    Example
    -------
    To use this function with the example file provided in the package:

    >>> import os
    >>> import protloc_mex_X
    >>> from protloc_mex_X.ESM2_fr import NetPhos_classic_txt_DataFrame
    ...
    >>> example_data = os.path.join(protloc_mex_X.__path__[0], "examples", "test1.txt")
    ...
    >>> with open(example_data, "r") as f:
    ...     data = f.read()
    ... print(data)
    >>> pattern = r".*YES"
    >>> result_df = NetPhos_classic_txt_DataFrame(pattern, data)

    '''

    # Extract lines that match the pattern
    seq_lines = re.findall(pattern, data)

    # Split the extracted lines into lists
    split_lines = [line.split() for line in seq_lines]

    # Remove '#' character
    split_lines = [line[1:] for line in split_lines]

    # Check if each list element has a length of 7, if not, skip that line
    filtered_split_lines = [line for line in split_lines if len(line) == 7]

    # Convert the filtered list to a DataFrame and set column names
    column_names = ['Sequence', 'position', 'x', 'Context', 'Score', 'Kinase', 'Answer']
    df = pd.DataFrame(filtered_split_lines, columns=column_names)

    # Convert the 'Score' column to float type
    df['Score'] = df['Score'].astype(float)

    return df



def phospho_feature_sim_cosine_weighted_average(dim, df_cls, df_phospho):
    # Check if all protein_id in df_phospho are in df_cls
    if not df_phospho.index.isin(df_cls.index).all():
        raise ValueError("Protein_id in df_phospho is not matched with df_cls.")

    # Merge df_cls and df_phospho on index (protein_id)
    df = pd.merge(df_cls, df_phospho, how='inner', left_index=True, right_index=True)

    # Calculate cosine similarity for each row in the merged dataframe
    # 前dim列
    array_cls = df.iloc[:, :dim].values

    # 接下来的dim列
    array_phospho = df.iloc[:, dim:2*dim].values

    # 对于全为0的行，添加一个小的正数以避免除零错误
    epsilon = 1e-10
    array_cls[np.where(~array_cls.any(axis=1))[0]] += epsilon
    array_phospho[np.where(~array_phospho.any(axis=1))[0]] += epsilon

    # 计算余弦相似度
    similarity = np.sum(array_cls * array_phospho, axis=1) / (np.linalg.norm(array_cls, axis=1) * np.linalg.norm(array_phospho, axis=1))

    # 添加到DataFrame
    df['similarity'] = similarity
    #Multiply similarity with phospho_feature for each row
    
    # 使用NumPy进行运算
    array_weighted_features= df['similarity'].values[:, None] * array_phospho

    # 将结果转换为 DataFrame 并合并
    weighted_features_df = pd.DataFrame(array_weighted_features, columns=[f'weighted{i}' for i in range(dim)], index=df.index)
    df = pd.concat([df, weighted_features_df], axis=1)

    # Calculate total weights (sum of abs similarities) for each protein
    total_weights = df['similarity'].abs().groupby(df.index).sum()
    # Calculate sum of weighted features for each protein
    grouped = df.groupby(df.index).agg({**{f'weighted{i}': 'sum' for i in range(dim)}})

    # Calculate weighted average by dividing sum of weighted features by total weights
    # for i in range(dim):
    #     grouped[f'average{i}'] = grouped[f'weighted{i}'] / total_weights
    average_features = {f'pho_average{i}': grouped[f'weighted{i}'] / total_weights for i in range(dim)}
    average_df = pd.DataFrame(average_features)
    grouped = pd.concat([grouped, average_df], axis=1)

    # Merge df_cls and grouped dataframe by protein_id (index)
    df_cls = pd.merge(df_cls, grouped[[f'pho_average{i}' for i in range(dim)]], how='left', left_index=True, right_index=True)

    # For proteins that do not have phospho_feature, set average_feature to zero
    # for i in range(dim):
    #     df_cls[f'average{i}'] = df_cls[f'average{i}'].fillna(0)
    if df_cls.isnull().any().any():
        df_cls = df_cls.fillna(0)
        
    return df_cls


def phospho_feature_sim_cosine_weighted_average_test(dim, df_cls, df_phospho):
    """
    This function computes the overall phospho-representation for a single amino acid sequence 
    using a cosine similarity-based weighting scheme. It merges the given dataframes on their 
    index (protein_id), calculates the cosine similarity for each row, and then calculates the 
    total weights for each protein. It then computes the sum of weighted features for each protein 
    and returns this sum.
    
    Note: This function is designed to work with one amino acid sequence at a time.
    
    Parameters:
    dim (int): The dimensionality of the feature vectors.
    df_cls (DataFrame): The cls feature DataFrame.
    df_phospho (DataFrame): The phospho feature DataFrame.
    
    Returns:
    Series: The sum of the weighted features.
    """
    # Merge df_cls and df_phospho on index (protein_id)
    df = pd.merge(df_cls, df_phospho, how='inner', left_index=True, right_index=True)
    
    # Calculate cosine similarity for each row in the merged dataframe
    similarity = np.sum(df_cls.to_numpy() * df_phospho.to_numpy(), axis=1) / (np.linalg.norm(df_cls.to_numpy(), axis=1) * np.linalg.norm(df_phospho.to_numpy(), axis=1))
    
    # Add the similarity to the DataFrame
    df['similarity'] = similarity
    
    # Calculate total weights (sum of abs similarities) for each protein
    total_weights = df['similarity'].abs().groupby(df.index).sum()
    
    # Calculate sum of weighted features for each protein
    weighted_features = df_phospho.copy()
    weighted_features.columns = [f'weighted{i}' for i in range(dim)]
    
    # Calculate the weight for each row
    df['weight'] = df['similarity'] / total_weights
    
    # Repeat the weights to the same dimension as the features
    weights_matrix = np.repeat(df['weight'].values[:, np.newaxis], dim, axis=1)
    
    # Multiply each row of features by its weight
    average_total = weighted_features.multiply(weights_matrix, axis=0)
    
    # Calculate the sum of the weighted features
    average_total_final = average_total.sum(axis=0)
    
    return average_total_final




###后面都是legacy，可用于检验模型是否正确或无用的代码即不在完整工作流中运行，但不允许删除。考虑在0.020版本中删除

# class Esm2LastHiddenFeatureExtractor_legacy:
#     def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False):
#         self.tokenizer = tokenizer
#         self.model = model
#         self.compute_cls = compute_cls
#         self.compute_eos = compute_eos
#         self.compute_mean = compute_mean
#         self.compute_segments = compute_segments

#     def get_last_hidden_states(self, outputs):
#         last_hidden_state = outputs.hidden_states[-1]
#         return last_hidden_state

#     def get_last_cls_token(self, last_hidden_state):
#         return last_hidden_state[:, 0, :]

#     def get_last_eos_token(self, last_hidden_state, eos_position):
#         return last_hidden_state[:, eos_position, :]

#     def get_last_mean_token(self, last_hidden_state, eos_position):
#         return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

#     def get_segment_mean_tokens(self, last_hidden_state, eos_position, num_segments=10):
#         seq_len = eos_position - 1
#         segment_size, remainder = divmod(seq_len, num_segments)
#         segment_means = []

#         start = 1
#         for i in range(num_segments):
#             end = start + segment_size + (1 if i < remainder else 0)
            
#             if end > start:  # Check if the segment has amino acids
#                 segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
#             else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
#                 segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)
            
#             segment_means.append(segment_mean.squeeze().tolist())
#             start = end

#         return segment_means
    
#     def extract_features(self, cls_token=None, eos_token=None, mean_token=None, segment_means=None):
#         features = {}

#         if cls_token is not None:
#             cls_features = cls_token.squeeze().tolist()
#             features.update({f"ESM2_cls{k}": feature for k, feature in enumerate(cls_features)})

#         if eos_token is not None:
#             eos_features = eos_token.squeeze().tolist()
#             features.update({f"ESM2_eos{k}": feature for k, feature in enumerate(eos_features)})

#         if mean_token is not None:
#             mean_features = mean_token.squeeze().tolist()
#             features.update({f"ESM2_mean{k}": feature for k, feature in enumerate(mean_features)})

#         if segment_means is not None:
#             features.update({
#                 f"ESM2_segment{seg}_mean{idx}": feature
#                 for seg, segment_mean in enumerate(segment_means)
#                 for idx, feature in enumerate(segment_mean)
#             })

#         return features
    
#     ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
#     def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
#         X_outcome = pd.DataFrame()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#         sequence = X_input[sequence_name].tolist()
#         with torch.no_grad():
#             for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
#                 batch_sequences = sequence[i:i+batch_size]
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(device)
#                 outputs = self.model(**inputs)

#                 for j in range(len(batch_sequences)):
#                     idx = i + j
#                     tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
#                     eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(tokens) - 1
#                     last_hidden_state = self.get_last_hidden_states(outputs)
#                     last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
#                     last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
#                     last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
#                     segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

#                     features = self.extract_features(cls_token=last_cls_token, eos_token=last_eos_token, mean_token=last_mean_token, segment_means=segment_means)
#                     result = pd.DataFrame.from_dict(features, orient='index').T
#                     result.index = [idx]
#                     X_outcome = pd.concat([X_outcome, result], axis=0)

#                 del inputs, outputs
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result
    
#     ##计算磷酸化表征
#     def get_last_hidden_phosphorylation_position_feature_legacy(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
#         X_outcome = pd.DataFrame()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#         sequence = X_input[sequence_name].tolist()
#         positions = X_input[phosphorylation_positions].tolist()
#         with torch.no_grad():
#             for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
#                 batch_sequences = sequence[i:i+batch_size]
#                 batch_positions = positions[i:i+batch_size]
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(device)
#                 outputs = self.model(**inputs)

#                 for j in range(len(batch_sequences)):
#                     idx = i + j
#                     last_hidden_state = self.get_last_hidden_states(outputs)

#                     # Extract features for the phosphorylation position
#                     position = int(batch_positions[j])  # Make sure position is an integer
#                     position_feature = last_hidden_state[j:j+1][:, position,:]  # Removed +1 since the sequence starts from 1, and consider remove cls
#                     features = {}
#                     features.update({
#                         f"ESM2_phospho_pos{k}": feature
#                         for k, feature in enumerate(position_feature.squeeze().tolist())
#                     })
#                     result = pd.DataFrame.from_dict(features, orient='index').T
#                     result.index = [idx]
#                     X_outcome = pd.concat([X_outcome, result], axis=0)
#                 del inputs, outputs
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result

#     #from itertools import islice
    
       
#     def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
#         X_outcome = pd.DataFrame()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)

#         # Group X_input by sequence
#         grouped_X_input = X_input.groupby(sequence_name)
#         sequence_to_indices = grouped_X_input.groups

#         with torch.no_grad():
#             for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
#                 batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
#                 batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

#                 # Get the unique sequences in the batch
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(device)
#                 outputs = self.model(**inputs)

#                 for j, sequence in enumerate(batch_sequences):
#                     sequence_indices = batch_grouped_sequences[sequence].index
#                     sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
#                     last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

#                     for idx, position in zip(sequence_indices, sequence_positions):
#                         position = int(position)  # Make sure position is an integer
#                         position_feature = last_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token

#                         features = {
#                             f"ESM2_phospho_pos{k}": feature
#                             for k, feature in enumerate(position_feature.squeeze().tolist())  # Added .cpu() before converting tensor to numpy
#                         }
#                         result = pd.DataFrame.from_dict(features, orient='index').T
#                         result.index = [idx]
#                         X_outcome = pd.concat([X_outcome, result], axis=0)

#                 del inputs, outputs
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result

#     # ###local计算每个氨基酸的表征
#     # def get_sequence_features(self, last_hidden_state, tokens_batch):
#     #     sequence_features = []

#     #     for i, tokens in enumerate(tokens_batch):
#     #         features = []

#     #         for j, token in enumerate(tokens):
#     #             # Ignore special tokens
#     #             if token in [self.tokenizer.cls_token, self.tokenizer.eos_token, self.tokenizer.pad_token]:
#     #                 continue

#     #             # Get the feature vector for the current token
#     #             token_features = last_hidden_state[i, j, :].tolist()

#     #             # Prepend the token to the feature names and add to the sequence features
#     #             features.extend([
#     #                 {f"ESM2_{token}_{k+1}": feature}
#     #                 for k, feature in enumerate(token_features)
#     #             ])

#     #         sequence_features.append(features)

#     #     return sequence_features

#     # def get_individual_sequence_features(self, X_input, sequence_name='sequence', batch_size=32):
#     #     all_sequence_features = []
#     #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #     self.model.to(device)
#     #     sequence = X_input[sequence_name].tolist()

#     #     with torch.no_grad():
#     #         for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
#     #             batch_sequences = sequence[i:i+batch_size]
#     #             inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(device)
#     #             outputs = self.model(**inputs)
#     #             tokens_batch = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in inputs['input_ids']]

#     #             sequence_features = self.get_sequence_features(outputs.hidden_states[-1], tokens_batch)
#     #             all_sequence_features.extend(sequence_features)

#     #             del inputs, outputs, sequence_features
#     #             torch.cuda.empty_cache()
#     #             torch.cuda.synchronize()

#     #     all_features_df = pd.concat([pd.DataFrame(features) for features in all_sequence_features], ignore_index=True)
#     #     return all_features_df
    
#     def get_amino_acid_representation(self, sequence, amino_acid, position):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
        
#         # Check if the amino acid at the given position matches the input
#         if sequence[position - 1] != amino_acid:
#             raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

#         # Convert the sequence to input tensors
#         inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(device)
        
#         # Get the model outputs
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # Get the last hidden state
#         last_hidden_state = self.get_last_hidden_states(outputs)
        
#         # Get the tokens from the input ids
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
#         # Get the position of the amino acid token in the tokens list
#         # We add 1 to the position to account for the CLS token at the start
#         token_position =  position if amino_acid == tokens[position] else -1
        
#         if token_position == -1:
#             raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
#         # Prepare the DataFrame
#         feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
#         amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

#         return amino_acid_features_df

# def phospho_feature_sim_cosine_mean_legacy(dim, df_cls, df_phospho):
#     # Check if all protein_id in df_phospho are in df_cls
#     if not df_phospho.index.isin(df_cls.index).all():
#         raise ValueError("Protein_id in df_phospho is not matched with df_cls.")

#     # Merge df_cls and df_phospho on index (protein_id)
#     df = pd.merge(df_cls, df_phospho, how='outer', left_index=True, right_index=True)

#     # Drop rows with NA values
#     df = df.dropna()

#     # Calculate cosine similarity for each row in the merged dataframe
#     df['similarity'] = df.apply(lambda row: cosine_similarity([row[:dim].values], [row[dim:2*dim].values])[0][0], axis=1)

#     # Multiply similarity with phospho_feature for each row
#     df[[f'weighted{i}' for i in range(dim)]] = df.apply(lambda row: row['similarity'] * row[dim:2*dim], axis=1)

#     # Group the dataframe by protein_id (index), and calculate the sum of weighted_feature
#     grouped = df.groupby(df.index).agg({**{f'weighted{i}': 'sum' for i in range(dim)}})

#     # Count the number of rows for each protein_id
#     grouped_counts = df.groupby(df.index).size()

#     # Divide sum of weighted_feature by count of rows for each protein_id
#     for i in range(dim):
#         grouped[f'average{i}'] = grouped.apply(lambda row: row[f'weighted{i}'] / grouped_counts[row.name], axis=1)

#     # Merge df_cls and grouped dataframe by protein_id (index)
#     df_cls = pd.merge(df_cls, grouped[[f'average{i}' for i in range(dim)]], how='left', left_index=True, right_index=True)

#     # For proteins that do not have phospho_feature, set average_feature to zero
#     for i in range(dim):
#         df_cls[f'average{i}'] = df_cls[f'average{i}'].fillna(0)
    
#     return df_cls


# def phospho_feature_sim_cosine_weighted_average_legacy(dim, df_cls, df_phospho):
#     # Check if all protein_id in df_phospho are in df_cls
#     if not df_phospho.index.isin(df_cls.index).all():
#         raise ValueError("Protein_id in df_phospho is not matched with df_cls.")

#     # Merge df_cls and df_phospho on index (protein_id)
#     df = pd.merge(df_cls, df_phospho, how='inner', left_index=True, right_index=True)

#     # Calculate cosine similarity for each row in the merged dataframe
#     df['similarity'] = df.apply(lambda row: cosine_similarity([row[:dim].values], [row[dim:2*dim].values])[0][0], axis=1)

#     #Multiply similarity with phospho_feature for each row
#     # df[[f'weighted{i}' for i in range(dim)]] = df.apply(lambda row: row['similarity'] * row[dim:2*dim], axis=1)
#     weighted_features = df.apply(lambda row: row['similarity'] * row[dim:2*dim], axis=1)
#     weighted_df = pd.DataFrame(weighted_features.values, columns=[f'weighted{i}' for i in range(dim)], index=df.index)
#     df = pd.concat([df, weighted_df], axis=1)

#     # Calculate total weights (sum of abs similarities) for each protein
#     total_weights = df['similarity'].abs().groupby(df.index).sum()
#     # Calculate sum of weighted features for each protein
#     grouped = df.groupby(df.index).agg({**{f'weighted{i}': 'sum' for i in range(dim)}})

#     # Calculate weighted average by dividing sum of weighted features by total weights
#     # for i in range(dim):
#     #     grouped[f'average{i}'] = grouped[f'weighted{i}'] / total_weights
#     average_features = {f'average{i}': grouped[f'weighted{i}'] / total_weights for i in range(dim)}
#     average_df = pd.DataFrame(average_features)
#     grouped = pd.concat([grouped, average_df], axis=1)

#     # Merge df_cls and grouped dataframe by protein_id (index)
#     df_cls = pd.merge(df_cls, grouped[[f'average{i}' for i in range(dim)]], how='left', left_index=True, right_index=True)

#     # For proteins that do not have phospho_feature, set average_feature to zero
#     # for i in range(dim):
#     #     df_cls[f'average{i}'] = df_cls[f'average{i}'].fillna(0)
#     if df_cls.isnull().any().any():
#         df_cls = df_cls.fillna(0)
        
#     return df_cls




   
class Esm2LastHiddenFeatureExtractor_legacy:
    def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        
        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        
    def get_last_hidden_states(self, outputs):
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def get_last_cls_token(self, last_hidden_state):
        return last_hidden_state[:, 0, :]

    def get_last_eos_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, eos_position, :]

    def get_last_mean_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, last_hidden_state, eos_position, num_segments=10):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, num_segments)
        segment_means = []

        start = 1
        for i in range(num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
        
    
    
    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        
        
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    last_hidden_state = self.get_last_hidden_states(outputs)
                    last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
                    last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if last_cls_token is not None:
                        cls_features = last_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if last_eos_token is not None:
                        eos_features = last_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if last_mean_token is not None:
                        mean_features = last_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    


    
    ##计算磷酸化表征
    def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', 
                                                         phosphorylation_positions='phosphorylation_positions', batch_size=32):
        
        X_input = X_input.reset_index(drop=True)
        
        
            
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = last_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    
 
    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = self.get_last_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df




class Esm2LayerHiddenFeatureExtractor_legacy:
    def __init__(self, tokenizer, model, layer_indicat, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False, device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.layer_indicat = layer_indicat
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
    
        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
    
    
    def get_layer_hidden_states(self, outputs):
        layer_hidden_state = outputs.hidden_states[self.layer_indicat]
        return layer_hidden_state

    def get_layer_cls_token(self, layer_hidden_state):
        return layer_hidden_state[:, 0, :]

    def get_layer_eos_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, eos_position, :]

    def get_layer_mean_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, layer_hidden_state, eos_position, num_segments=10):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, num_segments)
        segment_means = []

        start = 1
        for i in range(num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = layer_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(layer_hidden_state[:, start:start+1, :].shape, device=layer_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
    
    
    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_layer_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    layer_hidden_state = self.get_layer_hidden_states(outputs)
                    layer_cls_token = self.get_layer_cls_token(layer_hidden_state[j:j+1]) if self.compute_cls else None
                    layer_eos_token = self.get_layer_eos_token(layer_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    layer_mean_token = self.get_layer_mean_token(layer_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(layer_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if layer_cls_token is not None:
                        cls_features = layer_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if layer_eos_token is not None:
                        eos_features = layer_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if layer_mean_token is not None:
                        mean_features = layer_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    


    
    ##计算磷酸化表征
    def get_layer_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    layer_hidden_state = self.get_layer_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = layer_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # 仅在使用CUDA时清理和同步
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    
 
    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the layer hidden state
        layer_hidden_state = self.get_layer_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df





















