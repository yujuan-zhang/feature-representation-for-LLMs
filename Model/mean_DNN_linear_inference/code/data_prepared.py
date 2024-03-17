# # Import necessary libraries
# from Bio import SeqIO
# import os
# import pandas as pd
# import re
# 
# # Define a function to check if strings contain a specified pattern
# def str_contains(str_input, input_names):
#     pattern = re.compile(str_input)
#     matching_names = [i for i in input_names if re.search(pattern, i)]
#     return matching_names
# 
# # Define a function to read a FASTA file and generate a dictionary
# def read_fasta(fasta_path):
#     fasta_dict = {}
#     with open(fasta_path) as file:
#         for line in file:
#             if line.startswith('>'):
#                 name = line[1:].rstrip()
#                 fasta_dict[name] = ''
#                 continue
#             fasta_dict[name] += line.rstrip().upper()
#     return fasta_dict
# 
# # Set the input and output paths
# input_path = './data/in'
# output_path = './data/in'
# 
# file_extension_faa = '.faa'
# file_extension_fasta = '.fasta'
# 
# # Check and create the output directory if it doesn't exist
# if not os.path.isdir(output_path):
#     os.makedirs(output_path)
# 
# # Get all filenames ending with .fasta and .faa in the directory
# all_file_names = os.listdir(input_path)
# fasta_files = str_contains(".fasta", all_file_names)
# faa_files = str_contains('.faa', all_file_names)
# 
# # Remove file extensions from the names
# fasta_names = [name.replace('.fasta', '') for name in fasta_files]
# faa_names = [name.replace('.faa', '') for name in faa_files]
# 
# # Process .fasta and .faa files separately
# fasta_results = {}
# for file_name in fasta_names:
#     data = {'Entry': [], 'Sequence': []}
# 
#     with open(os.path.join(input_path, f'{file_name}{file_extension_fasta}'), "r") as input_file:
#         sequences = list(SeqIO.parse(input_file, "fasta"))
# 
#         prot_ids = [str(seq.id) for seq in sequences]
#         prot_seqs = [str(seq.seq) for seq in sequences]
# 
#         data['Entry'].extend(prot_ids)
#         data['Sequence'].extend(prot_seqs)
# 
#     fasta_results[file_name] = pd.DataFrame(data)
# 
# faa_results = {}
# for file_name in faa_names:
#     data = {'Entry': [], 'Sequence': []}
# 
#     with open(os.path.join(input_path, f'{file_name}{file_extension_faa}', "r")) as input_file:  # Typo: missing closing parenthesis
#         sequences = list(SeqIO.parse(input_file, "fasta"))
# 
#         prot_ids = [str(seq.id) for seq in sequences]
#         prot_seqs = [str(seq.seq) for seq in sequences]
# 
#         data['Entry'].extend(prot_ids)
#         data['Sequence'].extend(prot_seqs)
# 
#     faa_results[file_name] = pd.DataFrame(data)
# 
# # Save the results of .fasta and .faa processing into separate Excel files
# for key, value in fasta_results.items():
#     output_file_name = os.path.join(output_path, f'{key}.xlsx')
#     value.to_excel(output_file_name, index=False)
# 
# for key, value in faa_results.items():
#     output_file_name = os.path.join(output_path, f'{key}.xlsx')
#     value.to_excel(output_file_name, index=False)
# 
# print('run data_prepared.py success. excel outcome are deployed in ./mean_DNN_linear_inference/output')


# Import necessary libraries
from Bio import SeqIO
import os
import pandas as pd


# Define a function to read a FASTA file and generate a dictionary or DataFrame
def read_fasta(fasta_path, extension):
    data = {'Entry': [], 'Sequence': []}

    with open(os.path.join(input_path, f'{os.path.splitext(os.path.basename(fasta_path))[0]}{extension}'),
              "r") as input_file:
        sequences = list(SeqIO.parse(input_file, "fasta"))

        prot_ids = [str(seq.id) for seq in sequences]
        prot_seqs = [str(seq.seq).upper() for seq in sequences]

        data['Entry'].extend(prot_ids)
        data['Sequence'].extend(prot_seqs)

    return pd.DataFrame(data)


# Set the input and output paths
input_path = './data/in'
output_path = './data/out'  # Modify this path to your desired output directory

if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Get all filenames ending with .fasta and .faa in the directory
all_file_names = [filename for filename in os.listdir(input_path) if filename.endswith(('.fasta', '.faa'))]

results = {}
for file_name in all_file_names:
    file_extension = os.path.splitext(file_name)[-1].lower()
    df = read_fasta(os.path.join(input_path, file_name), file_extension)
    results[file_name.replace(file_extension, '')] = df

# Save the results of .fasta and .faa processing into separate Excel files
for key, value in results.items():
    output_file_name = os.path.join(output_path, f'{key}.xlsx')
    value.to_excel(output_file_name, index=False)

print('run data_prepared.py success. excel outcome are deployed in ./mean_DNN_linear_inference/data/in')