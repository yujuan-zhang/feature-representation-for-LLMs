## Practical Tutorial

This section provides a convenient example for efficiently accomplishing the protein sequence feature extraction task in this project. Specifically, it leverages the ESM2 model to perform deep analysis and feature extraction on protein sequences, followed by utilizing the extracted features for subcellular localization prediction of proteins. For detailed steps and methods, please refer to the following content:

**Note**: To work with this section, download the folder to your local directory or manually create a corresponding path and place the files and code from this section within that path.

Procedure:

1. To ensure a smooth process, first verify that your current working environment is set up and configured according to the **Work Environment Setup** guide under `Setting Up the Work Environment` in the GitHub project **Feature Representation for LLMs**. This is a prerequisite for all operations to proceed as expected.

3. Navigate to the `mean_DNN_linear_inference` directory:
   ```bash
   cd path_to_file\mean_DNN_linear_inference
   
   # In my personal PC case:
   cd D:\work\mean_DNN_linear_inference
   ```

4. The input file format should be Excel, where the column name for protein sequences must be set to 'Sequence', and the column name for protein IDs should be set to 'Entry'. Place this file in the `data/in` directory within the current folder (note: currently, only a single Excel file is supported).

   If the file has a `.faa` or `.fasta` extension (i.e., it's a FASTA formatted protein sequence), after placing it in the `data/in` directory, run the following command in the `step 2` environment path:
   
   ```bash
   python ./code/data_prepared.py
   # This script converts the FASTA-formatted protein sequence data file into a format suitable for further processing.
   ```

5. Run the following codes to perform model inference and obtain predicted subcellular localization labels along with computed feature values:

   ```bash
   python ./code/mean_DNN_linear_mean_inference.py  # This script acquires mean-pooled representation features; run this command to make predictions based on these features.
   python ./code/mean_DNN_linear_segment0_mean_inference.py  # This script gets features pooled from segment0; if you wish to make predictions based on these features, execute this command.
   ```

   When running this script, the system will prompt whether to use an ESM2 model downloaded locally. If choosing 'local', you need to pre-download the ESM2 model from Hugging Face and place it in the `mean_DNN_linear_inference\data\local_model` path (this is where details about the required files are provided). If selecting 'download', the script will automatically connect to the ESM2 model hosted on Hugging Face (not recommended if network conditions are poor).