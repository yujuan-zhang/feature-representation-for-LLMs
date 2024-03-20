## Conducting Protein Homology Analysis using NCBI Blast Software and Excluding Proteins with High Homology

### Downloading and Installing NCBI Blast:

1. Download Software

   Download the NCBI Blast software (version: `ncbi-blast-2.14.1+-win64.exe`) using this link, [Index of ast/executablesast+/LATEST](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).

2. Install Software

   Install the downloaded software on a non-system default, for example, `E:\blast`.
   Upon completion of the installation, you will see two sub-folders: `bin` (program directory) and `doc` (documentation directory).

3. Set Environment Variables

   Right-click on "My Computer" and select "Properties," then click on "Advanced System Settings" - "Environment Variables."
   Under "User Variables," find `Path`, which should have already been auto-updated with the value `E:\Blast\bin`.
   Click on "New," enter the variable name `BLASTDB`, and the variable value `E:\Blast\db` (this is the database path).

4. Verify Installation

   Click on the Windows "Start" menu, type `cmd` to open the Command Prompt (in XP system, type `cmd` in the Run box).
   In the command line, navigate to the Blast installation directory, and type the command `blastn -version` to check the version. If the version information appears, the installation was successful.

5. Installation on Other Systems or Versions

   For installations on other systems or versions, you can visit the official installation guide: [Installation - BLAST Command Line Applications User Manual - NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK569861/#intro_Installation.Building_sources_in_W).

### Conducting Homology Analysis of Protein Data:

1. Ensure that the protein sequence data for the swiss data (training set, test set) and independent test set are all stored in `FASTA` format files. The code below provides a method for converting `xlsx` and `csv` format protein sequence files to `FASTA` format files.

   ```python
   import os
   import pandas as pd
   
   # Input and output directories
   open_path = "<your_data_path>"
   save_path = "<your_save_path>"
   filenames = os.listdir(open_path)
   
   for filename in filenames:
       file_path = os.path.join(open_path, filename)
       # Define output FASTA filename
       if filename.endswith('.xlsx'):
           # Read Excel file
           df = pd.read_excel(file_path)
           fasta_filename = filename[:-5] + '.fasta'  # Remove '.xlsx' from filename and append '.fasta'
           
           # Define output FASTA file path
           fasta_file_path = os.path.join(save_path, fasta_filename)
   
           # Convert to FASTA format and save
           with open(fasta_file_path, 'w') as fasta_file:
               for index, row in df.iterrows():
                   entry = row["Entry"]
                   sequence = row["Sequence"]
                   fasta_file.write(f">{entry}\n{sequence}\n")
       elif filename.endswith('.csv'):
           # Read CSV file
           df = pd.read_csv(file_path)
           fasta_filename = filename[:-4] + '.fasta'  # Remove '.csv' from filename and append '.fasta'
           
           # Define output FASTA file path
           fasta_file_path = os.path.join(save_path, fasta_filename)
   
           # Convert to FASTA format and save
           with open(fasta_file_path, 'w') as fasta_file:
               for index, row in df.iterrows():
                   entry = row["Entry"]
                   sequence = row["Sequence"]
                   fasta_file.write(f">{entry}\n{sequence}\n")
       
       else:
           print(f"If the file {filename} is not in xlsx or csv format, please convert it based on the code.")
           continue
   ```

2. Based on comparison protein sequence data, create a local homologous comparison datasets, specifically，in our experiment，we set swiss data （combine swiss train and test）as local homologous comparison datasets. please execute the following tasks in the terminal.

   ```
   makeblastdb -in "<your_train_data_path/train_data.fasta>" -dbtype prot -out "<Blast_database_path/Train_protein_seq_database>"
   # Note: The paths provided above should not contain any spaces!
   ```

3. Perform homology analysis of Independent test set protein sequences against the local homologous comparison datasets (Execute the following tasks in the Terminal).

   ```
   blastp -query "<your_test_data_path/test_data.fasta>" -db "<Blast_database_path/Train_protein_seq_database>" -outfmt 5 -out "<your_save_path/test_data_blast_results.xml>"
   # Note: The paths provided above should not contain any spaces!
   ```


### Excluding Proteins with High Homology

1. Retrieve the homology analysis information required from the XML file, which includes: E-value, Bitscore, Identical Sites, and Positives.

   ```python
   import pandas as pd
   from Bio import SearchIO
   
   # Define the columns for the DataFrame
   columns = ['Query_ID', 'Hit_ID', 'HSP_E-value', 'HSP_Bitscore', 'HSP_Identical_sites', 'HSP_Positives']
   df = pd.DataFrame(columns=columns)  # Initialize an empty DataFrame
   
   # Parse the BLAST XML file
   blast_file_path = "<your_save_path/test_data_blast_results.xml>"  # Specify the path to the BLAST XML file
   blast_qresults = list(SearchIO.parse(blast_file_path, "blast-xml"))
   
   # Iterate through all Query Results
   for qresult in blast_qresults:
       query_id = qresult.id
       
       # Iterate through all Hits within a Query Result
       for hit in qresult:
           hit_id = hit.id
           
           # Iterate through all High-scoring Segment Pairs (HSPs) within a Hit
           for hsp in hit.hsps:
               hsp_data = {
                   'Query_ID': query_id, 
                   'Hit_ID': hit_id, 
                   'HSP_E-value': hsp.evalue, 
                   'HSP_Bitscore': hsp.bitscore, 
                   'HSP_Identical_sites': hsp.ident_num, 
                   'HSP_Positives': hsp.pos_num
               }
               df = df.append(hsp_data, ignore_index=True)  # Append the HSP data to the DataFrame
   
   # Define the directory and filename for saving
   save_path = '<your_save_path/test_data_blast_results.xlsx>'  # Specify the path to save the Excel file
   df.to_excel(save_path, index=False)
   ```

2. Remove protein sequences from the independent test set that exhibit some degree homology with the local homologous comparison datasets, to obtain a set of proteins with relatively lower homology. In this experiment, a Bitscore > 50 is utilized as the threshold for filtering high-homology sequences.

   ```python
   import pandas as pd
   
   # Data obtained after homology analysis
   blast_results_path = '<your_save_path/test_data_blast_results.xlsx>'
   
   # Original test set data
   all_data_path = '<your_test_data_path/test_data.csv>'
   
   df = pd.read_excel(blast_results_path)
   df_test = pd.read_csv(all_data_path, usecols=["Entry"])  # Choose an appropriate read method based on your data
   df_test.rename(columns={'Entry': 'Query_ID'}, inplace=True)
   
   # Filtering data
   filtered_df = df[df['HSP_Bitscore'] > 50]
   remaining_unique_set = set(df_test['Query_ID']) - set(filtered_df['Query_ID'].unique())
   filtered_low_homology_test_df = df_test[df_test['Query_ID'].isin(remaining_unique_set)]
   
   # Saving to Excel
   output_excel_path = '<your_save_path/filtered_low_homology_test_data.xlsx>'
   filtered_low_homology_test_df.to_excel(output_excel_path, index=False)
   ```
