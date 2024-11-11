import os
import shutil
import gzip
import zipfile
import tarfile
import rarfile  
import numpy as np
import pickle
import bz2
from scipy import sparse


def extract_files(path, keep_original=True):
    """
    Extract compressed files in the specified directory and optionally delete the original files.
    
    Args:
        path (str): The directory path containing files to check and extract.
        keep_original (bool): If False, the function will delete the compressed files after extraction.
                              Defaults to True, meaning the original compressed files will be kept.
    """
    # Check if the provided path exists
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return
    
    # List all files in the given directory
    files = os.listdir(path)
    compressed_files_found = False  # Flag to track if any compressed files are found
    
    for file in files:
        file_path = os.path.join(path, file)  # Construct full file path
        
        # Check and extract .gz files
        if file.endswith('.gz'):
            compressed_files_found = True  # Set flag to True since we found a compressed file
            # Open the .gz file in read mode (binary format)
            with gzip.open(file_path, 'rb') as f_in:
                output_path = os.path.splitext(file_path)[0]  # Create output file name by removing the .gz extension
                # Write the decompressed content to a new file
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted: {file}")  # Print message to confirm extraction
            # Optionally delete the original .gz file
            if not keep_original:
                os.remove(file_path)  # Remove the original .gz file
                print(f"Deleted: {file}")

        # Check and extract .zip files
        elif file.endswith('.zip'):
            compressed_files_found = True
            # Open the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)  # Extract all contents to the same directory
            print(f"Extracted: {file}")
            # Optionally delete the original .zip file
            if not keep_original:
                os.remove(file_path)  # Remove the original .zip file
                print(f"Deleted: {file}")

        # Check and extract .tar, .tar.gz, .tar.bz2, .tar.xz files
        elif file.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz')):
            compressed_files_found = True
            # Open the tar file (supports multiple compression formats)
            with tarfile.open(file_path) as tar_ref:
                tar_ref.extractall(path)  # Extract all contents to the directory
            print(f"Extracted: {file}")
            # Optionally delete the original tar file
            if not keep_original:
                os.remove(file_path)  # Remove the original tar file
                print(f"Deleted: {file}")
        
        # Check and extract .rar files (rarfile package must be installed separately)
        elif file.endswith('.rar'):
            compressed_files_found = True
            # Open the rar file
            with rarfile.RarFile(file_path) as rar_ref:
                rar_ref.extractall(path)  # Extract all contents to the directory
            print(f"Extracted: {file}")
            # Optionally delete the original .rar file
            if not keep_original:
                os.remove(file_path)  # Remove the original .rar file
                print(f"Deleted: {file}")
    
    # After iterating over all files, check if any compressed files were found
    if compressed_files_found:
        print("Extraction complete.")  # Print message when extraction is completed
    else:
        print("No compressed files found in the given path.")  # Print message if no compressed files are found



def process_single_cell_data(file_path, save_as='npy', compress=False):
    """
    Function to process single-cell gene expression data from a txt file,
    store gene names and expression levels, and save them into an npy or pickle file.

    Parameters:
    - file_path (str): Path to the input txt file.
    - save_as (str): File format for saving ('npy' or 'pickle'). Defaults to 'npy'.
    - compress (bool): Whether to compress the saved file. Defaults to False.

    Returns:
    - None. Saves the processed data into the appropriate file format.
    """
    
    gene_names = []  # List to store gene names
    expression_data = []  # List to store expression levels

    # Step 1: Open and read the input file
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines at once
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Step 2: Process each line after skipping the barcode section
    reading_expression_data = False  # Flag to indicate when we are reading expression data
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')  # Split the line by tabs

        # Check if this line contains only barcode sequence
        if all(c in 'CGAT' for c in parts[0]):
            print(f"Barcode sequence detected, skipping: {parts[0]}")
            continue  # Skip barcode lines

        # Now we check if this is the part with gene expression data (second type of sequence)
        if not reading_expression_data and parts[0].isalpha():
            reading_expression_data = True  # We are now reading expression data

        if reading_expression_data:
            try:
                gene_name = parts[0]  # First part is the gene name
                expression_values = list(map(float, parts[1:]))  # Rest are expression values
                
                # Ensure expression values are valid
                if len(expression_values) == 0:
                    print(f"No expression values found for gene '{gene_name}', skipping.")
                    continue

                gene_names.append(gene_name)  # Store gene name
                expression_data.append(expression_values)  # Store expression data
            except ValueError as ve:
                # Skip non-numeric gene names (e.g., barcode sequences)
                if 'could not convert' in str(ve):
                    continue
                print(f"ValueError: {ve} - Non-numeric data found for gene '{parts[0]}', skipping.")
                continue

    # Step 3: Convert expression data to a matrix (n x k)
    if expression_data:
        expression_matrix = np.array(expression_data)  # Convert to numpy array
        print(f"Expression matrix shape: {expression_matrix.shape}")
        
        # Convert to a sparse matrix format
        sparse_expression_matrix = sparse.csr_matrix(expression_matrix)
    else:
        print("No valid expression data found.")
        return

    # Print the gene names processed
    if len(gene_names) > 100:
        print("Gene names processed (first 100):")
        print(gene_names[:100])
    else:
        print("Gene names processed (all):")
        print(gene_names)

    # Step 4: Prepare to save data to the processed directory
    file_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(file_dir)  # Get the parent directory
    folder_name = os.path.basename(file_dir)  # Get the name of the folder containing the txt file
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Get the name of the txt file without extension
    processed_folder = os.path.join(parent_dir, f"{folder_name}_processed")  # Create the processed folder path

    # Step 5: Check if the processed folder exists, if not, create it
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        print(f"Processed folder created at: {processed_folder}")

    # Step 6: Save the gene names and expression matrix into either an npy or pickle file
    if save_as == 'npy':
        sparse.save_npz(os.path.join(processed_folder, f"{file_name}.npz"), sparse_expression_matrix, compressed=compress)
        print(f"Sparse matrix saved as .npz at {processed_folder}/{file_name}.npz")
    elif save_as == 'pickle':
        if compress:
            with bz2.BZ2File(os.path.join(processed_folder, f"{file_name}.pkl.bz2"), 'wb') as f:
                pickle.dump({'gene_names': gene_names, 'expression_matrix': sparse_expression_matrix}, f)
            print(f"Compressed data saved as .pkl.bz2 at {processed_folder}/{file_name}.pkl.bz2")
        else:
            with open(os.path.join(processed_folder, f"{file_name}.pkl"), 'wb') as f:
                pickle.dump({'gene_names': gene_names, 'expression_matrix': sparse_expression_matrix}, f)
            print(f"Data saved as .pkl at {processed_folder}/{file_name}.pkl")
    else:
        raise ValueError("Unsupported save format. Choose 'npy' or 'pickle'.")

    print(f"Processed {len(gene_names)} genes and saved the results successfully.")


def process_all_sc_data(directory, save_as='npy', compress=False):
    """
    Process all .txt files in the specified directory using the process_single_cell_data function.

    Parameters:
    - directory (str): The directory containing .txt files to process.
    - save_as (str): File format for saving ('npy' or 'pickle'). Defaults to 'npy'.
    - compress (bool): Whether to compress the saved file. Defaults to False.

    Returns:
    - None. Processes each .txt file in the directory.
    """
    # Check if the provided directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory)
    txt_files = [f for f in files if f.endswith('.txt')]  # Filter for .txt files

    if not txt_files:
        print("No .txt files found in the specified directory.")
        return

    # Process each .txt file
    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)  # Construct full file path
        print(f"Processing file: {file_path}")
        process_single_cell_data(file_path, save_as=save_as, compress=compress)  # Call the processing function


def unified_sc_data_processing(base_path, keep_original=False, save_as='npy', compress=False):
    """
    Process all single-cell data in each subdirectory of the specified base path.

    Parameters:
    - base_path (str): The base directory containing subdirectories with single-cell data.
    - save_as (str): File format for saving ('npy' or 'pickle'). Defaults to 'npy'.
    - compress (bool): Whether to compress the saved file. Defaults to False.

    Returns:
    - None. Processes each subdirectory in the base path.
    """
    # Check if the provided base path exists
    if not os.path.exists(base_path):
        print(f"The base path {base_path} does not exist.")
        return

    # List all subdirectories in the base path
    subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    if not subdirectories:
        print("No subdirectories found in the specified base path.")
        return

    # Process each subdirectory
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(base_path, subdirectory)  # Construct full subdirectory path
        print(f"Processing subdirectory: {subdirectory_path}")
        
        # Extract files in the subdirectory
        extract_files(subdirectory_path, keep_original=keep_original)
        
        # Process all .txt files in the subdirectory
        process_all_sc_data(subdirectory_path, save_as=save_as, compress=compress)


def load_single_processed_data(file_path, convert_to_dense=False):
    """
    Load processed single-cell data from a specified file.

    Parameters:
    - file_path (str): Path to the input file (.npz or .pkl).
    - convert_to_dense (bool): If True, convert the expression matrix to a dense numpy array. Defaults to False.

    Returns:
    - tuple: A tuple containing gene names and expression matrix.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    # Determine the file format and load accordingly
    if file_path.endswith('.npz'):
        # Load from .npz file
        data = sparse.load_npz(file_path)
        gene_names = None  # Gene names are not stored in .npz format
        expression_matrix = data.toarray() if convert_to_dense else data  # Convert if requested
        print(f"Loaded data from {file_path} with shape: {expression_matrix.shape}")
    elif file_path.endswith('.pkl') or file_path.endswith('.pkl.bz2'):
        # Load from .pkl or .pkl.bz2 file
        with bz2.BZ2File(file_path, 'rb') if file_path.endswith('.pkl.bz2') else open(file_path, 'rb') as f:
            data = pickle.load(f)
            gene_names = data['gene_names']
            expression_matrix = data['expression_matrix'].toarray() if convert_to_dense else data['expression_matrix']  # Convert if requested
            print(f"Loaded data from {file_path} with {len(gene_names)} genes.")
    else:
        raise ValueError("Unsupported file format. Please provide a .npz or .pkl file.")

    return gene_names, expression_matrix

def load_all_processed_data(directory, convert_to_dense=False):
    """
    Load processed single-cell data from all files in the specified directory.

    Parameters:
    - directory (str): The directory containing the processed files (.npz or .pkl).
    - convert_to_dense (bool): If True, convert the expression matrix to a dense numpy array. Defaults to False.

    Returns:
    - tuple: A tuple containing two lists: one for gene names and one for expression matrices.
    """
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return [], []  # Return empty lists if the directory does not exist

    # List all files in the directory
    files = os.listdir(directory)
    all_gene_names = []  # List to store all gene names
    all_expression_matrices = []  # List to store all expression matrices

    # Process each file
    for file in files:
        file_path = os.path.join(directory, file)  # Construct full file path
        if file.endswith(('.npz', '.pkl', '.pkl.bz2')):  # Check for valid file extensions
            print(f"Loading data from: {file_path}")
            gene_names, expression_matrix = load_single_processed_data(file_path, convert_to_dense=convert_to_dense)
            all_gene_names.append(gene_names)  # Append gene names to the list
            all_expression_matrices.append(expression_matrix)  # Append expression matrix to the list

    return all_gene_names, all_expression_matrices

        
if __name__ == "__main__":
    # Example usage
    # selected_path = "D:/Files/Theoretical Physics/CausalDiscovery/Data/Pancreas/GSE134355_RAW"
    # extract_files(selected_path, keep_original=False)  # Replace 'selected_path with the path you want to check
    # #selected_file = selected_path + "/" + "GSM3943045_Adult-Bone-Marrow1_dge.txt"
    # process_all_sc_data(selected_path, save_as='pickle', compress=False)
    # #process_single_cell_data(selected_file, save_as='pickle', compress=True)
    # base_path = "D:/Files/Theoretical Physics/CausalDiscovery/Data/Pancreas"
    # unified_sc_data_processing(base_path, keep_original=False, save_as='pickle', compress=True)
    # gene_names, expression_matrix = load_single_processed_data("D:\Files\Theoretical Physics\CausalDiscovery\Data\Pancreas\GSE134355_RAW_processed\GSM3943045_Adult-Bone-Marrow1_dge.pkl.bz2", convert_to_dense=False)
    processed_file_path = "D:\Files\Theoretical Physics\CausalDiscovery\Data\Pancreas\GSE134355_RAW_processed"
    all_gene_names, all_expression_matrices = load_all_processed_data(processed_file_path, convert_to_dense=False)