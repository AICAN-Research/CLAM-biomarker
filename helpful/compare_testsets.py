import pandas as pd
import os

# Folder containing the CSV files
folder_path = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/biomarker_ER_100'

# File pattern for the split files (from 0 to 9)
file_pattern = 'splits_{}.csv'

# Initialize a list to store the 'test' column values from each file
test_columns = []
file_names = []

# Load each CSV file and extract the 'test' column
for i in range(10):
    file_path = os.path.join(folder_path, file_pattern.format(i))
    try:
        df = pd.read_csv(file_path)
        test_column = df['test'].dropna().tolist()
        test_columns.append(test_column)
        file_names.append(file_pattern.format(i))
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except KeyError:
        print(f"'test' column not found in {file_path}.")

# Compare all test columns and print differences
reference_column = test_columns[0]
differences_found = False

for idx, test_column in enumerate(test_columns[1:], start=1):
    differences = [(i, reference_column[i], test_column[i])
                   for i in range(len(reference_column))
                   if reference_column[i] != test_column[i]]

    if differences:
        differences_found = True
        print(f"Differences in file {file_names[idx]} compared to {file_names[0]}:")
        for i, ref_val, diff_val in differences:
            print(f"  Row {i}: {file_names[0]} = {ref_val}, {file_names[idx]} = {diff_val}")

if not differences_found:
    print("The 'test' column is consistent across all files.")