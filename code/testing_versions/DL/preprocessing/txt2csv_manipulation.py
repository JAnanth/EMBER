import pandas as pd
from os import getcwd

def add_commas(n_patient_initial, n_patients_total):
    n_patients = n_patient_initial
    for i in range(n_patients_total):
        file_path = getcwd() + f'/data/primary/GSM{n_patients}-tbl-1.txt'
        n_patients += 1
        # Read in the file
        with open(file_path, 'r') as file:
          filedata = file.read()
        # Replace the target strinh
        filedata = filedata.replace('\t', ',')
        # Write the file out again
        with open(file_path, 'w') as file:
          file.write(filedata)
add_commas(4067570,3924)
# pd.read_csv(getcwd() + '/data/primary/GSM4067570-tbl-1.csv', on_bad_lines='skip', names=['miRNA','val'])
