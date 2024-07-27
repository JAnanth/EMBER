import pandas as pd
import numpy as np
from os import getcwd, listdir

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

def load_data(file_path):
    # Initialize necessary local variables
    i = 0
    num_controls = 0
    df = pd.DataFrame()
    target_ordered_lst = []
    target_dict = {}
    #Created unordered dictionary with all target values regarding presence of LC
    for patient in listdir(file_path):
        num = int(patient[3:10])
        if (num >= 4067600 and num <= 4069165) or (num >= 4071314):
            target_dict[patient] = 1
        else:
            target_dict[patient] = 0
            num_controls += 1
    #Sort dictionary for standardization of data order
    target_dict = dict(sorted(target_dict.items()))
    #Read in all necessary data and fill out the dataframe
    for k,v in target_dict.items():
        target_ordered_lst.append(v)
        sub_df = pd.read_csv(file_path + k, on_bad_lines='skip', names=['miRNA', 'vals'])
        df[i] = sub_df['vals']
        i += 1
    #Reorganize df and add the target data in separate column
    df.rename(sub_df['miRNA'], inplace=True)
    df = df.transpose()
    df['target'] = pd.Series(target_ordered_lst)
    return df, num_controls

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

class Preprocessing:
    ''' The preprocessing class handles the cleaning and standardization of the data such that it is ready to be inputted into EMBER. All code from this close is original.'''
    def __init__(self, start_patient_id, num_patients, base_path, study_name):
        self.start_patient_id = start_patient_id
        self.num_patients = num_patients
        self.base_path = base_path
        self.study_name = study_name

    def add_commas(self):
        n_patients = self.start_patient_id
        for i in range(self.num_patients):
            file_path = getcwd() + f'/data/DirectPull/GSE137140/primary/GSM{n_patients}-tbl-1.txt'
            n_patients += 1
            # Read in the file
            with open(file_path, 'r') as file:
              filedata = file.read()
            # Replace the target strinh
            filedata = filedata.replace('\t', ',')
            # Write the file out again
            with open(file_path, 'w') as file:
              file.write(filedata)

    def initialize_and_store_data(self):
        #Read in the data
        input_path = f'{self.base_path}/DirectPull/{self.study_name}/primary/'
        df, num_controls = load_data(input_path)
        #Save the data to the directory for later use
        output_path = f'{self.base_path}/Manipulated/{self.study_name}/expression_matrix.csv'
        df.to_csv(output_path, index=False)
        return num_controls

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#IMPLEMENTATION OF THE CONSTRUCTED CLASSES (All code is ORIGINAL)
base_path = getcwd() + '/data'
preprocessor = Preprocessing(4067570, 3924, base_path, 'GSE137140')
num_controls = preprocessor.initialize_and_store_data()
