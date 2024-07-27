import pandas as pd
from os import getcwd, listdir
import sklearn

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


def initialize_and_store_data(study_name, base_path):
    #Read in the data
    input_path = f'{base_path}/DirectPull/{study_name}/primary/'
    df, num_controls = load_data(input_path)
    #Save the data to the directory for later use
    output_path = f'{base_path}/Manipulated/{study_name}/expression_matrix.csv'
    df.to_csv(output_path, index=False)
    return num_controls

#implement above functions and save data for application
base_path = getcwd() + '/data'
num_controls = initialize_and_store_data('GSE137140', base_path)
num_controls
