from numpy import array
import pandas as pd
from os import getcwd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool2D, Flatten, Dropout, Dense, Activation, MaxPooling3D

def load_data(study_name):
    file_path = f'{getcwd()}/database_data/Manipulated/{study_name}/expression_matrix.csv'
    df = pd.read_csv(file_path)
    return df

data = load_data('GSE137140')

#Create necessary data variables
expression_data = data.copy()
del expression_data['target']

target_data = data['target']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(expression_data, target_data, train_size=0.7, random_state=3)
X_train = array(X_train).reshape(2746, 2550, 1)
X_test = array(X_test).reshape(1178, 2550, 1)
y_train = pd.Series(y_train.reset_index()['target'])
y_test = pd.Series(y_test.reset_index()['target'])
#
# model = Sequential()
# model.add(Conv3D(32, kernel_size=(3,3,3), strides=(1, 1, 1), input_shape=(2746, 2550, 1)))
# model.add(Activation('relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#
# model.compile(X_train, y_train)
