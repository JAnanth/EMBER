# Necessary Module Imports
import pandas as pd
import numpy as np
from os import getcwd, listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

def load_preprocessed_data(study_name):
    file_path = f'{getcwd()}/data/Manipulated/{study_name}/expression_matrix.csv'
    df = pd.read_csv(file_path)
    return df

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

class FeatureSelection():
    '''The feature selection uses Deep Learning to select the 7 most statistically significant genes. All code is ORIGINAL.'''
    def __init__(self, study_name):
        self.study_name = study_name

    def create_DL_network(self):
        data = load_preprocessed_data(self.study_name)
        #Create necessary data variables
        expression_data = data.copy()
        del expression_data['target']
        target_data = data['target']
        #Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(expression_data, target_data, train_size=0.7, random_state=3)
        y_train = pd.Series(y_train.reset_index()['target'])
        y_test = pd.Series(y_test.reset_index()['target'])
        #Implementation of algorithms to make predictions
        preds_alg = MLPClassifier(random_state=42)
        preds_alg.fit(X_train, y_train)
        preds = preds_alg.predict(X_test)
        cfs = preds_alg.coef_
        #return the relevant values and metrics for outlier identification
        self.y_test = y_test
        self.preds = preds
        return np.array(y_test), preds

    def get_stats(self):
        #Get basic stats
        acc = accuracy_score(self.y_test, self.preds)
        num_wrong = round(len(y_test)*(1-acc))
        #Get true pos, false pos, etc. for calculating further stats
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for i in range(len(y_test)):
            if y_test[i] == 0 and preds[i] == 1:
                fp += 1
            elif y_test[i] == 1 and preds[i] == 0:
                fn += 1
            elif y_test[i] == 0 and preds[i] == 0:
                tn += 1
            else:
                tp += 1
        #Put all stats together for return val
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        data = {'Accuracy (%)': [round(acc*100,3)],
                'Precision (%)': [round(prec*100,3)],
                'Recall (%)': [round(recall*100,3)],
                'F1_Score': [round((100*(2*prec*recall)/(prec+recall)),3)]
                }
        stats_df = pd.DataFrame(data)
        #Return the stats dataframe and other info incase needed
        return stats_df

#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#
#IMPLEMENTATION OF THE CONSTRUCTED CLASSES (All code is ORIGINAL)
test = FeatureSelection('GSE137140')
y_test,preds = test.create_DL_network()
stats_df = test.get_stats()
stats_df
