import utility_new
from scipy.stats import ttest_ind
from scipy.stats import levene
import pandas as pd
import numpy as np
from pandas import Series
import math
import copy

#import numpy as np
#import pandas as pd
from minepy import MINE
from pandas import DataFrame, Series
from sklearn.model_selection import cross_val_score, train_test_split
#from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import array
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from numpy import vstack, array, nan
from sklearn.feature_selection import  SelectFromModel
from sklearn.svm import  LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import  SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
from scipy import stats
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
import time
warnings.filterwarnings("ignore")

'''
Function: Read data and label from file and convert to dataframe format

:parameters:
    data_filename: name of datafile, the shape is feature * sample, including index_col and header.
    label_filename: name of labelfile, the shape is sample * 1, including index_col and header.
:return:
    data_df: data in the format of dataframe, the shape is [sample , feature]
    label_df: label in the format of dataframe, the shape is [sample , 1]
'''
def read_data(data_filename, label_filename):
    data_df = pd.read_csv(data_filename, index_col = 0, header = 0)
    label_df = pd.read_csv(label_filename, index_col = 0, header = 0)
    print(label_df.describe())
    #print(data_df.head())
    #print(label_df.head())
    #print(label_df[label_df['class'] == 0])
    #print(data_df.values)
    print(data_filename)
    print(data_df.shape)
    return data_df.T, label_df




'''
Function: Select 5 same samples (label is 0) as reference samples and remove them and return them.

:parameters:
    data: in the format of array, the shape is  [sample , feature].
    label: in the format of array, the shape is [sample , 1].
    ref_sample_num: number of reference samples.
:return:
    new_data: data without 5 reference samples in the format of array, the shape is [sample , feature]
    new_label: label without 5 reference samples in the format of array, the shape is [sample , 1]
    ref_samples: 5 reference samples in the format of array, the shape is [sample(5) , feature]
'''
def remove_ref_samples(data, label, ref_sample_num = 5):
    new_data = []
    new_label = []
    cnt = 0
    flag = []
    for i in range(len(label)):
        if label[i] == 0:
            new_data.append(data[i])
            new_label.append(label[i])
            flag.append(i)
            cnt += 1
        if cnt == ref_sample_num:
            break
    ref_samples = np.array(new_data)
    for i in range(len(label)):
        if i not in flag:
            new_data.append(data[i])
            new_label.append(label[i])
    new_data = np.array(new_data)[ref_sample_num:,:]
    new_label = (new_label)[ref_sample_num:]

    return new_data, new_label, ref_samples


'''
Function: construct new feature  and return them.

:parameters:
    data: in the format of array, the shape is  [sample , feature].
    label: in the format of array, the shape is [sample , 1].
    ttest_threshold: select top-ttest_threshold features to construct features.
:return:
    new_data: new features in the format of array, the shape is [sample , feature]
    rest_label: label without 5 reference samples in the format of array, the shape is [sample , 1]
    ranking: the ranking of original data,
    ind_i: to search original feature
    ind_j: to search original feature
'''
def feature_construct_step1(data, label, ttest_threshold):
    data_select, ranking = utility_new.filter_fea_num_to_m_using_ttest(data, label, ttest_threshold)
    rest_data, rest_label, ref_samples = remove_ref_samples(data_select, label, 5)
    print('rest data shape:', rest_data.shape)
    print('rest label shape:', len(rest_label))
    print('reference sample shape:', ref_samples.shape)
    mean_ref_sample = np.mean(ref_samples, axis = 0)
    var_ref_sample = np.var(ref_samples, axis = 0)

    def sPCC1(i, j):
        c = (cur_sample[i] - mean_ref_sample[i]) * (cur_sample[j] - mean_ref_sample[j])
        v = math.sqrt(var_ref_sample[i] * var_ref_sample[j])
        if c == 0:
            return 0
        if v == 0:
            return 1000000000000
        return abs(c / v)


    new_data = []
    ind_i = []
    ind_j = []
    for k in range(len(rest_label)):
        cur_sample = rest_data[k,:]
        spcc = [[0 for i in range(ttest_threshold)] for j in range(ttest_threshold)]
        spcc1d = []
        for i in range(ttest_threshold):
            for j in range(i+1, ttest_threshold):
                spcc[i][j] = sPCC1(i, j)
                spcc1d.append(spcc[i][j])
                if k==0:
                    ind_i.append(i)
                    ind_j.append(j)
        new_data.append(spcc1d)
    return np.array(new_data), rest_label, ranking, ind_i, ind_j








if __name__ == '__main__':
    file_path = ''
    ifs_threshold = 3
    mcTwoR = 0.1
    first_ttest = 800
    second_ttest = 1100

    data_df ,label_df = read_data(file_path + 'clean_matrix.csv', file_path + 'label.csv')
    data = data_df.values
    label = label_df.values.T[0]
    #data, label = remove_ref_samples(data_df.values, label_df.values)
    #label = label.T[0]
    print('data shape:', data.shape)
    print('label shape:', len(label))
    new_data1, new_label1, ranking1, ind1_i, ind1_j = feature_construct_step1(data, label, first_ttest)
    print('data shape after 1st construction:', new_data1.shape)
    print('label shape after 1st construction:', len(new_label1))
    new_data2, new_label2, ranking2, ind2_i, ind2_j = feature_construct_step1(new_data1, new_label1, second_ttest)
    print('data shape after 2nd construction:', new_data2.shape)
    print('label shape after 2nd construction:', len(new_label2))
    new_data, ranking3 = utility_new.filter_fea_num_to_m_using_ttest(new_data2, new_label2, data.shape[1])


    utility_new.use_ifs_to_select_feature(new_data, new_label2, ifs_threshold, mcTwoR, ranking1, ind1_i, ind1_j,
                                      ranking2, ind2_i, ind2_j, ranking3, data)

