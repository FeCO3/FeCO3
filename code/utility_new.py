from scipy.stats import ttest_ind
from scipy.stats import levene
import pandas as pd
import numpy as np
from pandas import Series
import math
import copy

# import numpy as np
# import pandas as pd
from minepy import MINE
from pandas import DataFrame, Series
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.cross_validation import cross_val_score, train_test_split
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
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
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


def ACC(estimator_, X, y):
    scores = cross_val_score(estimator_, X, y, cv=5)
    accuracy = np.mean(scores)
    return accuracy


def ACC2(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    correct = (y_pre == y_test).sum()
    # print("correct:",correct)
    accuracy = correct / len(y_test)
    return accuracy


def Recall(estimator_, X, y):
    scores = cross_val_score(estimator_, X, y, cv=5, scoring='recall')
    recall = np.mean(scores)
    return recall


def best_ACC(X, y):
    estimators = [SVC(random_state=0), DecisionTreeClassifier(random_state=0), LogisticRegression(random_state=0),
                  GaussianNB(), KNeighborsClassifier(), RandomForestClassifier(random_state=0)]
    acc_max = 0
    # counttemp=0
    acc_list = []
    for clf in estimators:
        acc_temp = ACC(clf, X, y)
        acc_list.append(acc_temp)
    # acc_ind=np.argsort(-np.array(acc_list))
    best_clf = estimators[np.argmax(acc_list)]

    return np.max(acc_list), best_clf


def MIC(a, b):  # return the MIC between a and b
    mine = MINE()
    mine.compute_score(a, b)
    MIC = mine.mic()
    # print('MIC=',MIC)
    return MIC


def McOne(F, C, r):
    micFC = np.array(list(map(lambda x: MIC(x, C), F.T)))  # rank the features base on their MIC
    fea_rank = np.argsort(-micFC)
    Subset = fea_rank[np.where(micFC[fea_rank] >= r)]
    numSubset = Subset.shape[0]  # Subset[1...numSubset] containing the selected features
    # rank the items in Subset in the descending order by micFC[Subset[i]]

    for e in range(0, numSubset):
        q = e + 1
        while q < numSubset:
            if MIC(F.T[Subset[e]], F.T[Subset[q]]) >= micFC[Subset[q]]:  # redundant feature discrimination
                Subset[q:numSubset - 1] = Subset[q + 1:numSubset]
                numSubset -= 1
            else:
                q += 1
    Subset = Subset[:numSubset]
    return F.T[Subset].T, Subset  # FReduce


# r=0.3, t=3
def McTwo(F, C, r, t):
    # C = list(C)
    FReduce, FReduce_rank = McOne(F, C, r)
    F_Mctwo = []
    F_temp = []
    current_macc = 0
    temp_macc = 0
    decrease = 0

    print('McOne Finish!!!!!!!!!')
    FReduce_size = len(FReduce_rank)
    print('feature num after McOne:', FReduce_size)
    while (len(F_Mctwo) < FReduce_size):
        # print(len(F_Mctwo))
        all_acc = []
        all_clf = []
        all = list(map(lambda x: best_ACC(F.T[F_Mctwo + [x]].T, C), FReduce_rank))

        for i in all:
            all_acc.append(i[0])
            all_clf.append(i[1])
        # print(all_acc)
        if np.max(all_acc) < current_macc:
            decrease = decrease + 1
            if (decrease >= t):
                for i in range(t - 1):
                    F_Mctwo.pop()
                return len(F_Mctwo), F_Mctwo, current_macc, best_clf2
        else:
            current_macc = np.max(all_acc)
            decrease = 0
            # acc_ind2=np.argsort(-np.array(all_acc))
            best_clf2 = all_clf[np.argmax(all_acc)]

        F_Mctwo.append(FReduce_rank[np.argmax(all_acc)])
        FReduce_rank = np.delete(FReduce_rank, [np.argmax(all_acc)])

    for i in range(decrease):
        F_Mctwo.pop()
    return len(F_Mctwo), F_Mctwo, current_macc, best_clf2



'''
function: rank the feature according to their importance. Return the index of features.
'''
def t_ranking(data, label):
    print('********************************************************')

    print('t_test')
    data_T = []
    data_N = []
    afterTtestValue = []
    for i in range(len(label)):
        if label[i] == 1:
            data_T.append(data[i])
        else:
            data_N.append(data[i])

    data_T = np.array(data_T)
    data_N = np.array(data_N)

    for i in range(len(data.T)):
        x = data_T[:, i]
        y = data_N[:, i]
        sx = Series(x)
        sy = Series(y)
        temp1 = levene(sx.astype(float), sy.astype(float))
        if temp1[1] > 0.05:
            temp2 = ttest_ind(sx.astype(float), sy.astype(float))
            afterTtestValue.append(temp2[1])
        else:
            temp2 = ttest_ind(sx.astype(float), sy.astype(float), equal_var=False)
            afterTtestValue.append(temp2[1])
    afterTtestValue = np.array(afterTtestValue)
    t_ranking_index = np.argsort(afterTtestValue)
    return t_ranking_index


def w_ranking(data, label):
    print('***********************************************************')

    print('w_test')
    data_T = []
    data_N = []
    afterWilTestValue = []
    for i in range(len(label)):
        if label[i] == 1:
            data_T.append(data[i])
        else:
            data_N.append(data[i])

    data_T = np.array(data_T)
    data_N = np.array(data_N)

    for i in range(len(data.T)):
        x = data_T[:, i]
        y = data_N[:, i]
        ax = set(x)
        ay = set(y)
        if (ax == ay) == True: continue
        sx = Series(x)
        sy = Series(y)
        afterWilTestValue.append(stats.mannwhitneyu(sx.astype(float), sy.astype(float))[1])

    afterWilTestValue = np.array(afterWilTestValue)
    w_ranking_index = np.argsort(afterWilTestValue)
    return w_ranking_index


def mic_ranking(data, label):
    print('***********************************************************')
    print('mic')  # 最大信息系数
    micsub = np.array(
        list(map(lambda x: MIC(x, label), data.T)))  # rank the features base on their MIC
    mic_ranking_index = np.argsort(-micsub)
    return mic_ranking_index


def decision_tree_ranking(data, label):
    print('***********************************************************')
    # RFS-DecisionTree
    print('DecisionTree')
    clf = DecisionTreeClassifier(random_state=0)
    # help(clf)
    clf.fit(data, label)
    d_ranking_index = np.argsort(-clf.feature_importances_)
    return d_ranking_index


def L2_LR_ranking(data, label):
    print('***********************************************************')
    # RFS-LR
    print('L2-LR')
    clf = LogisticRegression(random_state=0)
    clf.fit(data, label)
    lr_ranking_index = np.argsort(-clf.coef_[0])
    return lr_ranking_index


def L2_SVM_ranking(data, label):
    print('***********************************************************')
    # RFS-SVM
    print('L2-SVM')
    clf = SVC(kernel="linear", random_state=0)
    clf.fit(data, label)
    svm_ranking_index = np.argsort(-clf.coef_[0])
    return svm_ranking_index


def xgb_ranking(data, label):
    print('***********************************************************')
    # -XGB
    print('XGBoost')
    clf = XGBClassifier()
    clf.fit(data, label)

    x_ranking_index = np.argsort(-clf.feature_importances_)
    return x_ranking_index




'''
function: use ifs strategy to select features that have best performance. Print accuracy of each classifier 
          and the original features used(both index and value).
'''
def ifs_new(data, label, t_ranking_index, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data):
    t_fea = []
    t_best_acc = 0
    t_num = 0
    t_ind = []
    t_decrease = 0

    for i in range(len(label)):
        t_fea.append([data[i, t_ranking_index[0]]])
    t_ind.append(t_ranking_index[0])
    # print(svm_fea)
    cur_t_macc, clf_t = best_ACC(t_fea, list(label))

    while (len(t_fea) <= len(data.T)):
        if (cur_t_macc > t_best_acc):
            t_num += 1
            t_best_acc = cur_t_macc
            best_clf_t = clf_t
            t_decrease = 0
        else:
            t_decrease += 1
            t_num += 1
            if (t_decrease >= ifs_threshold):
                break;
        if (len(t_fea) == len(data.T)):
            break;
        for i in range(len(label)):
            t_fea[i].append(data[i, t_ranking_index[t_num]])
        t_ind.append(t_ranking_index[t_num])
        cur_t_macc, clf_t = best_ACC(t_fea, list(label))
    for ifs_k in range(t_decrease):
        for i in range(len(label)):
            t_fea[i].pop()
        t_ind.pop()
    print('num=', t_num - t_decrease)
    print('macc=', t_best_acc)
    print('best_clf', best_clf_t)
    # print(d_ind)

    s_acc = ACC(SVC(random_state=0), t_fea, list(label))
    d_acc = ACC(DecisionTreeClassifier(random_state=0), t_fea, list(label))
    lr_acc = ACC(LogisticRegression(random_state=0), t_fea, list(label))
    g_acc = ACC(GaussianNB(), t_fea, list(label))
    k_acc = ACC(KNeighborsClassifier(), t_fea, list(label))
    r_acc = ACC(RandomForestClassifier(random_state=0), t_fea, list(label))
    print('各分类器：')

    print('SVC-acc:', s_acc)
    print('DecisionTreeClassifier-acc:', d_acc)
    print('LogisticRegression-acc:', lr_acc)
    print('GaussianNB-acc:', g_acc)
    print('KNeighborsClassifier-acc:', k_acc)
    print('RandomForestClassifier-acc:', r_acc)

    print('********The feature index used for classification is:')
    for item in t_ind:
        print('This feature is constrcuted by the following original features[index]:')
        print(ranking1[ind1_i[ranking2[ind2_i[ranking3[item]]]]])
        print(ranking1[ind1_i[ranking2[ind2_j[ranking3[item]]]]])
        print(ranking1[ind1_j[ranking2[ind2_i[ranking3[item]]]]])
        print(ranking1[ind1_j[ranking2[ind2_j[ranking3[item]]]]])

    print('**********The feature matrix used for classification is')
    for item in t_ind:
        print('This feature is constrcuted by the following original features[value]:')
        print(original_data[:,ranking1[ind1_i[ranking2[ind2_i[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_i[ranking2[ind2_j[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_j[ranking2[ind2_i[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_j[ranking2[ind2_j[ranking3[item]]]]]])
    print('---------------------------------------------------------------------')
    return


def use_ifs_to_select_feature(data, label, ifs_threshold, mcTwoR, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data):
    tr = t_ranking(data, label)
    ifs_new(data, label, tr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    wr = w_ranking(data, label)
    ifs_new(data, label, wr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    mr = mic_ranking(data, label)
    ifs_new(data, label, mr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    dtr = decision_tree_ranking(data, label)
    ifs_new(data, label, dtr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    llr = L2_LR_ranking(data, label)
    ifs_new(data, label, llr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    lsr = L2_SVM_ranking(data, label)
    ifs_new(data, label, lsr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    xgbr = xgb_ranking(data, label)
    ifs_new(data, label, xgbr, ifs_threshold, ranking1, ind1_i, ind1_j, ranking2, ind2_i, ind2_j, ranking3,
            original_data)

    print('***********************************************************')
    print("mcTwo:")
    fea_num, fea_ind, macc, best_clf = McTwo(data, label, mcTwoR, ifs_threshold)
    print('num=', fea_num)
    print('macc=', macc)
    print('best clf:', best_clf)
    fea_selected = []
    for i in fea_ind:
        fea_selected.append(data.T[i])
    fea_selected = np.array(fea_selected).T

    print('----------------------------------')
    print('McTwo_SVC')
    McTwoAccSVC = ACC(SVC(random_state=0), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccSVC)
    print('----------------------------------')

    print('McTwo_DT')
    McTwoAccDT = ACC(DecisionTreeClassifier(random_state=0), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccDT)
    print('----------------------------------')

    print('McTwo_LR')
    McTwoAccLR = ACC(LogisticRegression(random_state=0), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccLR)
    print('----------------------------------')

    print('McTwo_NByes')
    McTwoAccNByes = ACC(GaussianNB(), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccNByes)
    print('----------------------------------')

    print('McTwo_KNN')
    McTwoAccKNN = ACC(KNeighborsClassifier(), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccKNN)
    print('----------------------------------')

    print('McTwo_RFC')
    McTwoAccRfc = ACC(RandomForestClassifier(random_state=0), fea_selected, label)
    print('num=', len(fea_selected.T))
    print('acc=', McTwoAccRfc)



    print('********The feature index used for classification is:')
    for item in fea_ind:
        print('This feature is constrcuted by the following original features[index]:')
        print(ranking1[ind1_i[ranking2[ind2_i[ranking3[item]]]]])
        print(ranking1[ind1_i[ranking2[ind2_j[ranking3[item]]]]])
        print(ranking1[ind1_j[ranking2[ind2_i[ranking3[item]]]]])
        print(ranking1[ind1_j[ranking2[ind2_j[ranking3[item]]]]])

    print('**********The feature matrix used for classification is')
    for item in fea_ind:
        print('This feature is constrcuted by the following original features[value]:')
        print(original_data[:,ranking1[ind1_i[ranking2[ind2_i[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_i[ranking2[ind2_j[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_j[ranking2[ind2_i[ranking3[item]]]]]])
        print(original_data[:,ranking1[ind1_j[ranking2[ind2_j[ranking3[item]]]]]])

    print('---------------------------------------------------------------------')

    return



'''
Function: fiter feature number to m using ttest . Return selected features and their ranking.

:parameters:
    data: in the format of array, the shape is  [sample , feature].
    label: in the format of array, the shape is [sample , 1].
    m: threshold, select top-m features.
:return:
    fea_select: top-m features in the format of array, the shape is [sample , m]
    ranking: the ranking of features
'''
def filter_fea_num_to_m_using_ttest(data, label, m):
    data_T = []
    data_N = []
    afterTtestValue = []
    for i in range(len(label)):
        if label[i] == 1:
            data_T.append(data[i])
        else:
            data_N.append(data[i])

    data_T = np.array(data_T)
    data_N = np.array(data_N)

    for i in range(len(data.T)):
        x = data_T[:, i]
        y = data_N[:, i]
        sx = Series(x)
        sy = Series(y)
        temp1 = levene(sx.astype(float), sy.astype(float))
        if temp1[1] > 0.05:
            temp2 = ttest_ind(sx.astype(float), sy.astype(float))
            afterTtestValue.append(temp2[1])
        else:
            temp2 = ttest_ind(sx.astype(float), sy.astype(float), equal_var=False)
            afterTtestValue.append(temp2[1])
    afterTtestValue = np.array(afterTtestValue)
    t_ranking_index = np.argsort(afterTtestValue)

    # print(t_ranking_index)
    fea_select = []
    for i in range(m):
        if i < len(data.T):
            fea_select.append(data[:, t_ranking_index[i]])

    print("The number of selected features is %d" % len(fea_select))
    return np.array(fea_select).T, t_ranking_index
