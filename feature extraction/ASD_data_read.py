import sys
import re
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib.numpy_pickle_utils import xrange
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement, permutations, product
from kmer import kmer_featurization

min_max_scaler = preprocessing.MinMaxScaler()


def pre_data(data_X, k_data):  # 处理序列数据
    # k = 5
    print('Start k-mer, and k =', k_data)
    obj = kmer_featurization(k_data)
    dataset = np.zeros((len(data_X), 4 ** k_data))
    for i in range(len(data_X)):
        kmer_feature = obj.obtain_kmer_feature_for_one_sequence(data_X[i], write_number_of_occurrences=True)  # True>1
        # kmer_feature = min_max_scaler.fit_transform(np.reshape(kmer_feature, (len(kmer_feature), -1)))
        # kmer_feature = np.reshape(kmer_feature, (len(kmer_feature),))
        kmer_feature = kmer_feature / len(data_X[i])
        dataset[i] = kmer_feature
    print("dataset's shape after k-mer", dataset.shape)  # (len, 4 ** k)
    print('dataset finish')
    # return train_X, train_y, test_X, test_y, k_data
    return dataset, k_data


print('Start Read File')
# s_data = pd.read_csv("my/expression+seq_ASD+Disease.csv")
s_data = pd.read_csv("my/lncRNA_seq+exp values.csv")
# 得到y标签
# y_data = s_data[['Gene Type']]
# y_data = np.asarray(y_data)
# y = to_categorical(y_data)
# 得到序列数据
seq_data = s_data[['seq']]
seq = list(seq_data['seq'])
'''
 获取序列表达值特征
'''
exp_data = s_data[
    ['Ocx_8pcw_M_13058', 'M1C_S1C_8pcw_M_13058', 'AMY_8pcw_M_13058', 'MGE_8pcw_M_13058', 'STC_8pcw_M_13058',
     'URL_8pcw_M_13058', 'CGE_8pcw_M_13058', 'DTH_8pcw_M_13058', 'MFC_8pcw_M_13058', 'DFC_8pcw_M_13058',
     'OFC_8pcw_M_13058', 'LGE_8pcw_M_13058', 'ITC_8pcw_M_13058', 'HIP_8pcw_M_13058', 'VFC_8pcw_M_13058',
     'PCx_8pcw_M_13058', 'DFC_9pcw_M_12833', 'MFC_9pcw_M_12833', 'AMY_9pcw_M_12833', 'DTH_9pcw_M_12833',
     'URL_9pcw_M_12833', 'LGE_9pcw_M_12833', 'M1C_S1C_9pcw_M_12833', 'MGE_9pcw_M_12833', 'TCx_9pcw_M_12833',
     'HIP_9pcw_M_12833', 'Ocx_9pcw_M_12833', 'CGE_9pcw_M_12833', 'OFC_9pcw_M_12833', 'PCx_9pcw_M_12833',
     'HIP_12pcw_F_12835', 'DTH_12pcw_F_12835', 'AMY_12pcw_F_12835', 'A1C_12pcw_F_12835', 'V1C_12pcw_F_12835',
     'MFC_12pcw_F_12835', 'STR_12pcw_F_12835', 'M1C_12pcw_F_12835', 'DFC_12pcw_F_12835', 'ITC_12pcw_F_12835',
     'VFC_12pcw_F_12835', 'STC_12pcw_F_12835', 'IPC_12pcw_F_12835', 'OFC_12pcw_F_12835', 'S1C_12pcw_F_12835',
     'STC_12pcw_F_12960', 'V1C_12pcw_F_12960', 'A1C_12pcw_F_12960', 'CB_12pcw_F_12960', 'ITC_12pcw_F_12960',
     'IPC_12pcw_F_12960', 'S1C_12pcw_F_12960', 'DTH_12pcw_F_12960', 'HIP_12pcw_F_12960', 'DFC_12pcw_F_12960',
     'OFC_12pcw_F_12960', 'M1C_12pcw_F_12960', 'VFC_12pcw_F_12960', 'AMY_12pcw_F_12960', 'STR_12pcw_F_12960',
     'AMY_12pcw_F_13060', 'M1C_12pcw_F_13060', 'ITC_12pcw_F_13060', 'VFC_12pcw_F_13060', 'STR_12pcw_F_13060',
     'CBC_12pcw_F_13060', 'DTH_12pcw_F_13060', 'DFC_12pcw_F_13060', 'OFC_12pcw_F_13060', 'S1C_12pcw_F_13060',
     'V1C_12pcw_F_13060', 'MFC_12pcw_F_13060', 'HIP_12pcw_F_13060', 'IPC_12pcw_F_13060', 'A1C_12pcw_F_13060',
     'STR_13pcw_M_12820', 'M1C_13pcw_M_12820', 'AMY_13pcw_M_12820', 'S1C_13pcw_M_12820', 'V1C_13pcw_M_12820',
     'A1C_13pcw_M_12820', 'HIP_13pcw_M_12820', 'VFC_13pcw_M_12820', 'MFC_13pcw_M_12820', 'STC_13pcw_M_12820',
     'OFC_13pcw_M_12820', 'IPC_13pcw_M_12820', 'DFC_13pcw_M_12820', 'ITC_13pcw_M_12820', 'STR_13pcw_F_12834',
     'S1C_13pcw_F_12834', 'V1C_13pcw_F_12834', 'AMY_13pcw_F_12834', 'IPC_13pcw_F_12834', 'CB_13pcw_F_12834',
     'DFC_13pcw_F_12834', 'OFC_13pcw_F_12834', 'STC_13pcw_F_12834', 'A1C_13pcw_F_12834', 'HIP_13pcw_F_12834',
     'M1C_13pcw_F_12834', 'ITC_13pcw_F_12834', 'MD_13pcw_F_12834', 'MFC_13pcw_F_12834', 'VFC_13pcw_F_12834',
     'S1C_13pcw_M_12888', 'DFC_13pcw_M_12888', 'CB_13pcw_M_12888', 'AMY_13pcw_M_12888', 'ITC_13pcw_M_12888',
     'HIP_13pcw_M_12888', 'IPC_13pcw_M_12888', 'A1C_13pcw_M_12888', 'VFC_13pcw_M_12888', 'STR_13pcw_M_12888',
     'OFC_13pcw_M_12888', 'MFC_13pcw_M_12888', 'V1C_13pcw_M_12888', 'M1C_13pcw_M_12888', 'VFC_16pcw_M_12287',
     'OFC_16pcw_M_12287', 'STC_16pcw_M_12287', 'DFC_16pcw_M_12287', 'IPC_16pcw_M_12287', 'A1C_16pcw_M_12287',
     'MFC_16pcw_M_12287', 'STR_16pcw_M_12287', 'MD_16pcw_M_12287', 'V1C_16pcw_M_12287', 'VFC_16pcw_M_12837',
     'STC_16pcw_M_12837', 'IPC_16pcw_M_12837', 'A1C_16pcw_M_12837', 'MD_16pcw_M_12837', 'AMY_16pcw_M_12837',
     'V1C_16pcw_M_12837', 'ITC_16pcw_M_12837', 'CBC_16pcw_M_12837', 'HIP_16pcw_M_12837', 'STR_16pcw_M_12837',
     'OFC_16pcw_M_12837', 'DFC_16pcw_M_12837', 'M1C_16pcw_M_12837', 'MFC_16pcw_M_12837', 'S1C_16pcw_M_12837',
     'HIP_16pcw_M_12879', 'ITC_16pcw_M_12879', 'DFC_16pcw_M_12879', 'IPC_16pcw_M_12879', 'STC_16pcw_M_12879',
     'V1C_16pcw_M_12879', 'MD_16pcw_M_12879', 'STR_16pcw_M_12879', 'M1C_S1C_16pcw_M_12879', 'A1C_16pcw_M_12879',
     'VFC_16pcw_M_12879', 'AMY_16pcw_M_12879', 'MFC_16pcw_M_12879', 'STR_17pcw_F_12880', 'STC_17pcw_F_12880',
     'MD_17pcw_F_12880', 'A1C_17pcw_F_12880', 'HIP_17pcw_F_12880', 'DFC_17pcw_F_12880', 'VFC_17pcw_F_12880',
     'IPC_17pcw_F_12880', 'M1C_S1C_17pcw_F_12880', 'V1C_17pcw_F_12880', 'CBC_17pcw_F_12880', 'OFC_17pcw_F_12880',
     'AMY_17pcw_F_12880', 'MFC_17pcw_F_12880', 'STC_19pcw_F_12885', 'IPC_19pcw_F_12885', 'M1C_S1C_19pcw_F_12885',
     'MFC_19pcw_F_12885', 'V1C_19pcw_F_12885', 'HIP_19pcw_F_12885', 'A1C_19pcw_F_12885', 'MD_19pcw_F_12885',
     'VFC_19pcw_F_12885', 'STR_19pcw_F_12885', 'DFC_19pcw_F_12885', 'CBC_21pcw_F_12365', 'ITC_21pcw_F_12365',
     'MFC_21pcw_M_12886', 'M1C_21pcw_M_12886', 'STR_21pcw_M_12886', 'HIP_21pcw_M_12886', 'STC_21pcw_M_12886',
     'ITC_21pcw_M_12886', 'AMY_21pcw_M_12886', 'VFC_21pcw_M_12886', 'OFC_21pcw_M_12886', 'V1C_21pcw_M_12886',
     'S1C_21pcw_M_12886', 'CBC_21pcw_M_12886', 'IPC_21pcw_M_12886', 'DFC_21pcw_M_12886', 'HIP_24pcw_M_12288',
     'AMY_24pcw_M_12288', 'DFC_24pcw_M_12288', 'S1C_24pcw_M_12288', 'MD_24pcw_M_12288', 'STR_24pcw_M_12288',
     'A1C_24pcw_M_12288', 'IPC_24pcw_M_12288', 'OFC_24pcw_M_12288', 'STC_24pcw_M_12288', 'ITC_24pcw_M_12288',
     'VFC_24pcw_M_12288', 'M1C_24pcw_M_12288', 'MFC_24pcw_M_12288', 'CBC_24pcw_M_12288', 'V1C_24pcw_M_12288',
     'A1C_25pcw_F_12948', 'STC_26pcw_F_12949', 'V1C_26pcw_F_12949', 'DFC_26pcw_F_12949', 'CBC_35pcw_F_12295',
     'VFC_35pcw_F_12295', 'A1C_37pcw_M_263195015', 'OFC_37pcw_M_263195015', 'AMY_37pcw_M_263195015',
     'V1C_37pcw_M_263195015', 'MFC_37pcw_M_263195015', 'MD_37pcw_M_263195015', 'CBC_37pcw_M_263195015',
     'STR_37pcw_M_263195015', 'IPC_37pcw_M_263195015', 'ITC_37pcw_M_263195015', 'S1C_37pcw_M_263195015',
     'DFC_37pcw_M_263195015', 'STC_37pcw_M_263195015', 'M1C_37pcw_M_263195015', 'VFC_37pcw_M_263195015',
     'HIP_37pcw_M_263195015', 'STC_4mos_M_12296', 'STR_4mos_M_12296', 'V1C_4mos_M_12296', 'CBC_4mos_M_12296',
     'HIP_4mos_M_12296', 'MD_4mos_M_12296', 'ITC_4mos_M_12296', 'AMY_4mos_M_12296', 'MFC_4mos_M_12296',
     'M1C_4mos_M_12889', 'DFC_4mos_M_12889', 'OFC_4mos_M_12889', 'A1C_4mos_M_12889', 'STC_4mos_M_12889',
     'VFC_4mos_M_12889', 'ITC_4mos_M_12889', 'AMY_4mos_M_12889', 'M1C_4mos_M_12890', 'MD_4mos_M_12890',
     'OFC_4mos_M_12890', 'STR_4mos_M_12890', 'STC_4mos_M_12890', 'A1C_4mos_M_12890', 'HIP_4mos_M_12890',
     'AMY_4mos_M_12890', 'MFC_4mos_M_12890', 'V1C_4mos_M_12890', 'ITC_4mos_M_12890', 'S1C_4mos_M_12890',
     'IPC_4mos_M_12890', 'CBC_4mos_M_12890', 'DFC_4mos_M_12890', 'VFC_4mos_M_12890', 'S1C_10mos_M_12977',
     'IPC_10mos_M_12977', 'STC_10mos_M_12977', 'DFC_10mos_M_12977', 'OFC_10mos_M_12977', 'MFC_10mos_M_12977',
     'CBC_10mos_M_12977', 'ITC_10mos_M_12977', 'MD_10mos_M_12977', 'V1C_10mos_M_12977', 'AMY_1yrs_F_12830',
     'M1C_1yrs_F_12830', 'DFC_1yrs_F_12830', 'STC_1yrs_F_12830', 'V1C_1yrs_F_12830', 'MFC_1yrs_F_12830',
     'OFC_1yrs_F_12830', 'ITC_1yrs_F_12830', 'S1C_1yrs_F_12830', 'VFC_1yrs_F_12830', 'CBC_1yrs_F_12830',
     'A1C_1yrs_F_12830', 'MD_1yrs_F_12830', 'HIP_1yrs_F_12830', 'IPC_1yrs_F_12830', 'STR_1yrs_F_12830',
     'VFC_2yrs_F_12979', 'IPC_2yrs_F_12979', 'S1C_2yrs_F_12979', 'ITC_2yrs_F_12979', 'MD_2yrs_F_12979',
     'DFC_2yrs_F_12979', 'MFC_2yrs_F_12979', 'CBC_2yrs_F_12979', 'HIP_2yrs_F_12979', 'OFC_2yrs_F_12979',
     'V1C_2yrs_F_12979', 'STC_2yrs_F_12979', 'STR_3yrs_F_12836', 'STC_3yrs_F_12836', 'IPC_3yrs_F_12836',
     'ITC_3yrs_F_12836', 'CBC_3yrs_F_12836', 'AMY_3yrs_F_12836', 'VFC_3yrs_F_12836', 'M1C_3yrs_F_12836',
     'A1C_3yrs_F_12836', 'MD_3yrs_F_12836', 'V1C_3yrs_F_12836', 'OFC_3yrs_M_12980', 'MFC_3yrs_M_12980',
     'M1C_3yrs_M_12980', 'S1C_3yrs_M_12980', 'STC_3yrs_M_12980', 'IPC_3yrs_M_12980', 'A1C_3yrs_M_12980',
     'CBC_3yrs_M_12980', 'V1C_3yrs_M_12980', 'AMY_3yrs_M_12980', 'VFC_3yrs_M_12980', 'DFC_3yrs_M_12980',
     'HIP_3yrs_M_12980', 'ITC_3yrs_M_12980', 'AMY_4yrs_M_12298', 'STR_4yrs_M_12298', 'VFC_4yrs_M_12298',
     'DFC_4yrs_M_12298', 'STC_4yrs_M_12298', 'CBC_4yrs_M_12298', 'MD_4yrs_M_12298', 'ITC_8yrs_M_12841',
     'S1C_8yrs_M_12841', 'IPC_8yrs_M_12841', 'V1C_8yrs_M_12841', 'STC_8yrs_M_12841', 'MD_8yrs_M_12841',
     'VFC_8yrs_M_12841', 'HIP_8yrs_M_12841', 'AMY_8yrs_M_12841', 'M1C_8yrs_M_12841', 'CBC_8yrs_M_12841',
     'DFC_8yrs_M_12841', 'STR_8yrs_M_12841', 'A1C_8yrs_M_12841', 'MFC_8yrs_M_12841', 'OFC_8yrs_M_12841',
     'MFC_8yrs_M_12981', 'CBC_8yrs_M_12981', 'A1C_8yrs_M_12981', 'VFC_8yrs_M_12981', 'HIP_8yrs_M_12981',
     'AMY_8yrs_M_12981', 'V1C_8yrs_M_12981', 'IPC_8yrs_M_12981', 'DFC_8yrs_M_12981', 'STC_8yrs_M_12981',
     'ITC_8yrs_M_12981', 'VFC_11yrs_F_12289', 'IPC_11yrs_F_12289', 'ITC_11yrs_F_12289', 'STC_11yrs_F_12289',
     'S1C_11yrs_F_12289', 'DFC_11yrs_F_12289', 'AMY_11yrs_F_12289', 'V1C_11yrs_F_12289', 'CBC_11yrs_F_12289',
     'A1C_11yrs_F_12289', 'HIP_11yrs_F_12289', 'M1C_11yrs_F_12289', 'OFC_11yrs_F_12289', 'MFC_11yrs_F_12289',
     'MFC_13yrs_F_12831', 'IPC_13yrs_F_12831', 'VFC_13yrs_F_12831', 'A1C_13yrs_F_12831', 'ITC_13yrs_F_12831',
     'STC_13yrs_F_12831', 'CBC_13yrs_F_12831', 'V1C_13yrs_F_12831', 'S1C_13yrs_F_12831', 'DFC_13yrs_F_12831',
     'OFC_13yrs_F_12831', 'MD_13yrs_F_12831', 'AMY_13yrs_F_12831', 'M1C_13yrs_F_12831', 'HIP_13yrs_F_12831',
     'STR_13yrs_F_12831', 'IPC_15yrs_M_12299', 'CBC_15yrs_M_12299', 'STC_15yrs_M_12299', 'ITC_15yrs_M_12299',
     'AMY_15yrs_M_12299', 'HIP_18yrs_M_12984', 'M1C_18yrs_M_12984', 'V1C_18yrs_M_12984', 'A1C_18yrs_M_12984',
     'ITC_18yrs_M_12984', 'VFC_18yrs_M_12984', 'MFC_18yrs_M_12984', 'STC_18yrs_M_12984', 'DFC_18yrs_M_12984',
     'S1C_18yrs_M_12984', 'IPC_18yrs_M_12984', 'OFC_18yrs_M_12984', 'CBC_18yrs_M_12984', 'STC_19yrs_F_12832',
     'CBC_19yrs_F_12832', 'DFC_19yrs_F_12832', 'AMY_19yrs_F_12832', 'OFC_19yrs_F_12832', 'ITC_19yrs_F_12832',
     'STR_19yrs_F_12832', 'S1C_19yrs_F_12832', 'M1C_19yrs_F_12832', 'HIP_19yrs_F_12832', 'A1C_19yrs_F_12832',
     'V1C_19yrs_F_12832', 'IPC_19yrs_F_12832', 'VFC_19yrs_F_12832', 'MFC_19yrs_F_12832', 'MD_19yrs_F_12832',
     'MFC_21yrs_F_13057', 'M1C_21yrs_F_13057', 'A1C_21yrs_F_13057', 'DFC_21yrs_F_13057', 'AMY_21yrs_F_13057',
     'MD_21yrs_F_13057', 'V1C_21yrs_F_13057', 'STC_21yrs_F_13057', 'VFC_21yrs_F_13057', 'CBC_21yrs_F_13057',
     'S1C_21yrs_F_13057', 'ITC_21yrs_F_13057', 'IPC_21yrs_F_13057', 'HIP_21yrs_F_13057', 'STR_21yrs_F_13057',
     'OFC_21yrs_F_13057', 'ITC_23yrs_M_12300', 'AMY_23yrs_M_12300', 'OFC_23yrs_M_12300', 'MD_23yrs_M_12300',
     'STC_23yrs_M_12300', 'VFC_23yrs_M_12300', 'MFC_23yrs_M_12300', 'M1C_23yrs_M_12300', 'HIP_23yrs_M_12300',
     'CBC_23yrs_M_12300', 'S1C_23yrs_M_12300', 'A1C_23yrs_M_12300', 'IPC_23yrs_M_12300', 'STR_23yrs_M_12300',
     'ITC_30yrs_F_12290', 'STR_30yrs_F_12290', 'MFC_30yrs_F_12290', 'OFC_30yrs_F_12290', 'IPC_30yrs_F_12290',
     'STC_30yrs_F_12290', 'S1C_30yrs_F_12290', 'MD_30yrs_F_12290', 'HIP_30yrs_F_12290', 'A1C_30yrs_F_12290',
     'V1C_30yrs_F_12290', 'DFC_30yrs_F_12290', 'AMY_30yrs_F_12290', 'M1C_30yrs_F_12290', 'VFC_30yrs_F_12290',
     'CBC_30yrs_F_12290', 'M1C_36yrs_M_12302', 'STC_36yrs_M_12302', 'MFC_36yrs_M_12302', 'ITC_36yrs_M_12302',
     'IPC_36yrs_M_12302', 'AMY_36yrs_M_12302', 'OFC_36yrs_M_12302', 'VFC_36yrs_M_12302', 'DFC_36yrs_M_12302',
     'A1C_36yrs_M_12302', 'V1C_36yrs_M_12302', 'CBC_36yrs_M_12302', 'S1C_36yrs_M_12302', 'HIP_36yrs_M_12302',
     'STR_36yrs_M_12302', 'MD_36yrs_M_12302', 'S1C_37yrs_M_12303', 'V1C_37yrs_M_12303', 'CBC_37yrs_M_12303',
     'HIP_37yrs_M_12303', 'IPC_37yrs_M_12303', 'MD_37yrs_M_12303', 'DFC_37yrs_M_12303', 'VFC_37yrs_M_12303',
     'ITC_37yrs_M_12303', 'A1C_37yrs_M_12303', 'M1C_37yrs_M_12303', 'OFC_37yrs_M_12303', 'STR_37yrs_M_12303',
     'STC_37yrs_M_12303', 'AMY_37yrs_M_12303', 'MFC_37yrs_M_12303', 'DFC_40yrs_F_12304', 'ITC_40yrs_F_12304',
     'VFC_40yrs_F_12304', 'MD_40yrs_F_12304', 'AMY_40yrs_F_12304', 'A1C_40yrs_F_12304', 'CBC_40yrs_F_12304',
     'V1C_40yrs_F_12304', 'OFC_40yrs_F_12304', 'STC_40yrs_F_12304', 'IPC_40yrs_F_12304', 'M1C_40yrs_F_12304',
     'HIP_40yrs_F_12304', 'STR_40yrs_F_12304', 'S1C_40yrs_F_12304']]
exp_data = np.asarray(exp_data)
for i in range(len(exp_data)):
    s = min_max_scaler.fit_transform(np.reshape(exp_data[i], (len(exp_data[i]), -1)))
    exp_data[i] = np.reshape(s, (len(exp_data[i]),))
exp_data = min_max_scaler.fit_transform(exp_data)
print("exp_data'shape:", exp_data.shape)
'''
k-mer编码
'''
# seq_data_2, k_2 = pre_data(seq, 2)
# seq_data_2 = min_max_scaler.fit_transform(seq_data_2)
# seq_data_3, k_3 = pre_data(seq, 3)
# seq_data_3 = min_max_scaler.fit_transform(seq_data_3)
seq_data_8, k = pre_data(seq, 8)
seq_data_8 = min_max_scaler.fit_transform(seq_data_8)

pd.DataFrame(seq_data_8).to_csv('feature select/seq_data_8_lncRNA.csv', header=None, index=False)
pd.DataFrame(exp_data).to_csv('feature select/exp_data_lncRNA.csv', header=None, index=False)
# pd.DataFrame(seq_data_3).to_csv('feature select/seq_data_3.csv', header=None, index=False)
# pd.DataFrame(seq_data_4).to_csv('feature select/seq_data_4.csv', header=None, index=False)
#
# PseKNC_3 = pd.read_csv('feature select/PseKNC_k=3.csv', header=None).astype(dtype='float64')
# PseKNC_3 = np.asarray(PseKNC_3)
# for i in range(len(PseKNC_3)):
#     s = PseKNC_3[i]
#     s = s / len(PseKNC_3[i])
#     PseKNC_3[i] = np.reshape(s, (len(PseKNC_3[i]),))
# PseKNC_3 = min_max_scaler.fit_transform(PseKNC_3)
# print("PseKNC_3'shape:", PseKNC_3.shape)
# pd.DataFrame(PseKNC_3).to_csv('feature select/PseKNC_k=3_after_mms.csv', header=None, index=False)
#
# PseKNC_4 = pd.read_csv('feature select/PseKNC_k=4.csv', header=None).astype(dtype='float64')
# PseKNC_4 = np.asarray(PseKNC_4)
# for i in range(len(PseKNC_4)):
#     s = PseKNC_3[i]
#     s = s / len(PseKNC_3[i])
#     PseKNC_3[i] = np.reshape(s, (len(PseKNC_3[i]),))
# PseKNC_4 = min_max_scaler.fit_transform(PseKNC_4)
# print("exp_data'shape:", PseKNC_4.shape)
# pd.DataFrame(PseKNC_4).to_csv('feature select/PseKNC_k=4_after_mms.csv', header=None, index=False)


print('data read finish')
