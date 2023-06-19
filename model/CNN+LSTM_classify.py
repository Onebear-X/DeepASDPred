import math
from keras import backend as K
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras import Model, Input, layers, regularizers
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Activation, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D, \
    Reshape, AveragePooling1D, LSTM, GlobalAveragePooling1D, Bidirectional, Embedding, Multiply, CuDNNLSTM
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, classification_report, \
    confusion_matrix, precision_recall_curve, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime
import tensorflow as tf
from keras.backend import set_session

aa = datetime.datetime.now()  # 开始时间点
# 得到y标签
s_data = pd.read_csv("my/expression+seq_ASD+Disease.csv")
y_data = s_data[['Gene Type']]
y = to_categorical(y_data)

all_features = pd.read_csv('my feature select/seqdata_select_LR_k=8+exp_auc.csv', header=None)

input_data1 = np.asarray(all_features)
print("input_data1.shape:", input_data1.shape)


def lr_reduce():
    ES = EarlyStopping(monitor='val_loss',  # val_loss  loss
                       patience=2, verbose=1, mode='auto')
    # 减小学习率
    RD = ReduceLROnPlateau(monitor='val_loss',
                           factor=0.1,
                           patience=2,
                           verbose=1,
                           mode='auto',
                           # min_delta=1e-10,
                           cooldown=2,
                           min_lr=0)
    # 下降的幅度如何设置
    return ES, RD


def lstm_cnnlstm():
    inputs_cnn = Input(shape=(input_data1.shape[1],))
    x_cnn = Reshape((input_data1.shape[1], 1))(inputs_cnn)
    x_cnn = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(x_cnn)
    x_cnn = Dropout(0.5)(x_cnn)
    x_cnn = Flatten()(x_cnn)

    inputs_lstm = Input(shape=(input_data1.shape[1],))
    # x_lstm = Reshape((input_data1.shape[1], 1))(inputs_lstm)
    x_lstm = Embedding(input_dim=input_data1.shape[1]+1, output_dim=32)(inputs_lstm)
    x_lstm = CuDNNLSTM(32, return_sequences=True)(x_lstm)
    # ATTENTION PART STARTS HERE 注意力层
    attention_probs = Dense(32, activation='softmax', name='attention_vec', kernel_initializer='he_normal')(x_lstm)
    attention_mul = Multiply()([x_lstm, attention_probs])
    # ATTENTION PART FINISHES HERE
    x_lstm = Flatten()(attention_mul)

    x_con = layers.concatenate([x_cnn, x_lstm], axis=1)

    x_con = Dense(64, activation='relu')(x_con)
    outputs = Dense(2, activation='softmax')(x_con)

    m = Model(inputs=[inputs_cnn, inputs_lstm], outputs=outputs)
    m.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=1e-4),
              metrics=['categorical_accuracy'])
    return m


EarlyStop, Reduce = lr_reduce()

losslist = []
acclist = []
random_state = 42

roc_auc_list = []
pr_auc_list = []
pr_auc_list1 = []
precision_list = []
sensitivity_list = []
specificity = []
accuracy_list = []
mcc_list = []
f1_score_list = []
sk_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
i = 0
lstm_cnnlstm().summary()

bb = datetime.datetime.now()  # 结束时间点
cc = bb - aa  # 运行时间，单位是  时:分:秒
print("分类一次十折时间为", cc)
