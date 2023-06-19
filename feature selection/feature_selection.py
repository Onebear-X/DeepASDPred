# from gcForest import GCForest
import datetime
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
aa = datetime.datetime.now()  # 开始时间点

# 读取特征数据
exp_data = pd.read_csv('feature select/exp_data.csv', header=None)
seq8_data = pd.read_csv('feature select/seq_data_8.csv', header=None)

all_features = np.column_stack((exp_data, seq8_data))

s_data = pd.read_csv("my/expression+seq_ASD+Disease.csv")
# 得到y标签
y_data = s_data[['Gene Type']]
target = np.asarray(y_data)

scorelist = []
percentilenumber = []
featurenumber = []
all_featuresScaled = all_features.shape
print("原始数据长度：", all_featuresScaled) # 1268
for i in np.arange(start=1, stop=200, step=0.1):
    selector = SelectPercentile(chi2, percentile=i)
    percentilenumber.append(math.floor(all_featuresScaled[1] * i))
    features_selected = selector.fit_transform(all_features, target)
    print(features_selected.shape)
    featurenumber.append(features_selected.shape[1])
    clf = LogisticRegression(random_state=42)
    # clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, features_selected, target, scoring='roc_auc', cv=5)
    print('The test roc_auc scores are: {} '.format(scores))
    print('The mean test roc_auc scores are: {:.3f} \n'.format(np.mean(scores)))
    scorelist.extend([np.mean(scores)])


percentilenumber = np.arange(start=1, stop=200, step=1)
percentilenumber.shape = [199, 1]
max_index = np.argmax(scorelist)
plt.plot(percentilenumber[max_index], scorelist[max_index], '.r', markersize=8)
plt.text(percentilenumber[max_index], scorelist[max_index],
         (np.around(percentilenumber[max_index][0], 1), (np.around(scorelist[max_index], decimals=3))))
print(percentilenumber[max_index][0])
print("--------")
plt.plot(percentilenumber, scorelist, 'b', markeredgecolor='k')
ax = plt.gca()
ax.set_xlabel('Feature Dimension(%)')
ax.set_ylabel('ROC AUC')
# ax.grid()
plt.savefig('pic/chi2+LR.tif', dpi=600, format='eps', bbox_inches='tight')
plt.show()


print('The selected feature numbers are: {}'.format(featurenumber[max_index]))

selector = SelectPercentile(chi2, percentile=percentilenumber[max_index])
features_selected = selector.fit_transform(all_features, target)
s1 = selector.get_support(True)#  获取最佳特征子集的索引数

print(features_selected.shape)
pd.DataFrame(features_selected).to_csv('my feature select/seqdata_select_LR_k=8+exp_acc_test_3-11.csv',
                                           header=None, index=False)
'''
The selected feature numbers are: 93
(2595, 93)
'''
bb = datetime.datetime.now()  # 结束时间点
cc = bb - aa  # 运行时间，单位是  时:分:秒
print("特征选择一次的时间为", cc)
print('process finish-------------------------------')

