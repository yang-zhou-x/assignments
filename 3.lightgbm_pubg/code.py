# -*- coding: utf-8 -*-
"""
Created on 2018
@author: Yang Zhou, zhouyang0995@ruc.edu.cn
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import lightgbm as lgb
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 导入实验数据：
t0 = time.time()
train_set = pd.read_csv(filepath_or_buffer='~/Downloads/train_V2.csv')
t = round(time.time() - t0, 2)
print(f'读取数据用时：{t}s')  # 18.62秒

# 查看数据集信息：
train_set.info(verbose=False)  # 数据集共有4,446,966行、29列，占用内存大小为984MB左右。
# 查看数据集：
train_set.head(5)  # 数据集共有29列。

# 查看特征名称及其数据类型：
print('feature_name', ' \t ', 'data_type')
print('-' * 30)
print(train_set.dtypes)
# 除玩家ID(Id)、玩家所在队伍ID(groupId)、该场比赛ID(matchId)、比赛游戏模式(matchType)为文本型外，其余特征均为数值型，符合实际意义。
# 其中玩家ID、玩家所在队伍ID、该场比赛的ID为哈希值。

# 查看数据的缺失情况
print('feature_name', '\t', 'number_missing')
print('-' * 30)
print(train_set.isna().sum())  # 只有最终百分比排名（winPlacePerc）存在1个缺失值。考虑到数据量有四百多万条，故直接删去该缺失记录。
train_set.dropna(axis=0, inplace=True)

# 可视化各数值型变量之间的相关性：
plt.figure(figsize=(10, 8))
sns.heatmap(data=train_set.corr(), vmax=1., cmap='RdBu', annot=False, linewidths=.1, linecolor='white', square=True)
plt.title('Correlations on PUBG Features')
plt.tight_layout()
# 当特征与目标winPlacePerc之间的相关系数接近0时，或者存在两个特征高度相关时，可以考虑删去相应特征：
train_set.drop(labels=['killPoints', 'maxPlace', 'numGroups', 'matchDuration', 'rankPoints', 'roadKills',
                       'winPoints', 'vehicleDestroys', 'teamKills'], axis=1, inplace=True)
# 综合考虑后，结合特征的实际含义，删去了当场比赛中有数据的队伍数（numGroups）等9个特征

# 目标是预测玩家的最终百分比排名，因此使用直方图对数据集中的最终百分比排名（winPlacePerc）进行可视化：
plt.figure(figsize=(10, 5))
sns.distplot(train_set.iloc[:, -1], bins=100, kde=False)  # 不进行高斯核密度估计，速度更快
plt.title('Histogram of winPlacePerc')
plt.tight_layout()

# 百分数排名并不是均匀分布，这是因为数据集中存在单人比赛、两人组队和四人组队等情况，当玩家处于队伍中时，队员排名相等，均为队伍排名。
# 此外，还存在第一人称、第三人称等游戏模式。
# 对游戏模式进行分组计数并可视化：
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='matchType', data=train_set)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.title('Number of Players on different Match Type')
plt.xlabel('')
plt.tight_layout()


# 基于上述信息建立特征工程
def team_rank(data):
    """
    计算一场比赛中各支参赛队伍在各个特征上的百分比排名（相同时取平均秩），并返回替换为新特征后的数据集
    玩家ID、队伍ID、比赛ID、游戏模式和最终排名除外
    :param data: pd.DataFrame
    :return: data
    """
    col_without = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in data.columns if col not in col_without]
    agg = data.groupby(by=['groupId', 'matchId'])[features].agg('mean')  # 一场比赛中一支队伍在各个特征上的平均值
    agg = agg.groupby('matchId')[features].rank(pct=True)  # 同一场比赛中各支参赛队伍在各个特征上的百分比排名（相同时取平均秩）
    data = data.drop(features, axis=1)
    return pd.merge(data, agg, suffixes=['', '_team_rank'], how='left', on=['groupId', 'matchId'])  # 有重名时，才加后缀


# 使用新特征替换原数据集中相应的部分：
t0 = time.time()
train_set = team_rank(data=train_set)
t = round(time.time() - t0, 2)
print(f'特征转换用时：{t}s')  # 58.83秒

# 查看新数据集的信息：
print(train_set.info(verbose=False))  # 数据集行数、列数不变，此时内存占用为1018M左右
gc.collect()
# 查看特征的数据类型：
print(train_set.dtypes)

# 查看比赛模式(matchType)的具体情况：
print(train_set.loc[:5, 'matchType'])
# 转换成整数型分类数据：
encoder = OrdinalEncoder()
train_set['matchType_int'] = encoder.fit_transform(train_set.loc[:, ['matchType']])
train_set['matchType_int'] = train_set['matchType_int'].astype('int32')

# 参与建模的特征
features = [f for f in train_set.columns if f not in ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']]
print(len(features))  # 16个特征参与建立LightGBM模型

# 划分训练集和测试集，比例为7:3
x_train, x_test, y_train, y_test = train_test_split(train_set[features], train_set['winPlacePerc'],
                                                    test_size=.3, random_state=2018)
# 建立模型
reg_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                            num_leaves=8,  # 基分类器的最大叶节点数量2**3=8
                            max_depth=4,  # 最大深度4
                            learning_rate=0.01,
                            n_estimators=150,  # 基分类器数量150
                            n_jobs=-1,
                            importance_type='split')  # 特征在模型中的使用次数作为衡量其重要性的指标
# 拟合模型
t0 = time.time()
reg_lgb.fit(X=x_train,
            y=y_train,
            categorical_feature=['matchType_int'])
t = round(time.time() - t0, 2)
print(f'拟合模型用时：{t}s')  # 58.8秒

# 进行模型预测：
t0 = time.time()
y_pred = reg_lgb.predict(X=x_test)
t = round(time.time() - t0, 2)
print(f'模型预测用时：{t}s')  # 9.13秒

# 计算均方误差
mse = round(mean_squared_error(y_true=y_test, y_pred=y_pred), 4)
print(f'测试集上的均方误差为{mse}')  # 0.02

# 将特征重要性可视化：
lgb.plot_importance(booster=reg_lgb, ylabel='', title='LightGBM Regression FI')
plt.tight_layout()
