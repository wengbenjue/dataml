import os
import sys
import string
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scorecardpy as sc
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
# import catboost as CB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_curve, precision_recall_curve, precision_recall_fscore_support
import pickle
from datetime import datetime as dt
# import shap
import seaborn as sns
from scipy.stats import scoreatpercentile
from sklearn import metrics
import copy
import scipy.stats as stats
# 定义逐步回归函数
import statsmodels.api as sm

from scipy.linalg import solve
import math
import re

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from pylab import mpl

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm.auto import trange
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif




from sklearn.preprocessing import OneHotEncoder

import pandas as pd



try:
    from wordcloud import WordCloud
except:
    pass

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd

from .my_logging import logger
from .feature_selector import FeatureSelector

plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Source Han Sans CN'

class CustomScaler:
    def __init__(self, file_name='scaler.pkl'):
        self.scaler = MinMaxScaler()
        self.file_name = file_name
    def fit_transform(self, X_train):
        self.scaler.fit(X_train)
        with open(self.file_name, 'wb') as file:
            pickle.dump(self.scaler, file)
        X_train_scaled = self.scaler.transform(X_train)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    def load_and_transform(self, X_test):
        with open(self.file_name, 'rb') as file:
            loaded_scaler = pickle.load(file)
            X_test_scaled = loaded_scaler.transform(X_test)
        return pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


def feature_classify(df, target='target'):
    '''
    将变量划分为标志型、分类型、连续型分别处理

    Parameters:
        df - 宽表数据Dataframe

    Returns:
        ind_df_list - 标志型变量
        cd_df_list - 分类型变量
        num_df_list - 连续型变量
    '''

    ind_df_list = []
    cd_df_list = []
    num_df_list = []

    for col in list(df):
        if col not in ['target',target]:
            if col.endswith('_ind'):
                ind_df_list.append(col)
                df[col] = df[col].astype('str')
            elif col.endswith('_cd'):
                cd_df_list.append(col)
                df[col] = df[col].astype('str')
            else:
                if col not in ['target', target]:
                    num_df_list.append(col)
    df_clean = df[num_df_list].copy()
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
    var_numrics = df_clean.select_dtypes(include=numerics).columns
    features = df_clean.columns
    numerical_features = [i for i in features if (i in var_numrics) and (i not in ['target', target])]
    category_features = [i for i in features if i not in numerical_features]
    df[category_features] = df[category_features].astype('str')
    cat_list = cd_df_list + category_features
    num_list = numerical_features
    logger.info('剩余分类型变量个数：%d' % len(cat_list))
    logger.info('剩余连续型变量个数：%d' % len(num_list))
    logger.info('剩余变量总数：%d' % len(cat_list + num_list))
    return ind_df_list, cd_df_list, num_df_list, cat_list, num_list


def process_data(df, num_features, cat_features, stats_file='stats.json', mode='train'):
    """
    缺失值填充
    :param df:输入DataFrame
    :param num_features:
    :param cat_features:
    :param stats_file: 统计值存储文件
    :param mode: 区分是训练还是测试或者预测
    :return: DataFrame
    """
    df = df.copy()
    missing_values = ['NaN', 'None', 'NULL']
    df.replace(missing_values, np.nan, inplace=True)
    if mode == 'train':
        # num_means = df[num_features].mean()
        num_means = df[num_features].median()
        # cat_modes = df[cat_features].apply(lambda x: x.mode()[0]).iloc[0]
        cat_modes = df[cat_features].apply(lambda x: x.mode()[0])

        stat_data = {'num_means': num_means.to_dict(), 'cat_modes': cat_modes.to_dict()}
        with open(stats_file, 'w') as json_file:
            json.dump(stat_data, json_file)
    else:
        stat_data = pd.read_json(stats_file)
        num_means = pd.Series(stat_data['num_means'])
        cat_modes = pd.Series(stat_data['cat_modes'])
    df[num_features] = df[num_features].fillna(num_means)
    df[cat_features] = df[cat_features].apply(lambda x:x.fillna(cat_modes[x.name]))
    df[cat_features] = df[cat_features].fillna('Unknown')
    df[cat_features] = df[cat_features].replace('nan','Unknown')
    df[cat_features] = df[cat_features].replace(missing_values,'Unknown')
    df[num_features] = df[num_features].replace(missing_values,-9999)
    df[cat_features] = df[cat_features].replace(np.nan, 'Unknown')
    df[num_features] = df[num_features].replace(np.nan, -9999)
    return df

def null_rate_filter(df, r=0.80):
    # 缺失率剔除
    na_rate = df.isnull().sum() / df.shape[0]
    na_list = list(na_rate[na_rate >= r].index)
    df_list = list(na_rate[na_rate < r].index)
    df = df[df_list].copy()
    logger.info("缺失率高于{}的变量个数为：{}".format(r, len(na_list)))
    logger.debug("剔除缺失率高于{}的变量：".format(r))
    # logger.info(', '.join(na_list))
    return df


# 倾斜特征
def value_counts_stats(df, same_value_rate=0.85, target='target'):
    '''
    同值率处理
    Args:
        df - 输入DataFrame
        same_value_rate - 倾斜率阈值
        target - 目标y
   Returns：
       df_filter - 筛选完的DataFrame
       lean_cols - 倾斜率高于阈值的特征列
    '''
    logger.info(df.shape)
    lean_cols = []
    filter_cols = []
    for col in df.columns.tolist():
        if col == 'target' or col == target:
            filter_cols.append(col)
            continue
        # logger.info(col)
        df_col = df[col].value_counts().to_frame()
        df_col[col + '_true'] = df_col.index
        df_col.reset_index(inplace=True)
        df_col.rename(columns={col: 'counts'}, inplace=True)
        # top1_value = df_col['count'].head(1).tolist()[0]
        top1_value = df_col['counts'].head(1).tolist()[0]
        if top1_value / df_col['counts'].sum() >= same_value_rate:
            lean_cols.append(col)
        else:
            filter_cols.append(col)
    df_filter = df[filter_cols]
    logger.info("同值率高于{}的特征数量有:{}".format(same_value_rate, len(lean_cols)))
    # logger.info("同值率高于{}的特征：{}".format(same_value_rate,lean_cols))
    logger.info(df_filter.shape)
    return df_filter, lean_cols


def iv_filter(var_bin_df, min_iv_limit=0.02, max_iv_limit=0.5,model_dir=None):
    '''
    按IV筛选特征
    Args:
        var_bin_df - 输入DataFrame
        min_iv_limit - 最小iv阈值
        max_iv_limit - 最大iv阈值
        target - 目标y变量名
    Returns：
        iv_features - 筛选后的特征
        var_iv_list - IV DataFrame

    '''
    var_iv_list = (var_bin_df.drop_duplicates('variable'))[['variable', 'total_iv']].sort_values(by='total_iv',
                                                                                                 ascending=False)
    var_iv_list.rename(columns={'total_iv': 'IV'}, inplace=True)
    if model_dir is not None:
        var_iv_list.to_csv(os.path.join(model_dir,'all_iv_report.csv'),encoding='utf8',index=False)
    var_iv_list = var_iv_list.set_index('variable')
    var_iv_list = var_iv_list[(var_iv_list['IV'] > min_iv_limit) & (var_iv_list['IV'] < max_iv_limit)]
    iv_features = list(var_iv_list.index)
    logger.debug('IV筛选后的特征个数:{}'.format(len(iv_features)) )
    # logger.info('IV筛选后的特征:', ', '.join(iv_features))
    return iv_features, var_iv_list


def corr_filter(train_df, x_var_iv_list, iv_df, corr_threshold=0.8):
    '''
    pearson相关性剔除变量
    Args:
        x_var_iv_list - x变量名列表
        df - 输入的DataFrame
        iv_df - IV
    Returns:
        corr_filter_list 相关性剔除后的变量列表
    '''
    # plot_corr(train_df)
    corr_num_df = pd.DataFrame()
    tmp_num = [c for c in x_var_iv_list]
    for i1 in tmp_num:
        tmp_num.remove(i1)
        for i2 in tmp_num:
            tmp_corr = train_df[[i1, i2]].corr('pearson').loc[i1, i2]
            #             logger.info(i1, i2, tmp_corr)
            tmp_df = pd.DataFrame([(i1, i2, tmp_corr)])
            corr_num_df = pd.concat([corr_num_df, tmp_df], axis=0)
    corr_num_df.columns = ['col1', 'col2', 'pearson']

    # 相关性剔除：剔除pearson相关系数>=0.8 且 IV值相对较低的变量
    num_iv_col = corr_num_df[np.abs(corr_num_df['pearson']) >= corr_threshold].copy()
    tmp_num = [c for c in x_var_iv_list]
    num_rem = []
    for n_c in x_var_iv_list:
        tmp = list(num_iv_col[num_iv_col['col1'] == n_c]['col2'].values)
        if len(tmp) > 0:
            tmp.append(n_c)
            tmp = pd.DataFrame(tmp)
            tmp = tmp.merge(iv_df[['variable', 'IV']], how='left', left_on=0, right_on='variable')
            for rm in tmp[tmp['IV'] < tmp['IV'].max()]['variable'].values:
                try:
                    tmp_num.remove(rm)
                    num_rem.append(rm)
                #                     logger.info(rm)
                except Exception as e:
                    # logger.error(e)
                    pass
            same_iv_columns = tmp[tmp['IV'] == tmp['IV'].max()]['variable'].values
            if len(same_iv_columns) >1:
                try:
                    tmp_num.remove(rm)
                    num_rem.append(rm)
                except:
                    pass

    corr_filter_list = tmp_num
    logger.info('相关性剔除后的变量个数为：{}'.format(len(corr_filter_list)))
    logger.debug('相关性剔除后的变量：{}'.format(', '.join(corr_filter_list)))
    return corr_filter_list


# 评估指标

def evaluation_index(y_val, y_pred, y_probs, thld=True):
    """
    仅支持0/1预测
    y_val：真实值
    y_pred：预测值(0/1)
    y_probs：预测为1的概率值
    thld:是否根据KS阀值计算指标，默认是
    """
    ### ROC AUC KS LIFT
    auc = metrics.roc_auc_score(y_val, y_probs)
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_probs)

    ks = max(tpr - fpr)
    threshold = thresholds[(tpr - fpr) == max(tpr - fpr)][0]

    def lift_value(y_val, y_probs, n=20):
        prb = y_probs.copy()
        point1 = scoreatpercentile(prb, (n - 1) * (100 / n))
        point2 = scoreatpercentile(prb, n * (100 / n))
        lift1 = y_val[(prb >= point1) & (prb <= point2)].copy()
        return lift1.sum() * n / y_val.sum()

    lift = lift_value(y_val, y_probs)

    # 根据阀值计算评估指标
    if thld:
        y_pred_t = np.where(y_probs > threshold, 1, 0)
    else:
        y_pred_t = y_pred.copy()

    # 准确率
    acc = metrics.accuracy_score(y_val, y_pred_t)
    # 精确率
    precision = metrics.precision_score(y_val, y_pred_t)
    # 召回率
    recall = metrics.recall_score(y_val, y_pred_t)
    # F1_score
    f1 = metrics.f1_score(y_val, y_pred_t)
    # Gini
    gini = 2 * auc - 1
    evaluation_list = {'AUC': auc, 'KS': ks, 'Lift': lift, 'Accuracy': acc, 'Precision': precision, 'Recall': recall,
                       'F1': f1, 'gini': gini}
    return pd.DataFrame.from_dict(evaluation_list, orient='index', columns=['评估值'])


def most_value_plot(df, cat_features, num_features):
    # 检查特征最多值占比情况
    total_record_count = df.shape[0]
    col_most_values, col_large_value = {}, {}
    features = cat_features + num_features
    for col in features:
        value_count = df[col].groupby(df[col]).count()
        col_most_values[col] = value_count.sort_values(ascending=False).values[0] / total_record_count
        large_value = value_count[value_count == value_count.sort_values(ascending=False).values[0]].index[0]
        col_large_value[col] = large_value
    col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient='index')
    col_most_values_df.columns = ['max_percent']
    col_most_values_df = col_most_values_df.sort_values(by='max_percent', ascending=False)
    plt.bar(range(len(col_most_values_df['max_percent'])), height=col_most_values_df['max_percent'])
    plt.title('max value distribution')
    return col_most_values_df, col_large_value


def little_pvalue_check(col_most_values_df, df, col_large_value, y_target):
    # 探查多数值占比超过90%的特征的少数值的坏样本率是否显著高于多数值
    large_percent_cols = list(col_most_values_df[col_most_values_df['max_percent'] > 0.90].index)
    bad_rate_diff = {}
    for col in large_percent_cols:
        large_value = col_large_value[col]
        temp = df[[col, y_target]]
        temp[col] = temp.apply(lambda x: int(x[col] == large_value), axis=1)
        bad_rate = temp.groupby(col).mean()
        if bad_rate.iloc[0][y_target] == 0:
            bad_rate_diff[col] = 0
            continue
        bad_rate_diff[col] = np.log(bad_rate.iloc[0][y_target] / bad_rate.iloc[1][y_target])
    bad_rate_diff_sorted = sorted(bad_rate_diff.items(), key=lambda x: x[1], reverse=True)
    bad_rate_diff_sorted_vals = [x[1] for x in bad_rate_diff_sorted]
    plt.bar(x=range(len(bad_rate_diff_sorted_vals)), height=bad_rate_diff_sorted_vals)
    logger.debug(len(large_percent_cols))
    return bad_rate_diff


def pvalue_log_filter(bad_rate_diff, cat_features, num_features, bad_good_log_rate=2):
    # 删除少数值坏样本率和多数值坏样本率对数比例在2之内的特征
    large_percent_columns = [k for k, v in bad_rate_diff.items() if v <= bad_good_log_rate]
    features = cat_features + num_features
    for col in large_percent_columns:
        if col in num_features:
            num_features.remove(col)
        elif col in cat_features:
            cat_features.remove(col)

        features.remove(col)

    logger.info('删除多数值变量:{}，个数为：{}'.format(large_percent_columns, len(large_percent_columns)))
    return cat_features, num_features, features


def outlier_process(df,cache_dir, num_features, flag='train'):
    logger.debug('异常值3 sigma盖帽处理')
    df_desc = None
    df_rem = df[num_features].copy()
    outlier_path = os.path.join(cache_dir,'outlier_df_desc.pickle')
    if flag == 'train':
        ### 异常值处理
        #### 盖帽处理
        df_desc = df[num_features].describe().T

        df_desc['lower'] = np.floor(df_desc['mean'] - 3 * df_desc['std'])
        df_desc['upper'] = np.floor(df_desc['mean'] + 3 * df_desc['std'])
        df_desc['if_abn'] = np.where((df_desc['min'] < df_desc['lower']) | (df_desc['max'] > df_desc['lower']), 1, 0)
        abn_col = df_desc[df_desc['if_abn'] == 1].index.tolist()
        ## 保存训练集特征分布
        with open(outlier_path, "wb") as f:
            pickle.dump(df_desc, f)
            logger.debug('训练集分布保存成功!')
    else:
        # 加载训练集分布
        with open(outlier_path, "rb") as f:
            df_desc = pickle.load(f)
            logger.debug('加载训练集分布成功!')
        abn_col = num_features

    for abn in abn_col:
        df_rem[abn] = np.where(df_rem[abn] > df_desc.loc[abn, 'upper'], df_desc.loc[abn, 'upper'],
                               np.where(df_rem[abn] < df_desc.loc[abn, 'lower'], df_desc.loc[abn, 'lower'],
                                        df_rem[abn]))
    return df_rem


def quartile_filter(num_df_list, df):
    '''
    剔除四分位差为0的连续变量
    '''
    rem_num_df_list = num_df_list[:]
    df_desc = df[num_df_list].describe().T

    for col in df_desc.index:
        if (df_desc.loc[col, '75%'] - df_desc.loc[col, '25%']) == 0:
            #         logger.info(col)
            rem_num_df_list.remove(col)
    num_df_list = rem_num_df_list

    logger.info('剔除四分位差为0的连续型变量, 剩余：{}个'.format(len(num_df_list)))
    return num_df_list


# Lift图
def lift_cure(y_val, probs):
    result = pd.DataFrame({'tg': y_val, 'proba': probs})
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result.proba, i * (100 / 10))
        point2 = scoreatpercentile(result.proba, (i + 1) * (100 / 10))
        proba_copy[(result.proba >= point1) & (result.proba <= point2)] = ((10 - i))
    result['grade'] = proba_copy
    df_gain = result.groupby(['grade'])['tg'].agg(['sum', 'count'])
    df_gain['Lift'] = (df_gain['sum'] / df_gain['count']) / (result['tg'].sum() / len(result))
    df_gain['base'] = 1
    plt.plot(df_gain['Lift'])
    plt.plot(df_gain['base'], 'r--')
    plt.title('Lift_curve')
    plt.xlabel('Decile')
    plt.ylabel('Lift Value')
    plt.xticks(list(range(1, 11, 1)))
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.show()


def var_equal_bin(df, col, target, bins):
    """
    模型报告
    df:contains target and probs
    col:variable which to show performance
    """
    data = copy.copy(df)
    data['var_bin'] = pd.qcut(data[col], bins, duplicates='drop')
    total = data.groupby(['var_bin'])[target].count()
    total = pd.DataFrame({'var': col, 'total': total})
    positive = data.groupby(['var_bin'])[target].sum()
    positive = pd.DataFrame({'positive': positive})
    regroup = total.merge(positive, left_index=True, right_index=True, how='left')
    regroup.reset_index(inplace=True)
    regroup = regroup.set_index(['var'], drop=True).reset_index()
    regroup['negative'] = regroup['total'] - regroup['positive']
    N = sum(regroup['total'])
    B = sum(regroup['positive'])
    G = N - B
    regroup['positive_pcnt'] = regroup['positive'].map(lambda x: x * 1.0 / B)
    regroup['negative_pcnt'] = regroup['negative'].map(lambda x: x * 1.0 / G)
    regroup['PD'] = regroup.apply(lambda x: x.positive * 1.0 / x.total, axis=1)

    regroup.sort_values(by='var_bin', ascending=False, inplace=True)
    regroup['WOE'] = regroup.apply(lambda x: -np.log(x.negative_pcnt * 1.0 / (x.positive_pcnt + 0.00001)), axis=1)
    regroup['IV'] = regroup.apply(
        lambda x: -np.log(x.negative_pcnt * 1.0 / (x.positive_pcnt + 0.00001)) * (x.positive_pcnt - x.negative_pcnt), axis=1)
    regroup = regroup.assign(total_IV=lambda x: sum(x.IV),
                             cumtotal=lambda x: np.cumsum(x.total) / sum(x.total),
                             cumbpositive=lambda x: np.cumsum(x.positive) / sum(x.positive))

    regroup['KS'] = np.abs(regroup['cumbpositive'] - regroup['cumtotal'])
    regroup['LIFT'] = np.round(regroup['cumbpositive'] / regroup['cumtotal'],decimals=2)

    return (regroup)

def forest_selection(df_train, labels, missing_threshold=0.99, correlation_threshold=0.9,
                     cumutative_importance=0.99, max_depth=3):
    '''
    LightGBM选择特征
    ：param df_train_woe:
    ：param labels:
    ：return:
    '''

    fs = FeatureSelector(data=df_train, labels=labels)
    fs.identify_all(selection_params={'missing_threshold': missing_threshold,
                                      'eval_metric': 'binary_error',
                                      'correlation_threshold': correlation_threshold,
                                      'task': 'classification',
                                      'cumulative_importance': cumutative_importance,
                                      'max_depth': max_depth
                                      })
    df_train = fs.remove(methods='all')
    return df_train


from statsmodels.stats.outliers_influence import variance_inflation_factor

def checkVIF(df):
    df['c'] = 1  # 增加常数项
    name = df.columns.tolist()
    x = np.matrix(df)
    VIF_list = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    VIF = pd.DataFrame({'feature': name, "VIF": VIF_list})
    VIF = VIF[~(VIF['feature'] == 'c')]
    return VIF





def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True):
    """
    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
    Arguments:
         X - pandas.DataFrame with candidate features
         y - list-like with the target
         initial_list - list of features to start with (column names of X)
         threshold_in - include a feature if its p-value < threshold_in
         threshold_out - exclude a feature if its p-value > threshold_out
         verbose - whether to print the sequence of inclusions and exclusions
    Returns:
        list of selected features
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                logger.debug('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                logger.debug('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


# 数值型计算PSI
def num_cal_psi(actual_df, expect_df, columns, grp, cuts):
    '''
    连续型变里的PSI计算
    Parameters:
    actual＿df --训练数据
    expect＿df --跨期验证数据
    columns --待计算PSI的变量
    grp --Y变量的变量名称
    cuts --分箱依据
    Returns:
    psi_df[['cut','expect_1','expect_rate',"actual_1',"actusl_rate',"act-exp','ln(act/exp)','sub_psi','psi']]
    '''
    psi_df = pd.DataFrame()
    try:
        actual_total = pd.DataFrame(pd.cut(actual_df[columns], bins=cuts).value_counts()).sort_index()
        actual_total.columns = ['actual_total']
        expect_total = pd.DataFrame(pd.cut(expect_df[columns], bins=cuts).value_counts()).sort_index()
        expect_total.columns = ['expect_total']

        actual_good = pd.DataFrame(
            pd.cut(actual_df[actual_df[grp] == 1][columns], bins=cuts).value_counts()).sort_index()
        actual_good.columns = ['actual_1']
        expect_good = pd.DataFrame(
            pd.cut(expect_df[expect_df[grp] == 1][columns], bins=cuts).value_counts()).sort_index()
        expect_good.columns = ['expect_1']

        #         logger.info(actual_total)
        #     logger.info(expect_total.shape, expect_good.shape, actual_total.shape, actual_good.shape)
        psi_df = pd.concat([expect_total, expect_good, actual_total, actual_good], axis=1).reset_index()
        psi_df.rename(columns={'index': 'cut'}, inplace=True)
        #     logger.info(psi_df)
        psi_df['expect_rate'] = psi_df['expect_1'] / psi_df['expect_total']
        psi_df['actual_rate'] = psi_df['actual_1'] / psi_df['actual_total']
        psi_df['act-exp'] = psi_df['actual_rate'] - psi_df['expect_rate']
        psi_df['ln(act/exp)'] = np.log(psi_df['actual_rate'] / psi_df['expect_rate'])
        psi_df['ln(act/exp)'] = np.where(np.abs(psi_df['ln(act/exp)']) == np.inf, np.nan, psi_df['ln(act/exp)'])
        psi_df['sub_psi'] = psi_df['act-exp'] * psi_df['ln(act/exp)']
        psi_df['psi'] = psi_df['sub_psi'].sum()
    except Exception as err:
        logger.error(err)
        logger.debug(columns)
    #
    return psi_df[
        ['cut', 'expect_1', 'expect_rate', 'actual_1', 'actual_rate', 'act-exp', 'ln(act/exp)', 'sub_psi', 'psi']]


def num_csi_filter(df_rem, df_kq, num_df_list, y_df, df_kq_y, csi_threshold=0.25, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   psi_rem_df=pd.DataFrame()):
    logger.info('PSI筛选-连续型变量：')
    df_kq = df_kq.replace(np.nan, 0)

    sel_num_df_list = []
    for var in num_df_list:
        actual_df = pd.DataFrame(df_rem[var])
        # actual df=sctual df reset index
        actual_df['Y'] = y_df
        # actual_df. drop(party_id',axis=1, inpIace=Irue)
        expect_df = pd.DataFrame(df_kq[var])
        expect_df['Y'] = df_kq_y
        mm = MinMaxScaler(feature_range=(0, 1))
        max_v = actual_df[var].quantile(0.9)
        min_v = actual_df[var].quantile(0.1)
        actual_df[var] = np.where(actual_df[var] >= max_v, max_v, actual_df[var])
        actual_df[var] = np.where(actual_df[var] <= min_v, min_v, actual_df[var])
        actual_df[var] = mm.fit_transform(actual_df[[var]])

        max_v = expect_df[var].quantile(0.9)
        min_v = expect_df[var].quantile(0.1)
        expect_df[var] = np.where(expect_df[var] >= max_v, max_v, expect_df[var])
        expect_df[var] = np.where(expect_df[var] <= min_v, min_v, expect_df[var])
        expect_df[var] = mm.transform(expect_df[[var]])

        psi_df = num_cal_psi(actual_df, expect_df, var, 'Y', bins)
        psi = psi_df['psi'][0].round(4)
        tmp_psi_df = pd.DataFrame([(var, psi)])
        psi_rem_df = pd.concat([psi_rem_df, tmp_psi_df], axis=0)
        if psi <= csi_threshold:
            sel_num_df_list.append(var)
    #     logger.info('%s PSI=%f'%(var,psi))
    logger.info('csi筛选之前数值变量个数为：{}'.format(len(num_df_list)))
    logger.debug('满足csi小于等于{}的变量数：{}'.format(csi_threshold, len(sel_num_df_list)))
    return sel_num_df_list, psi_rem_df


# 类别变量PSI筛选
def obj_cal_psi(actual_df, expect_df, columns, grp):
    '''
    标志型、分类型变里的PSI计算
    Par ameters:
    actusl＿df -训练数据
    expect＿df -跨期验证数据
    columns   -待计算PSI的变里
    grp-Y变量的变量名称
    Returns:
    psi_df[['cut','expect_1','expect_rate','actual_1','actusl_rate',' act-exp',' ln(act/exp)','sub_psi','psi']]
    '''
    actual_total = pd.DataFrame(actual_df[columns].value_counts())
    actual_total.columns = ['actual_total']
    expect_total = pd.DataFrame(expect_df[columns].value_counts())
    expect_total.columns = ['expect_total']
    actual_good = pd.DataFrame(actual_df[actual_df[grp] == 1][columns].value_counts())
    actual_good.columns = ['actual_1']
    expect_good = pd.DataFrame(expect_df[expect_df[grp] == 1][columns].value_counts())
    expect_good.columns = ['expect_1']
    #     logger.info(actual_good)
    psi_df = pd.concat([expect_total, expect_good, actual_total, actual_good], axis=1).reset_index()
    psi_df.rename(columns={'index': 'cut'}, inplace=True)
    psi_df['expect_rate'] = psi_df['expect_1'] / psi_df['expect_total']
    psi_df['actual_rate'] = psi_df['actual_1'] / psi_df['actual_total']
    psi_df['act-exp'] = psi_df['actual_rate'] - psi_df['expect_rate']
    psi_df['ln(act/exp)'] = np.log(psi_df['actual_rate'] / psi_df['expect_rate'])
    psi_df['ln(act/exp)'] = np.where(np.abs(psi_df['ln(act/exp)']) == np.inf, np.nan, psi_df['ln(act/exp)'])
    psi_df['sub_psi'] = psi_df['act-exp'] * psi_df['ln(act/exp)']
    psi_df['psi'] = psi_df['sub_psi'].sum()
    return psi_df[
        ['cut', 'expect_1', 'expect_rate', 'actual_1', 'actual_rate', 'act-exp', 'ln(act/exp)', 'sub_psi', 'psi']]


def cat_csi_filter(cat_features, df, df_kq, y_df, df_kq_y, psi_rem_df, csi_threshold=0.25):
    # 类别变量PSI计算
    sel_cat_df_list = []
    for var in cat_features:
        actual_df = pd.DataFrame(df[var])
        # actual_df= actual_df reset_index()
        actual_df['Y'] = y_df
        # actual_df.drop(party_jd',axis=1, inplaoe=True)
        expect_df = pd.DataFrame(df_kq[var])
        expect_df['Y'] = df_kq_y
        psi_df = obj_cal_psi(actual_df, expect_df, var, 'Y')
        psi = psi_df['psi'][0].round(4)
        tmp_psi_df = pd.DataFrame([(var, psi)])
        psi_rem_df = pd.concat([psi_rem_df, tmp_psi_df], axis=0)
        if psi <= csi_threshold:
            sel_cat_df_list.append(var)
        else:
            logger.debug('%s PSI=%f' % (var, psi))

    logger.info('csi筛选之前类别变量个数为：{}'.format(len(sel_cat_df_list)))
    logger.debug('满足csi小于等于{}的变量数：{}'.format(csi_threshold, len(sel_cat_df_list)))
    return sel_cat_df_list, psi_rem_df


# 计算变量PSI
def csi(actual, expect, val, q):
    var_cat = pd.qcut(actual[val].values, q=q, duplicates='drop')
    var_cat_cnt = var_cat.value_counts() + 0.0001
    var_cnt_proportion = var_cat_cnt / sum(var_cat_cnt)

    expect_pred_cnt = pd.cut(expect[val].values, bins=var_cat.categories).value_counts()

    expect_pred_cnt_1 = expect_pred_cnt + 0.0001
    expect_pred_proportion = expect_pred_cnt_1 / sum(expect_pred_cnt_1)

    return sum((var_cnt_proportion - expect_pred_proportion) * np.log(var_cnt_proportion / expect_pred_proportion))


def var_csi(actual, expect, var_name, q):
    result = []
    for i in var_name:
        # rint(i)
        result.append(csi(actual, expect, i, q=q))
    return pd.DataFrame(result, index=var_name, columns=['csi'])


def show_feature_importance(best_model,cols, data_dict_path: str, left_on_name: str='index', right_on_name='字段英文名',
                       sheet_name='Sheet1',var_zh_name='字段中文名') -> pd.DataFrame:
    """
    特征重要性
    :param best_model:
    :param data_dict_path:
    :param left_on_name:
    :param right_on_name:
    :param sheet_name:
    :param var_zh_name:
    :return:
    """
    #读取数据字典
    data_map = {}
    if data_dict_path is not None:
        data_dict = pd.read_excel(data_dict_path, sheet_name=sheet_name)
        data_dict[right_on_name] = data_dict[right_on_name].str.lower()
        dat_dict_unique = data_dict.drop_duplicates(subset=right_on_name)
        data_dict_map_df = dat_dict_unique.copy()
        data_map = data_dict_map_df.set_index(right_on_name)[var_zh_name].to_dict()

    # feature_importance_lg = best_model.feature_importance(importance_type='split')
    feature_importance_lg = best_model.feature_importances_
    feature_imp = pd.Series(feature_importance_lg, index=[c.replace('woe_', '') for c in cols])
    if data_dict_path is not None:
        feature_imp = pd.DataFrame(feature_imp, columns=['feature_importance']).reset_index().merge(data_dict, how='left',
                                                                                                left_on=left_on_name,
                                                                                                right_on=right_on_name)
        feature_imp_plt = feature_imp[['feature_importance', right_on_name, var_zh_name]].copy()
    else:
        feature_imp = pd.DataFrame(feature_imp, columns=['feature_importance']).reset_index()
        feature_imp_plt = feature_imp[['feature_importance', left_on_name]].copy()
        var_zh_name = left_on_name

    feature_imp_plt.sort_values(by='feature_importance', ascending=True, inplace=True)
    feature_imp_plt.dropna(axis=0, inplace=True)
    feature_imp_plt_top20 = feature_imp_plt.head(20)

    feature_imp_plt_sort = feature_imp_plt.copy()
    feature_imp_plt_sort.sort_values(by='feature_importance',ascending=False,inplace=True)

    # zh_columns = feature_imp[var_zh_name].values
    # feature_imp.dropna(axis = 0,inplace = True)
    # plt.figure()
    # plt.title("Feature Importance")
    # plt.barh(feature_imp_plt_top20[var_zh_name], feature_imp_plt_top20['feature_importance'])
    # plt.show()

    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_imp_plt_top20, x='feature_importance', y=var_zh_name)
    plt.title('20 TOP feature importance')
    plt.show()


    # feature_imp.sort_values(by='feature_importance', ascending=False, inplace=True)
    return feature_imp_plt_sort,data_map

def feature_importance1(best_model, cols):
    # 特征重要性
    # 读取数据字典
    data_dict = pd.read_excel("data/建模宽表0807.xlsx")
    plt.figure(figsize=(12, 15))

    feature_imp = pd.Series(best_model.feature_importances_, index=[c.replace('_woe', '') for c in cols])
    # logger.info(feature_imp)
    # feature_imp = pd.DataFrame(feature_imp, columns=['feature_importance']).merge(data_dict,how = 'left',
    #                                                                               left_on = '变量中文名称',right_on = '变量名称')
    feature_imp = pd.DataFrame(feature_imp, columns=['feature_importance']).reset_index().merge(data_dict, how='left',
                                                                                                left_on='index',
                                                                                                right_on='变量名称')
    feature_imp_plt = feature_imp[['feature_importance', '变量名称']].copy()
    feature_imp_plt.sort_values(by='feature_importance', ascending=True, inplace=True)
    feature_imp_plt.dropna(axis=0, inplace=True)

    zh_columns = feature_imp['变量中文名称'].values
    feature_imp.sort_values(by='feature_importance', ascending=False, inplace=True)
    # feature_imp.dropna(axis = 0,inplace = True)
    plt.title("Feature Importance")
    plt.barh(feature_imp_plt['变量名称'], feature_imp_plt['feature_importance'])
    plt.show()
    feature_imp.sort_values(by='feature_importance', ascending=False, inplace=True)
    return feature_imp


def forest_selection(df_train_woe, labels, missing_threshold=0.99, correlation_threshold=0.9,
                     cumutative_importance=0.99, max_depth=3):
    '''
    LightGBM选择特征
    ：param df_train_woe:
    ：param labels:
    ：return:
    '''

    fs = FeatureSelector(data=df_train_woe, labels=labels)
    fs.identify_all(selection_params={'missing_threshold': missing_threshold,
                                      'eval_metric': 'binary_error',
                                      'correlation_threshold': correlation_threshold,
                                      'task': 'classification',
                                      'cumulative_importance': cumutative_importance,
                                      'max_depth': max_depth
                                      })
    df_train_woe = fs.remove(methods='all',keep_one_hot=False)
    return df_train_woe


def plot_corr(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap='RdBu', annot=True, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

def lr_scored_card(lr_model,model_dir,model_features,Basepoints=600,PDO=50,odds0=None):
    '''
    制作评分卡
    :param lr_model:
    :param model_dir:
    :param model_features:
    :param Basepoints:
    :param PDO:
    :param odds0:
    :return:
    '''

    bins = load_sc_bins(model_dir)
    logger.debug('ODDS:{}'.format(odds0) )
    logger.debug('A/B系数：{}'.format(ab(Basepoints, odds0, PDO)) )
    logger.debug('模型特征数：{}'.format(model_features))
    df_card = scorecard(bins, lr_model, model_features, points0=Basepoints, odds0=odds0, pdo=PDO,
                        basepoints_eq0=False)
    logger.info('score card output.....')
    logger.debug(df_card.head())
    df_card.to_csv(os.path.join(model_dir, 'score_card_report.csv'), encoding='utf8',
                            index=False)
    return df_card

# odds坏好比：1.684  PDO:20  P基准分：600
def ab(Basepoints, odds, PDO):
    a = np.array([[1, -math.log(odds)], [1, -math.log(odds * 2)]])
    b = np.array([Basepoints, Basepoints - PDO])
    x = solve(a, b)
    return {'A': x[0], 'B': x[1]}


def scorecard(bins, model, xcolumns, points0=500, odds0=2, pdo=50, basepoints_eq0=False):
    '''
    Creating a Scorecard
    ------
    `scorecard` creates a scorecard based on the results from `woebin`
    and LogisticRegression of sklearn.linear_model

    Params
    ------
    bins: Binning information generated from `woebin` function.
    model: A LogisticRegression model object.
    xcolumns:selected variables contain sub of 'woe'
    points0: Target points, default 500.
    odds0: Target odds, default 1/19. Odds = p/(1-p).
    pdo: Points to Double the Odds, default 50.
    basepoints_eq0: Logical, default is FALSE. If it is TRUE, the
      basepoints will equally distribute to each variable.

    Returns
    ------
    DataFrame
        scorecard dataframe

    Examples
    ------
    import scorecardpy as sc

    # load data
    dat = sc.germancredit()

    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")

    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)

    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']

    # logistic regression ------
    import statsmodels.api as sm
    LR= sm.Logit(y, X).fit()

    # scorecard
    # Example I # creat a scorecard
    card = scorecard(bins, lr, X.columns)

    '''
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a = aabb['A']
    b = aabb['B']
    # odds = pred/(1-pred); score = a - b*log(odds)

    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    xs = [i for i in xcolumns]

    #     logger.info(model.coef_[0])

    # coefficients
    #     coef_df = pd.Series(model.coef_[0][1:], index=np.array(xs))\
    #       .loc[lambda x: x != 0]#.reset_index(drop=True)
    coef_df = pd.Series(model.coef_[0], index=np.array(xs)) \
        .loc[lambda x: x != 0]  # .reset_index(drop=True)
    # scorecard
    len_x = len(coef_df)

    basepoints = a - b * model.coef_[0][0]
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable': "basepoints", 'bin': np.nan, 'points': 0}, index=np.arange(1))
        #         logger.info('rrrr')
        #         logger.info(coef_df.index)
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable'] == re.sub('_woe$', '', i), ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-b * x['woe'] * coef_df[i] + basepoints / len_x)) \
                [["variable", "bin", "woe", "points"]]
    else:
        #         logger.info('===================',len_x)
        card['basepoints'] = pd.DataFrame({'variable': "basepoints", 'bin': np.nan, 'points': round(basepoints)},
                                          index=np.arange(1))
        #         logger.info(coef_df.head())
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable'] == re.sub('_woe$', '', i), ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-b * x['woe'] * coef_df[i],5)) \
                [["variable", "bin", "woe", "points"]]

    return pd.concat(card, ignore_index=True)[['variable', 'bin', 'woe', 'points']]


def Score(prod, odds0, Basepoints=600, PDO=20):
    aabb = ab(Basepoints, odds0, PDO)
    A = aabb['A']
    B = aabb['B']
    #     return int(A -B*math.log(prod/(1-prod)))
    return int(A + B * math.log(prod / (1 - prod)))


def prob_plot(score, prob):
    '''
    概率评分卡映射
    '''

    plt.subplots(figsize=(15, 8))
    plt.scatter(prob, score, marker='o', label='prob_to_score')
    plt.legend(loc=0)
    plt.plot(np.ones(score.shape[0]) * 0.5, score, mfc='r')
    plt.xticks(np.arange(0.00, 1.01, 0.10))
    plt.yticks(np.arange(0, 1001, 100))
    plt.xlabel('prob')
    plt.ylabel("score")
    plt.show()

def lgbm_evaluate(X, X_test,y, **params):
    logger.debug(params)
    warnings.simplefilter('ignore')

    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])

    train_df = X
    test_df = X_test

    clf = LGBMClassifier(**params, n_estimators=500, nthread=8)

    folds = KFold(n_splits=3, shuffle=True, random_state=1001)

    test_pred_proba = np.zeros(train_df.shape[0])

    feats = [f for f in train_df.columns]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], y)):
        train_x, train_y = train_df[feats].iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], y.iloc[valid_idx]

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc',
                verbose=False, early_stopping_rounds=100)

        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

    #         del train_x, train_y, valid_x, valid_y
    #         gc.collect()

    return roc_auc_score(y, test_pred_proba)


# 计算PSI
def psi(actual, expect, val, q):
    var_cat = pd.qcut(actual[val].values, q=q, duplicates='drop')
    var_cat_cnt = var_cat.value_counts() + 0.0001
    var_cnt_proportion = var_cat_cnt / sum(var_cat_cnt)
    expect_pred_cnt = pd.cut(expect[val].values, bins=var_cat.categories).value_counts()
    expect_pred_cnt_1 = expect_pred_cnt + 0.0001
    expect_pred_proportion = expect_pred_cnt_1 / sum(expect_pred_cnt_1)
    return sum((var_cnt_proportion - expect_pred_proportion) * np.log(var_cnt_proportion / expect_pred_proportion))


def calculate_psi_histogram(expected, actual, buckets=10):
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min + 1e-7)
        input += min
        return input

    breakpoints = np.linspace(0, 1, buckets + 1)

    # Scale data
    expected_percents = np.histogram(scale_range(expected, 0, 1), breakpoints)[0] / len(expected)
    actual_percents = np.histogram(scale_range(actual, 0, 1), breakpoints)[0] / len(actual)

    # Replace 0% with a small number to avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

    return psi_value



def calculate_psi(expected, actual, buckets=10):
    def get_bins(data, bins):
        # 使用qcut进行等频分箱
        bin_edges = pd.qcut(data, q=bins, duplicates='drop', retbins=True)[1]
        # 计算每个箱的边界
        return bin_edges

    def get_counts(data, bins):
        # 根据边界计算每个箱的样本数
        counts, _ = np.histogram(data, bins)
        return counts

    bin_edges = get_bins(expected, buckets)

    expected_counts = get_counts(expected, bin_edges)
    actual_counts = get_counts(actual, bin_edges)

    # 计算每个箱的比例
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Replace 0% with a small number to avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

    return psi_value



def model_psi(actual_probs, expect_probs, q):
    # Calculate quantiles for actual_probs
    var_cat = pd.qcut(actual_probs, q=q, duplicates='drop')

    # Count occurrences in each bin and adjust counts to avoid zero values
    var_cat_cnt = var_cat.value_counts() + 0.001
    var_cnt_proportion = var_cat_cnt / var_cat_cnt.sum()

    # Bin expected_probs using the same bins as var_cat
    expect_pred_cnt = pd.cut(expect_probs, bins=var_cat.categories).value_counts() + 0.001
    expect_pred_proportion = expect_pred_cnt / expect_pred_cnt.sum()

    psi_values = (var_cnt_proportion - expect_pred_proportion) * np.log(var_cnt_proportion / expect_pred_proportion)

    return psi_values.sum()


import pandas as pd
import pickle
import os
from sklearn.preprocessing import OneHotEncoder


def onehot_encode(df, save_path, flag='train', prefix='n_'):
    df_return = pd.DataFrame()
    if flag == 'train':
        # 训练模式下进行OneHot编码并保存编码器
        enc = OneHotEncoder(dtype='int', handle_unknown='ignore').fit(df)
        # 保存编码模型
        with open(os.path.join(save_path, prefix + 'onehot.pkl'), 'wb') as save_model:
            pickle.dump(enc, save_model)

        # 进行OneHot编码并获取特征名
        df_return = pd.DataFrame(enc.transform(df).toarray(), index=df.index)
        df_return.columns = enc.get_feature_names(df.columns)

        # 保存特征名
        with open(os.path.join(save_path, prefix + 'onehot_features.pkl'), 'wb') as feature_file:
            pickle.dump(df_return.columns, feature_file)
    elif flag == 'inverse_transform':
        # 逆变换模式下加载编码器和特征名
        with open(os.path.join(save_path, prefix + 'onehot.pkl'), 'rb') as read_model:
            enc = pickle.load(read_model)

        with open(os.path.join(save_path, prefix + 'onehot_features.pkl'), 'rb') as feature_file:
            encoded_columns = pickle.load(feature_file)
            encoded_columns = list(encoded_columns)

        new_columns = list(set([col.split('_')[0] for col in encoded_columns]))
        print(new_columns)
        num_cols = [col for col in df.columns if col not in encoded_columns]

        # 确保DataFrame中存在这些列
        if all(col in df.columns for col in encoded_columns):
            df_return_cat = pd.DataFrame(
                enc.inverse_transform(df[encoded_columns].values),
                columns=new_columns
            )
            df_return = pd.concat([df[num_cols], df_return_cat], axis=1)
        else:
            raise KeyError(f"Columns {encoded_columns} are not found in the DataFrame.")
    else:
        with open(os.path.join(save_path, prefix + 'onehot.pkl'), 'rb') as read_model:
            enc = pickle.load(read_model)

        with open(os.path.join(save_path, prefix + 'onehot_features.pkl'), 'rb') as feature_file:
            encoded_columns = pickle.load(feature_file)

        # 对新数据进行转换，确保列的顺序和训练时一致
        transformed_df = pd.DataFrame(enc.transform(df).toarray(), index=df.index)
        transformed_df.columns = encoded_columns

        df_return = transformed_df
    return df_return

def onehot_encode1(df, save_path, flag='train',prefix='n_'):
    df_return = pd.DataFrame()
    if flag == 'train':
        enc = OneHotEncoder(dtype='int', handle_unknown='ignore').fit(df)
        # 保存编码模型
        save_model = open(os.path.join(save_path,prefix+'onehot.pkl'), 'wb')
        pickle.dump(enc, save_model, 0)
        save_model.close()
        df_return = pd.DataFrame(enc.transform(df).toarray(),index=df.index)
        df_return.columns = enc.get_feature_names(df.columns)
    elif flag == 'transform':
        read_model = open(os.path.join(save_path, prefix + 'onehot.pkl'), 'rb')
        enc = pickle.load(read_model)
        read_model.close()

        print(df.columns)
        encoded_columns = enc.get_feature_names(df.columns)

        # 确保 DataFrame 中存在这些列
        if all(col in df.columns for col in encoded_columns):
            # 使用逆变换将 OneHot 编码的列转换回原始类别特征
            df_return = pd.DataFrame(
                enc.inverse_transform(df[encoded_columns].values),
                columns=df.columns)  # 原始列名
        else:
            raise KeyError(f"Columns {encoded_columns} are not found in the DataFrame.")

        # df_return = pd.DataFrame(
        #     enc.inverse_transform(df[enc.get_feature_names()].values),
        #     columns=df.columns)
    else:
        read_model = open(os.path.join(save_path, prefix+'onehot.pkl'), 'rb')
        enc = pickle.load(read_model)
        read_model.close()
        df_return = pd.DataFrame(enc.transform(df).toarray(),index=df.index)
        df_return.columns = enc.get_feature_names(df.columns)
    return df_return


def model_predict(model_fit, X):
    """
    模型预测
    :param model_fit:
    :param X:
    :return:
    """

    y_score = model_fit.predict_proba(X)[:, 1]
    y_pred = model_fit.predict(X)
    df = X.copy()
    df['y_prob'] = y_score
    df['y_pred'] = y_pred
    return y_score, y_pred, df


def plot_pr(recall,precision):
    # 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def model_test_eval(model_fit,
                    X_test,
                    y_test,
                    y_target='label',
                    pos_label=1,
                    cache_dir='.',
                    model_type='lr',
                    flag='test',
                    index_col='cst_id'):
    """

    :param model_fit: 模型
    :param X_test:
    :param y_test:
    :param y_target:目标y字段名
    :param pos_label:
    :param cache_dir:
    :param model_type: 模型类型
    :param flag:
    :return:
    """
    logger.info('model predict of {}...'.format(flag))

    y_score_test,y_pred_test,test_df = model_predict(model_fit,X_test)
    # 检查 test_df 是否有重复索引
    logger.debug('是否有重复索引：'.format(test_df.index.duplicated().any()))
    # 检查 y_test 是否有重复索引
    # logger.warning('是否有重复索引：'.format(y_test.index.duplicated().any()))

    # test_df = test_df[~test_df.index.duplicated(keep='first')]
    # y_test = y_test[~y_test.index.duplicated(keep='first')]

    # 确保 y_test 是一个 Series 对象
    # y_test = pd.Series(y_test)
    # 对齐索引并赋值
    # test_df[y_target] = y_test.reindex(test_df.index)
    test_df[y_target] = y_test
    test_df[index_col] = X_test.index


    #模型评测
    logger.info('Model Eval Of {}...'.format(flag))
    test_eval = evaluation_index(y_test,y_pred_test,y_score_test,thld=False)\
        .rename(columns={'评估值':flag+'_评估值'})
    # logger.info(test_eval.merge(test_eval,how='left',right_index=True,left_index=True).T)
    logger.debug(test_eval.T)
    #ROC曲线
    logger.info('ROC Curve Of {}...'.format(flag))
    sc.perf_eva(y_test,y_score_test,title=flag+'_performance',show_plot=True)

    #PR曲线
    # 计算精确率和召回率
    logger.info('PR Curve Of {}...'.format(flag))
    precision, recall, _ = precision_recall_curve(y_test, y_score_test)
    # 绘制 PR 曲线
    plot_pr(recall, precision)

    #混淆矩阵
    logger.info('Confusion Matrix of {} Data...'.format(flag))
    report_test = classification_report(y_test,y_pred_test)
    logger.info(report_test)

    #Lift曲线
    lift_cure(y_test,y_score_test)

    logger.info('model report of {}...'.format(flag))
    report_df = var_equal_bin(test_df,'y_prob',y_target,10)
    report_df.rename(columns={
        'var':'变量'
        ,'var_bin':'分箱'
        ,'total':'总样本'
        ,'positive':'响应样本'
        ,'negative':'未响应样本'
        ,'PD':'响应率'
        ,'cumtotal':'累计样本率'
        ,'cumbpositive':'累计响应率'
    },inplace=True)
    report_df_filter = report_df[['变量','分箱','总样本','响应样本','未响应样本','响应率','累计样本率','累计响应率','IV','KS','LIFT']]
    report_df_filter.to_csv(os.path.join(cache_dir,model_type+'_'+flag+'_report.csv'),encoding='utf8',index=False)
    logger.debug(report_df_filter.head())

    #模型评分分布
    logger.info('model distribution of {}...'.format(flag))
    prob_distplot(test_df,var_name='y_prob',bins=10)

    #评分卡映射
    # 概率转分数
    Basepoints = 600
    PDO=50
    # odds0 = test_df[y_target].value_counts()[1] / test_df[y_target].value_counts()[0]  # 好坏比
    odds0 = test_df[y_target].value_counts()[0] / test_df[y_target].value_counts()[1]  # 好坏比
    logger.debug(test_df[y_target].value_counts())
    # odds0 = 20  # 好坏比 ,默认违约率假设为5% 95/5 //TODO
    test_df['score'] = test_df['y_prob'].apply(lambda x: Score(x, odds0, Basepoints=Basepoints, PDO=PDO))

    #概率评分映射关系
    logger.info('{} model score...'.format(flag))
    prob_plot(score=test_df['score'],prob=test_df['y_prob'])

    test_df[[y_target,'y_prob','y_pred','score']].sort_values('score',ascending=True)
    test_df[[y_target,'y_prob','y_pred','score']].to_csv(os.path.join(cache_dir,model_type+'_model_probs.csv'),encoding='utf8',index=False)
    # test_df.to_csv(os.path.join(cache_dir,model_type+'_model_probs.csv'),encoding='utf8',index=False)
    return y_score_test,y_pred_test,test_df

def prob_distplot(df,var_name='y_prob',bins=10):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.style.use('ggplot')
    plt.subplots(figsize=(15, 6))
    sns.distplot(df[var_name], bins=bins)
    plt.show()


def model_predict_eval(model_fit,
                        X,
                        X_test,
                        y,
                        y_test,
                        y_target='label',
                        pos_label=1,
                        cache_dir='.',
                        model_type='lr',
                        index_col='cst_id'
                  ):
    """
    模型预测及评估
    :param model_fit:
    :param X:
    :param X_test:
    :param y:
    :param y_test:
    :param y_target:
    :param pos_label:
    :param cache_dir:
    :param model_type:
    :return:
    """

    y_score_train, y_pred_train, train_df = model_test_eval(model_fit,
                                                         X,
                                                         y,
                                                         y_target=y_target,
                                                         pos_label=pos_label,
                                                         cache_dir=cache_dir,
                                                         model_type=model_type,
                                                         flag='train',
                                                         index_col=index_col)

    y_score_test,y_pred_test,test_df = model_test_eval(model_fit,
                    X_test,
                    y_test,
                    y_target=y_target,
                    pos_label=pos_label,
                    cache_dir=cache_dir,
                    model_type=model_type,
                    flag='test',
                    index_col=index_col)

    logger.debug(y_score_train.shape)
    logger.debug(y_score_test.shape)
    # model_psi_test = model_psi(y_score_test,y_test,10)
    model_psi_test = calculate_psi(y_score_train,y_score_test,10)
    logger.info('训练测试模型PSI：{}'.format(model_psi_test))
    return y_score_test,y_pred_test,test_df

def extract_table_names(sql):
    '''
    抽取表名
    :param sql:
    :return:
    '''
    tablename_rule = r'FROM\s+([^\s:,]+)'
    table_names = re.findall(tablename_rule,sql,flags=re.IGNORECASE)
    return table_names[0]


def read_data_by_query(engine, sql,cache_dir, need_reload_data=False, dtype=None, index_col='cst_id'):
    """
    按sql query 获取数据
    :param engine:
    :param sql:
    :param cache_dir:
    :param need_reload_data:
    :param dtype:
    :param index_col:
    :return:
    """
    big_tablename = extract_table_names(sql)
    cust_info_path = os.path.join(cache_dir, '{}.csv'.format(big_tablename))

    if need_reload_data or not os.path.exists(cust_info_path):
        logger.debug(sql)
        if index_col is not None:
            df = pd.read_sql_query(sql, con=engine, index_col=index_col)
            df.to_csv(cust_info_path, encoding='utf-8',index=True)
            logger.info('persist customer info {} to file success.'.format(cust_info_path))
        else:
            df = pd.read_sql_query(sql, con=engine)
            df.to_csv(cust_info_path, encoding='utf_8', index=False)
            logger.info('persist customer info {} to file success.'.format(cust_info_path))
    else:
        if index_col is not None:
            if dtype is None:
                df = pd.read_csv(cust_info_path, index_col=index_col)
            else:
                df = pd.read_csv(cust_info_path, index_col=index_col,dtype=dtype)
        else:
            if dtype is not None:
                df = pd.read_csv(cust_info_path, dtype=dtype)
            else:
                df = pd.read_csv(cust_info_path)

    logger.info('read data from {}.'.format(cust_info_path))
    return df


def read_data(engine, cache_dir, big_tablename, watch_point=None, need_reload_data=False, dtype=None, index_col='cst_id', dt_name='data_dt'):
    """
    读取数据
    :param engine:
    :param cache_dir:
    :param big_tablename:
    :param watch_point:
    :param need_reload_data:
    :param dtype:
    :param index_col:
    :param dt_name:
    :return:
    """
    cust_info_path = os.path.join(cache_dir, '{}_{}.csv'.format(big_tablename, dt_name))
    if need_reload_data or not os.path.exists(cust_info_path):
        if watch_point is not None:
            query = 'select * from {} where {} >= {}'.format(big_tablename, dt_name, watch_point)
        else:
            query = 'select * from {}'.format(big_tablename)
        logger.debug(query)
        if index_col is not None:
            df = pd.read_sql_query(query, con=engine,index_col=index_col).set_index(index_col)
            df.to_csv(cust_info_path, encoding='utf-8', index=True)
        else:
            df = pd.read_sql_query(query, con=engine).set_index(False)
            df.to_csv(cust_info_path, encoding='utf-8', index=False)
    else:
        if index_col is not None:
            if dtype is None:
                df = pd.read_csv(cust_info_path,index_col=index_col)
            else:
                df = pd.read_csv(cust_info_path, index_col=index_col,dtype=dtype)
        else:
            if dtype is None:
                df = pd.read_csv(cust_info_path)
            else:
                df = pd.read_csv(cust_info_path, dtype=dtype)

    return df


def compute_similarity(data, white_list, threshold=0.3, back_filter=True):
    """
    计算白名单和其他客户相似度
    Args:
        data - 客户特征
        white_list - 白名单列表
        threshold - 相似度阈值
        back_filter - 是否进行反向筛选
    """

    sim_matrix = cosine_similarity(data, white_list)
    row_min_sim = np.min(sim_matrix, axis=1)

    if back_filter:
        return row_min_sim <= threshold
    else:
        return row_min_sim > threshold


def get_cst_feature_by_label(df, cst_id_to_row_index, x_n_reduced,label='1', y_target='label_cd'):
    """
    通过标签获取客户特征
    :param df:
    :param cst_id_to_row_index:
    :param x_n_reduced:
    :param label:
    :param y_target:
    :return:
    """
    filtered_df = df[df[y_target] == label]
    filtered_cst_ids = filtered_df.index
    filtered_row_indices = [cst_id_to_row_index[cst_id] for cst_id in filtered_cst_ids if cst_id in cst_id_to_row_index]
    cst_list_data = x_n_reduced[filtered_row_indices]
    return cst_list_data, filtered_df


def catch_sim_or_notsim_cst(df_y,x_n_reduced, threshold=0.3, y_target='label', back_filter=True, positive_label='1',
                            unlablel_label='-1'):
    """
    获取相似或者不相似客户
    :param df_y:标签DataFrame
    :param x_n_reduced:特征
    :param threshold:相似度阈值
    :param y_target:
    :param back_filter:是否筛选黑名单，若为False则是寻找相似客户
    :param positive_label:
    :param unlablel_label:
    :return: df_y 标签
    """

    cst_id_to_row_index = {cst_id: inx for inx, cst_id in enumerate(df_y.index)}

    white_list,white_filtered_df = get_cst_feature_by_label(df_y,cst_id_to_row_index,x_n_reduced,label=positive_label)
    unlabel_list,unlabel_filtered_df = get_cst_feature_by_label(df_y,cst_id_to_row_index,x_n_reduced,label=unlablel_label)
    logger.info('正样本集合数目：{}'.format(len(white_list)) )
    logger.info('无标签样本集合数目：{}'.format(len(unlabel_list)) )


    all_similar_row_indices = []
    batch_size = 10000
    results = []
    data = unlabel_list

    for i in tqdm(range(0,len(data),batch_size)):
        batch = data[i:i+batch_size]
        similar_items = compute_similarity(batch,white_list,threshold=threshold,back_filter=back_filter)
        similar_row_indices = np.where(similar_items)[0] + i
        logger.debug('length of similar row indices:{}'.format(len(similar_row_indices)))
        all_similar_row_indices.extend(similar_row_indices)

    #映射回cst_id
    row_index_to_cst_id = {inx:cst_id for inx,cst_id in enumerate(unlabel_filtered_df.index)}
    logger.debug('length of all similar row indices:{}'.format(len(all_similar_row_indices)))
    similary_cst_ids = [row_index_to_cst_id[idx] for idx in all_similar_row_indices if idx in row_index_to_cst_id]
    if back_filter:
        logger.debug('number of negative customer:{}'.format(len(similary_cst_ids)))
    else:
        logger.debug('number of similarity customer:{}'.format(len(similary_cst_ids)))
    df_y.loc[similary_cst_ids,y_target] = '0'
    logger.debug('distribution of positive and negatice:')
    logger.debug(df_y[y_target].value_counts())
    return df_y

def plt_reduce_dim(X_tsne,x_label='t-SNE Component 1',y_label='t-SNE Component 2',title='t-SNE Visualization of TruncatedSVD data'):
    """
    画降维后的特征
    :param X_tsne:
    :param x_label:
    :param y_label:
    :param title:
    :return:
    """
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0],X_tsne[:,1],alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def merge_dict(cache_dir,file_name='企业级用户客户标签_数据字典.xlsx',
               sheet_name='标签访问',var_zh_name='字段中文名',var_en_name='字段英文名',
               other_file_name='手机银行加工字段.xlsx'):
    """
    :param cache_dir:
    :return:
    """

    data_dict_df = pd.read_excel(os.path.join(cache_dir,file_name),sheet_name=sheet_name)
    dict_map_df = data_dict_df[[var_en_name,var_zh_name]]
    logger.debug(dict_map_df.shape)
    mobile_bank_dict_df = pd.read_excel(os.path.join(cache_dir,other_file_name))
    mobile_bank_dict_df.columns = [var_en_name,var_zh_name]
    logger.debug(mobile_bank_dict_df.shape)
    dict_all_df = pd.concat([dict_map_df,mobile_bank_dict_df],axis=0)
    dict_all_df[var_en_name] = dict_all_df[var_en_name] .str.strip()
    dict_all_df[var_zh_name] = dict_all_df[var_zh_name] .str.strip()
    logger.debug(dict_all_df.shape)
    data_dict_path = os.path.join(cache_dir,'dict_all.xlsx')
    dict_all_df.to_excel(data_dict_path)
    logger.debug(dict_all_df.shape)
    return data_dict_path



def plot_trend(x,y,label='PV',title='每日点击次数趋势图(PV)',xlabel='日期',ylabel='点击次数'):
    """
    绘制趋势图
    :param x:横坐标数据
    :param y:纵坐标数据
    :param label:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    """

    plt.figure(figsize=(10,6))
    plt.plot(x,y,label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()


def load_all_model_features(model_dir):
    """
    加载模型特征字段
    :return:
    """
    model_features = []
    with open(os.path.join(model_dir,'model_features.pickle'),'rb') as f:
        model_features = pickle.load(f)
        logger.debug('load mode success.')
    return model_features

def load_sc_bins(persist_dir):
    '''
    加载分箱规则
    :param persist_dir:
    :return:
    '''
    bins  = None
    bin_path = os.path.join(persist_dir,'bins.pkl')
    if os.path.exists(bin_path):
        with open(bin_path, 'rb') as readfile:
            bins = pickle.load(readfile)
    else:
        logger.debug('sc bin not exists!')

    return bins

def cst_stat_plot(df, dt_name, title='每日点击次数趋势图', ylabel='点击次数', xlabel='日期', stats_field='view_cnt', cst_field=None):
    """
    每日触达以及点击数据统计
    :param df:
    :param dt_name:
    :param title:
    :param ylabel:
    :param xlabel:
    :param stats_field:
    :param cst_field:
    :return:
    """
    df[dt_name] = pd.to_datetime(df[dt_name])
    if stats_field is None:
        print('total PV:',len(df))
        pv_clicks_per_day = df.groupby(dt_name).size()
    else:
        print('total PV:', df[stats_field].sum())
        pv_clicks_per_day = df.groupby(dt_name)[stats_field].agg(['sum'])

    logger.debug(pv_clicks_per_day.head())

    plot_trend(x=pv_clicks_per_day.index, y=pv_clicks_per_day.values, label='PV', title=title + ' (PV)', xlabel=xlabel,
               ylabel=ylabel)

    if cst_field is None:
        print('total UV:',len(df))
        uv_clicks_per_day = df.groupby(dt_name).nunique()
    else:
        print('total UV', df[cst_field].nunique())
        uv_clicks_per_day = df.groupby(dt_name)[cst_field].nunique()

    logger.debug(uv_clicks_per_day.head(20))

    plot_trend(uv_clicks_per_day.index, uv_clicks_per_day.values, label='UV', title=title + ' (UV)', xlabel=xlabel,
               ylabel=ylabel)


def plot_histogram(df,col, col_name, label_bad='响应样本',
                   label_good='未响应样本', label='响应率%',
                   rotation=45):
    """

    :param var_woe: 
    :param col: 
    :param col_name: 
    :param label_bad: 
    :param label_good: 
    :param label: 
    :param rotation: 
    :return: 
    """

    x_list = df[df.variable==col] ['bin'] #X轴

    y_value = df[df.variable==col]['count'] #Y1轴数值
    y11_value = df[df.variable==col]['good'] #Y1轴数值
    y12_value = df[df.variable==col]['bad'] #Y1轴数值

    y1 = y_value.max()*(1+0.30) #Y1轴取值上限
    y2_value = df[df.variable==col]['badprob'] #Y2轴数值
    y2 = y2_value.max() * (1 + 0.20)  # Y2轴取值上限
    iv = df[df.variable == col]['total_iv'].values[0].round(3)

    # value of bad
    y1_value = list(y11_value.sort_values(ascending=False))

    # value of good
    y2_value = list(y2_value.sort_values(ascending=False))

    # total value
    total_y_value = [y1 + y2 for y1, y2 in zip(y1_value, y2_value)]

    def to_percent(temp,position):
        return '%2.1f'%(100*temp)+"%"

    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent)) #设置纵轴百分比
    ax = plt.subplot(111,facecolor='lightgrey')
    plt.ylim(0,y1)
    #柱状图并列排布
    x = np.arange(len(x_list))
    rects1= ax.bar(x-0.1,y11_value, alpha=0.5, width=0.3, color='cornflowerblue', label=label_good)
    # height effect
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    rects2 = ax.bar(x + 0.1, y12_value, alpha=0.5, width=0.2, color='red', label=label_bad)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(x_list)
    ax.legend(loc=2,fontsize=15,prop='Source Han Sans CN')
    plt.xticks(size=15,color='black',rotation=rotation,FontProperties='Source Han Sans CN')
    ax = ax.twinx()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))  # 设置纵轴百分比
    plt.ylim(0,y2)
    ax.plot(x_list,y2_value,color='darkblue',label=label)
    ax.legend(loc=1,prop='Source Han Sans CN')
    ax.set_title(label=col_name+':IV='+str(iv),fontsize=20)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.yticks(size=13,color='black')

    for x,y in zip(x_list,y2_value):
        ax.text(x,y,'%.2f%%'%(y*100),horizontalalignment='center',verticalalignment='bottom',fontdict={'fontsize':12})
    plt.show()

def output_list_by_model(path, min_lift=4):
    """
    输出清单
    :param path:
    :param min_lift:
    :return:
    """
    report_df = pd.read_csv(path, encoding='utf-8')
    report_df.columns = ['变量', '分箱', '总样本', '响应样本', '未响应样本', '响应率','累计样本率','累计响应率','IV', 'KS', 'LIFT']
    logger.info('模型报告：')
    print(report_df.head(20))

    report_df['分箱'] = report_df['分箱'].apply(lambda x: float(x.split(',')[0][1:]))
    max_ks_index = report_df['KS'].sort_values(ascending=False).index[0]
    max_ks_cutoff = report_df.loc[max_ks_index,'分箱']

    bin_max_ks_val = report_df['LIFT'].sort_values(ascending=False)[0]

    max_lift_series = report_df['LIFT'].diff().abs().sort_values(ascending=False)
    max_lift_val = max_lift_series.values[0]
    optimize_lift_val = max(max_lift_val,min_lift)

    if min_lift > bin_max_ks_val:
        logger.debug('设置lift超过最大lift，改为lift阈值自动降级为最大的一半！')
        optimize_lift_val = bin_max_ks_val/2

    max_lift_index = max_lift_series.index[0]
    opti_lift_index = report_df[report_df['LIFT']>= optimize_lift_val]['LIFT'].sort_values(ascending=True).index[0]

    optimize_lift_cutoff = report_df.loc[max_lift_index]['分箱']

    cutoff_high_response, cutoff_medium_response\
        = max(max_ks_cutoff,optimize_lift_cutoff),min(max_ks_cutoff,optimize_lift_cutoff)

    logger.info(f'最大KS对应概率分数cutoff：{max_ks_cutoff}')
    logger.info(f'最优LIFT对应概率分数cutoff：{optimize_lift_cutoff}')
    logger.info(f'高响应度阈值:{cutoff_high_response}')
    logger.info(f'中响应度阈值:{cutoff_medium_response}')
    return cutoff_high_response, cutoff_medium_response, max_ks_cutoff,optimize_lift_cutoff


def gmm_train(model_dir, X, gmm_need_retrain=True,model_size=7):
    gmms_model_list_path = os.path.join(model_dir,'gmms_list_model.pkl')
    gmm_model_path = os.path.join(model_dir,'gmm_model.pkl')
    if gmm_need_retrain:
        models = []
        progress_guassian_bar = trange(2,model_size+1,disable=False)
        min_bic = np.inf
        best_k = 2
        best_inx = 1
        cnt = 0
        for n in progress_guassian_bar:
            gmm_model = GaussianMixture(n_components=n,covariance_type='full',random_state=2023).fit(X)
            models.append(gmm_model)
            bic = gmm_model.bic(X)
            if bic <= min_bic:
                min_bic=bic
                best_k=n
                best_inx = cnt
            cnt += 1
        print('最小的BIC为{}，最好的K为：{}'.format(min_bic,best_k))
        with open(gmms_model_list_path,'wb') as savefile:
            pickle.dump(models ,savefile ,0)

        gmm = models[best_inx]

        with open(gmm_model_path,'wb') as savefile:
            pickle.dump(gmm ,savefile ,0)

    elif os.path.exists(gmms_model_list_path):
        with open(gmms_model_list_path,'rb') as readfile:
            models = pickle.load(readfile)
        with open(gmm_model_path,'rb') as readfile:
            gmm=pickle.load(readfile)
    else:
        logger.debug('请重新训练GMM模型')
    return models,gmm

def jaccard_similarity(set1,set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection/union
    return similarity

def vif_check(X):

    vif_val = pd.DataFrame({"Col": X.columns})
    vif_val["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
    ''''
    	Col	        VIF
0	battery_power	8.076717
1	blue            1.981927
    '''
    return vif_val

def con_plot(var,df):
    fig, ax = plt.subplots(int(np.ceil(len(df.columns) / 3)), 3, figsize=(16, 16))
    ax = ax.flatten()
    i = 0
    for col in var:
        skew = df[col].skew()
        sns.distplot(df[col], fit=stats.norm, ax=ax[i])
        ax[i].set_title("Variable %s skew : %.4f" % (col, skew))
        i += 1
    plt.tight_layout()
    plt.show()


def cat_plot(var,df):
    fig, ax = plt.subplots(int(np.ceil(len(var) / 3)), 3, figsize=(16, 8))
    ax = ax.flatten()
    i = 0
    for col in var:
        sns.countplot(df[col], ax=ax[i])
        ax[i].set_title("devices in each category for %s" % (col))
        i += 1
    plt.tight_layout()
    plt.show()

def pairplot(df,target='target'):
    sns.pairplot(df, hue=target, diag_kind='hist')
    plt.show()




def circle_distrib(df_dis,labels=['负样本', '正样本']):
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    plt.pie(df_dis, labels=labels, colors=['#a8d8ea', '#aa96da'], autopct='%1.1f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

def plot_distribuion(df,y_target='target',plot_type='all'):
    _, _, _, cat_features, num_features = feature_classify(df, target=y_target)
    if plot_type !='all':
        df[cat_features].hist(figsize=(10, 4))
        # _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        #
        # sns.countplot(x="Churn", data=df, ax=axes[0])
        # sns.countplot(x="Customer service calls", data=df, ax=axes[1]);
        for col in cat_features:
            sns.countplot(x=col, data=df)
        df[num_features].plot(
            kind="density", subplots=True, layout=(1, 2), sharex=False, figsize=(10, 4)
        )
    else:
        sns.distplot(df[cat_features+num_features])


def tgi_wordcloud(feature_tgi_df):
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 将DataFrame转换为字典
    data = feature_tgi_df['TGI'].to_dict()
    cleaned_freq_dict = {k: (0 if np.isnan(v) else v) for k, v in data.items()}

    # font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'

    # 以下为上述代码中的 WordCloud 部分
    # 创建WordCloud对象
    # wordcloud = WordCloud(font_path=font_path,width=800, height=400, background_color='white')
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    # 生成词云
    wordcloud.generate_from_frequencies(cleaned_freq_dict)
    # 显示词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 关闭坐标轴
    # 保存图片到文件
    # plt.savefig('./images/profile/用户画像_客群三.png', dpi=600, format='png')  # 设置 dpi 为超清
    # plt.savefig('./images/profile/用户画像_客群三.svg', dpi=600, format='svg')  # 设置 dpi 为超清
    plt.show()


def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r"'", "", text)
    text = re.sub('\s+', ' ', text).strip()  # Remove and double spaces
    text = re.sub(r'&amp;?', r'and', text)  # replace & -> and
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)  # Remove URLs
    # remove some puncts (except . ! # ?)
    text = re.sub(r'[:"$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'EMOJI', text)
    return text

def generate_short_uuid():
    # characters = string.ascii_letters + string.digits
    str_characters = string.ascii_letters
    digit_str = str(string.digits)
    short_uuid = ''.join(random.choice(str_characters) for _ in range(4))
    short_uuid = short_uuid+'_'
    short_uuid = short_uuid+''.join(random.choice(digit_str) for _ in range(4))
    return short_uuid