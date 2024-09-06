

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np

from ..feature_encoder.feature_encoder import FeatureEncoder,OnehotEncoder,WoeEncoder
from .my_logging import logger
from .utils import feature_classify



def under_sample_en(X, y, y_target, desired_ratio):
    '''
    根据设定的正负样本比例对DataFrame进行降采样。

    :param X: 特征数据集的DataFrame
    :param y: 目标变量的Series
    :param y_target: 目标变量的列名
    :param desired_ratio: 设定的正负样本比例 (正样本 / 负样本)
    :return: 降采样后的X和y
    '''

    # 合并 X 和 y 为一个 DataFrame 以便操作
    df = pd.concat([X, y], axis=1)

    # 计算正负样本的数量
    n_class_0 = df[df[y_target] == 0].shape[0]  # 多数类样本
    n_class_1 = df[df[y_target] == 1].shape[0]  # 少数类样本

    # 计算当前样本比例
    actual_ratio = n_class_1 / n_class_0

    # 检测是否需要降采样
    if actual_ratio < desired_ratio:
        # 负样本是多数类，需要降采样
        n_class_0_new = int(n_class_1 / desired_ratio)
        df_majority_downsampled = df[df[y_target] == 0].sample(n_class_0_new, random_state=42)
        df_minority = df[df[y_target] == 1]

        # 合并回一个新的平衡DataFrame
        df_balanced = pd.concat([df_minority, df_majority_downsampled])

        # 洗牌DataFrame
        df_balanced = df_balanced.sample(frac=1, random_state=42)
    else:
        # 当前比例已经满足或低于设定的比例，不需要降采样
        logger.warning('当前比例已经满足或低于设定的比例，不需要降采样')
        df_balanced = df

    # 分离 X 和 y
    X_balanced = df_balanced.drop(columns=[y_target])
    y_balanced = df_balanced[y_target]

    return X_balanced, y_balanced
def under_sample(df, y_target, desired_ratio):
    '''
    根据设定的正负样本比例对DataFrame进行降采样。

    :param df: 原始数据集的DataFrame
    :param y_target: 目标变量的列名
    :param desired_ratio: 设定的正负样本比例 (正样本 / 负样本)
    :return: 降采样后的DataFrame，如果整体样本比例低于设定的比例，则按实际比例返回。
    '''

    # 计算正负样本的数量
    n_class_0 = df[df[y_target] == 0].shape[0]  # 多数类样本
    n_class_1 = df[df[y_target] == 1].shape[0]  # 少数类样本

    # 计算当前样本比例
    actual_ratio = n_class_1 / n_class_0

    # 检测是否需要降采样
    if actual_ratio < desired_ratio:
        # 负样本是多数类，需要降采样
        n_class_0_new = int(n_class_1 / desired_ratio)
        df_majority_downsampled = df[df[y_target] == 0].sample(n_class_0_new, random_state=42)
        df_minority = df[df[y_target] == 1]
    else:
        # 当前比例已经满足或低于设定的比例，不需要降采样
        return df

    # 合并回一个新的平衡DataFrame
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # 可以选择洗牌DataFrame
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    return df_balanced

def under_sample1(df,y_target,desired_ratio):
    '''
    降采样
    :param df:
    :param y_target:
    :param desired_ratio:
    :return:
    '''
    # 计算正负样本的数量
    n_class_0 = df[df[y_target] == 0].shape[0]  # 正样本
    n_class_1 = df[df[y_target] == 1].shape[0]  # 负样本

    # 检测多数类并计算降采样数量
    if n_class_0 > n_class_1 * desired_ratio:
        # 正样本是多数类
        n_class_0_new = int(n_class_1 * desired_ratio)
        df_majority_downsampled = df[df[y_target] == 0].sample(n_class_0_new, random_state=42)
        df_minority = df[df[y_target] == 1]
    else:
        # 负样本是多数类
        n_class_1_new = int(n_class_0 / desired_ratio)
        df_majority_downsampled = df[df[y_target] == 1].sample(n_class_1_new, random_state=42)
        df_minority = df[df[y_target] == 0]

    # 合并回一个新的平衡DataFrame
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # 可以选择洗牌DataFrame
    # df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42)
    return  df_balanced




def knn_smote_oversample(df,k_neighbors=5, y_target='target'):
    '''
    过采样
    :param df:
    :param k_neighbors:
    :param y_target:
    :return:
    '''
    from imblearn.over_sampling import SMOTE
    target_column = y_target  # 目标变量的列名

    # 分离特征和目标变量
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 初始化SMOTE对象
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)  # 可以调整k_neighbors的值

    # 应用SMOTE过采样
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 将过采样后的数据转换回DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    return df_resampled

    # 现在 df_resampled 是过采样后的数据集

# from sklearn.datasets import make_classification
def smote_oversample(df, y_target, positive_class_ratio):
    """
    根据指定的正样本比例对DataFrame进行过采样。

    :param df: 原始数据集的DataFrame
    :param target_column: 目标变量的列名
    :param positive_class_ratio: 目标正样本比例
    :return: 过采样后的DataFrame
    """
    # 分割特征和目标变量
    X = df.drop(y_target, axis=1)
    y = df[y_target]

    # 计算当前正负样本的数量
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)

    # 计算目标正样本数量
    target_pos = int(n_neg * positive_class_ratio / (1 - positive_class_ratio))

    # 设置过采样策略
    sampling_strategy = {1: target_pos}

    # 使用SMOTE进行过采样
    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 将过采样后的数据重新组装成DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[y_target] = y_resampled

    return df_resampled





def oversample_dataframe(df, y_target, positive_class_ratio,data_dir):
    """
    根据指定的正样本比例对DataFrame进行过采样，并处理类别特征。

    :param df: 原始数据集的DataFrame
    :param y_target: 目标变量的列名
    :param positive_class_ratio: 目标正样本比例(少数类样本 / 多数类样本的比例)
    :return: 过采样后的DataFrame
    """

    # 识别类别特征和数值特征
    _, _, _, cat_features, num_features = feature_classify(df, target=y_target)
    df_encoded = df.copy()

    # 对类别特征进行OneHot编码
    onehot_encoder = OnehotEncoder(model_dir=data_dir, y_target=y_target)
    df_onehot_encoded  = onehot_encoder.encode(X=df, flag='train',prefix='oversample_')
    # df_onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(df[cat_features]),
    #                                  columns=onehot_encoder.get_feature_names_out(cat_features))

    # 合并编码后的数据
    # df_encoded = pd.concat([df[num_features], df_onehot_encoded, df[y_target]], axis=1)
    df_encoded = pd.concat([df_onehot_encoded, df[y_target]], axis=1)

    # 分割特征和目标变量
    X = df_encoded.drop(y_target, axis=1)
    y = df_encoded[y_target]

    # 计算类别的数量
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)

    # 计算实际正样本比例
    actual_positive_ratio = n_pos / n_neg
    logger.debug('actual positive(1) ratio:{}'.format(round(actual_positive_ratio,2)))
    logger.debug('expect positive(1) ratio:{}'.format(round(positive_class_ratio,2)))

    # 检查设定的正样本比例是否合理
    if positive_class_ratio <= actual_positive_ratio:
        logger.warning(
            'The desired positive class ratio is less than or equal to the actual ratio. No oversampling needed.')
        return df

    # 计算过采样的目标数量
    target_pos = int(n_neg * positive_class_ratio)

    # 使用SMOTE进行过采样
    smote = SMOTE(sampling_strategy=positive_class_ratio)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 将过采样后的数据重新组装成DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    df_onehot_inverse = onehot_encoder.encode(X=df_resampled, flag='inverse_transform', prefix='oversample_',inverse_cat_features=cat_features)
    df_onehot_inverse[y_target] = y_resampled
    df_onehot_inverse.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 使用列的平均值替换 NaN 值
    df_onehot_inverse.fillna(df_onehot_inverse.mean(), inplace=True)
    return df_onehot_inverse


from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import logging

# 初始化日志记录器
logger = logging.getLogger(__name__)

def oversample_x_y(X, y, positive_class_ratio, data_dir):
    """
    根据指定的正样本比例对DataFrame进行过采样，并处理类别特征。

    :param X: 特征数据集的DataFrame
    :param y: 目标变量的DataFrame
    :param positive_class_ratio: 目标正样本比例(少数类样本 / 多数类样本的比例)
    :param data_dir: OneHot编码器模型存储目录
    :return: 过采样后的特征和目标变量DataFrame
    """

    # 识别类别特征和数值特征
    _, _, _, cat_features, num_features = feature_classify(X, target=y.name)
    X_encoded = X.copy()

    # 对类别特征进行OneHot编码
    onehot_encoder = OnehotEncoder(model_dir=data_dir, y_target=y.name)
    X_onehot_encoded = onehot_encoder.encode(X=X, flag='train', prefix='oversample_')

    # 合并编码后的数据
    X_encoded = pd.concat([X_onehot_encoded], axis=1)

    # 计算类别的数量
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)

    # 计算实际正样本比例
    actual_positive_ratio = n_pos / n_neg
    logger.debug(f'Actual positive(1) ratio: {round(actual_positive_ratio, 2)}')
    logger.debug(f'Expected positive(1) ratio: {round(positive_class_ratio, 2)}')

    # 检查设定的正样本比例是否合理
    if positive_class_ratio <= actual_positive_ratio:
        logger.warning('The desired positive class ratio is less than or equal to the actual ratio. No oversampling needed.')
        return X, y

    # 计算过采样的目标数量
    target_pos = int(n_neg * positive_class_ratio)

    # 使用SMOTE进行过采样
    smote = SMOTE(sampling_strategy=positive_class_ratio)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # 将过采样后的数据重新组装成DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_encoded.columns)

    # 恢复OneHot编码前的类别特征
    X_inverse_encoded = onehot_encoder.encode(X=X_resampled_df, flag='inverse_transform', prefix='oversample_', inverse_cat_features=cat_features)

    # 替换目标变量
    X_inverse_encoded[y.name] = y_resampled
    X_inverse_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 使用列的平均值替换 NaN 值
    X_inverse_encoded.fillna(X_inverse_encoded.mean(), inplace=True)

    return X_inverse_encoded.drop(columns=[y.name]), X_inverse_encoded[y.name]
def oversample_encoded(X, y, positive_class_ratio,encoder_type='onehot'):
    """
    根据指定的正样本比例对DataFrame进行过采样，并处理类别特征。

    :param df: 原始数据集的DataFrame
    :param y_target: 目标变量的列名
    :param positive_class_ratio: 目标正样本比例(少数类样本 / 多数类样本的比例)
    :return: 过采样后的DataFrame
    """
    # 计算类别的数量
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)

    # 计算实际正样本比例
    actual_positive_ratio = n_pos / n_neg
    logger.debug('actual positive(1) ratio:{}'.format(round(actual_positive_ratio,2)))
    logger.debug('expect positive(1) ratio:{}'.format(round(positive_class_ratio,2)))

    # 检查设定的正样本比例是否合理
    if positive_class_ratio <= actual_positive_ratio:
        logger.warning(
            'The desired positive class ratio is less than or equal to the actual ratio. No oversampling needed.')
        return X,y

    # 计算过采样的目标数量
    target_pos = int(n_neg * positive_class_ratio)

    # 使用SMOTE进行过采样
    smote = SMOTE(sampling_strategy=positive_class_ratio)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return  X_resampled, y_resampled



def mixed_sampling(df,  oversample_strategy=0.2, undersample_strategy=0.5,y_target='target'):
    """
    对DataFrame进行混合采样。

    :param df: 原始数据集的DataFrame
    :param target_column: 目标变量的列名
    :param oversample_strategy: 过采样策略
    :param undersample_strategy: 欠采样策略
    :return: 混合采样后的DataFrame
    """
    # 分割特征和目标变量
    X = df.drop(y_target, axis=1)
    y = df[y_target]

    # 配置SMOTE过采样和RandomUnderSampler欠采样
    over = SMOTE(sampling_strategy=oversample_strategy)
    under = RandomUnderSampler(sampling_strategy=undersample_strategy)

    # 创建采样管道
    pipeline = Pipeline(steps=[('o', over), ('u', under)])

    # 应用管道进行采样
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    # 将采样后的数据重新组装成DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[y_target] = y_resampled

    return df_resampled

# 示例使用:
# 假设 df 是原始数据集的 DataFrame，'y_target' 是目标变量的列名
# df_resampled = mixed_sampling(df, 'y_target', oversample_strategy=


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd


def calculate_sampling_ratios(n_pos, n_neg, r_target,r_under = 0.5):
    """
    计算过采样和欠采样的desired_ratio，以实现指定的少数类/多数类目标比例。

    :param n_pos: 原始少数类样本数量
    :param n_neg: 原始多数类样本数量
    :param r_target: 希望达到的少数类/多数类目标比例
    :param r_under: 1.先指定一个欠采样比例(少数类(1)/多数类目标比例(0),可以调整这个值，决定欠采样的强度
    :return: 欠采样的desired_ratio (r_under), 过采样的desired_ratio (r_over)
    """

    # 2. 计算过采样比例
    r_over = (r_target * n_neg * r_under) / n_pos

    # 3. 如果计算的过采样比例大于1，则调整欠采样比例
    if r_over > 1:
        r_over = 1
        r_under = (n_pos * r_over) / (r_target * n_neg)

    return r_under, r_over

def mixed_sampling_with_categorical(df, desired_ratio=0.5, y_target='target',under_desired_ratio = 0.5,data_dir=None):
    """
       混合采样，按指定目标比例进行过采样和欠采样。

       :param df: 数据集
       :param desired_ratio: 希望达到的少数类/多数类目标比例
       :param y_target: 目标变量列名
       :param under_desired_ratio: 1.先指定一个欠采样比例(少数类(1)/多数类目标比例(0),可以调整这个值，决定欠采样的强度
       :return: 经过混合采样后的DataFrame
       """
    # 获取少数类和多数类样本数量
    n_pos = sum(df[y_target] == 1)
    print('pos:',n_pos)
    n_neg = sum(df[y_target] == 0)
    print('n_neg:', n_neg)

    # 计算过采样和欠采样的比例
    # r_under, r_over = calculate_sampling_ratios(n_pos, n_neg, desired_ratio,r_under=under_desired_ratio)
    r_under = under_desired_ratio
    r_over = desired_ratio
    # 进行欠采样
    df_under = under_sample(df, y_target=y_target, desired_ratio=r_under)

    # 进行过采样
    df_mixed = oversample_dataframe(df_under, y_target=y_target, positive_class_ratio=r_over,data_dir=data_dir)

    return df_mixed

def mixed_sampling_with_categorical_encoded(X,y, desired_ratio=0.5, y_target='target',under_desired_ratio = 0.5,data_dir=None,encoder_type='onehot'):
    """
       混合采样，按指定目标比例进行过采样和欠采样。

       :param X,y: 数据集
       :param desired_ratio: 希望达到的少数类/多数类目标比例
       :param y_target: 目标变量列名
       :param under_desired_ratio: 1.先指定一个欠采样比例(少数类(1)/多数类目标比例(0),可以调整这个值，决定欠采样的强度
       :return: 经过混合采样后的DataFrame
       """
    # 获取少数类和多数类样本数量
    n_pos = sum(y == 1)
    n_neg = sum(y == 0)

    # 计算过采样和欠采样的比例
    # r_under, r_over = calculate_sampling_ratios(n_pos, n_neg, desired_ratio,r_under=under_desired_ratio)
    r_under = under_desired_ratio
    r_over = desired_ratio
    # 进行欠采样
    X,y = under_sample_en(X,y, y_target=y_target, desired_ratio=r_under)

    # 进行过采样
    if encoder_type is None or encoder_type == 'none':
        X_sampled, y_sampled = oversample_x_y(X, y, positive_class_ratio=r_over, data_dir=data_dir)
    else:
        X_sampled, y_sampled = oversample_encoded(X, y, positive_class_ratio=r_over)

    return X_sampled, y_sampled

