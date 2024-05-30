from dataml.feature_process import VarDerivaticeFeatureProcess
from dataml.util.my_logging import logger
import  pandas as pd
import numpy as np
class ScoreCardDerivationFeatureProcess(VarDerivaticeFeatureProcess):
    def __init__(self, name: str='自定义申请评分卡特征衍生'):
        self.name = name

    def process(self, df, df1=None, df2=None, mode='train'):

        # columns = list(df.columns)
        # 对earliesCreditLine进行预处理-取年份
        columns = list(df.columns)
        if 'earliesCreditLine' in columns:
            df['earliesCreditLine'] = df['earliesCreditLine'].astype(str)
            df['earliesCreditLine'] = df['earliesCreditLine'].apply(lambda s: float(s[-4:]))
            logger.info('{},{}'.format(self.name, df.shape))

        if 'grade' in columns:
            df['grade'] = df['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

        if  'employmentLength' in columns:
            df['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
            df['employmentLength'].replace('< 1 year', '0 years', inplace=True)
            df['employmentLength'] = df['employmentLength'].apply(employmentLength_to_int)
        if  'employmentTitle' in columns:
            transform_cnt(df,'employmentTitle')
        if 'postCode' in columns:
            transform_cnt(df, 'postCode')
        if 'regionCode' in columns:
            transform_cnt(df, 'regionCode')
        if  'title' in columns:
            transform_cnt(df,'title')

        for item in ['n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']:
            df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
            df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')

        return df,None

def transform_cnt(df,f):
    df.reset_index(inplace=True)
    df[f + '_cnts'] = df.groupby([f])['id'].transform('count')
    df[f + '_rank'] = df.groupby([f])['id'].rank(ascending=False).astype(int)
    df.set_index('id',inplace=True)
    del df[f]

def employmentLength_to_int(x):
    if pd.isnull(x):
        return x
    else:
        return np.float(x.split()[0])
