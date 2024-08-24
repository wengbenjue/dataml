import os

import numpy as np
from sklearn.feature_selection import VarianceThreshold,chi2
from scipy.sparse import csr_matrix
import matplotlib
import scorecardpy as sc
import pickle
from ..util.utils import feature_classify,null_rate_filter,value_counts_stats,iv_filter,corr_filter,process_data
from ..util.utils import var_equal_bin,prob_plot,most_value_plot,outlier_process,quartile_filter
from ..util.utils import num_csi_filter,cat_csi_filter,var_csi,forest_selection,plot_corr,checkVIF,stepwise_selection
from ..util.imbance_utils import under_sample,oversample_dataframe,mixed_sampling_with_categorical
from ..util.utils import clean_text

from typing import List
import sys
import gc
print(sys.executable)
import warnings


warnings.filterwarnings('ignore')
import pandas as pd

from ..util.my_logging import logger

pd.set_option('display.max_columns', None)


class FeatureProcess:
    def __init__(self):
        pass

    def process(self, df, df1=None, df2=None, mode='train'):
        pass


class UniFeatureProcess(FeatureProcess):
    def __init__(self):
        pass

    def process(self, df, df1=None, df2=None, mode='train'):
        pass



class DoubleFeatureProcess(FeatureProcess):
    def __init__(self):
        pass

    def process(self, df, df1=None, df2=None, mode='train'):
        pass


class PreProcessFeatureProcess(UniFeatureProcess):
    '''
    预处理
    '''
    def __init__(self):
        pass

    def process(self, df, df1=None, df2=None, mode='train'):
        pass


class CleanPreProcessFeatureProcess(PreProcessFeatureProcess):
    def __init__(self,name='数据清洗预处理',y_target='target'):
        self.name = name
        self.y_target = y_target

    def process(self, df, df1=None, df2=None, mode='train'):
        _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
        for cat_fea in cat_features:
            df[cat_fea] = df[cat_fea].apply(lambda x:clean_text(x))
        return df, None


class VarDropFeatureProcess(UniFeatureProcess):
    def __init__(self, name='无关特征删除', drop_columns=None):
        self.name = name
        self.drop_columns = drop_columns

    def process(self, df, df1=None, df2=None, mode='train'):
        if self.drop_columns is not None:
            if isinstance(self.drop_columns,list):
                df =df.drop(columns=self.drop_columns)
            elif isinstance(self.drop_columns,str):
                drop_cols = df.columns[df.columns.str.contains(self.drop_columns)]
                df = df.drop(columns=drop_cols)
            print(self.name,df.shape)
            return df, None

class VarDerivaticeFeatureProcess (UniFeatureProcess):
    def __init__(self,name='变量衍生'):
        self.name =name

    def process(self, df, df1=None, df2=None, mode='train'):
        logger.debug('{},{}'.format(self.name,'暂无衍生，建议通过SQL衍生宽表。'))
        return df,None

class MissValueFeatureProcess(UniFeatureProcess):
    def __init__(self,name='缺失值剔除',miss_rate=0.85):
        self.name =name
        self.miss_rate = miss_rate

    def process(self, df, df1=None, df2=None, mode='train'):
        df = null_rate_filter(df,r=self.miss_rate)
        logger.debug('{},{}'.format(self.name, df.shape))
        return df,None
class CatClusterValueFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '类别变量值合并处理',min_value_count=100,y_target='label'):
        self.name = name
        self.min_value_count= min_value_count
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _,_,_,cat_features,num_features=feature_classify(df,target=self.y_target)
         for f in cat_features:
             print('{},{},{}'.format(f, '类型数：', df[f].nunique()))
         df[cat_features] = df[cat_features].fillna('Unknown')
         df[cat_features] = df[cat_features].replace(np.nan,'Unknown')
         df[cat_features] = df[cat_features].replace('nan','Unknown')
         for col in cat_features:
             cat_value_counts = df[col].value_counts()
             df[col] = df[col].apply(lambda x: 'Unknown' if cat_value_counts[x] < self.min_value_count else x)
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None



class MostValueValueFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '同值率剔除',same_value_rate=0.85,y_target='label'):
        self.name = name
        self.same_value_rate= same_value_rate
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         df,_ = value_counts_stats(df,same_value_rate=self.same_value_rate,target=self.y_target)
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None

class VarianceThresholdFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '方差剔除',v_threshold=0.01,y_target='label'):
        self.name = name
        self.v_threshold= v_threshold
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         df[cat_features] = df[cat_features].astype(str)
         df[num_features] = df[num_features].astype(float)

         sel = VarianceThreshold(self.v_threshold)
         sel.fit(df[num_features])
         mask = sel.get_support()
         num_features = list(df[num_features].columns[mask])
         logger.info('筛选出的数值特征个数：{}'.format(len(num_features)))
         df = df[cat_features+num_features]
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None

class OutlierFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '异常值处理',cache_dir='./data',y_target='label'):
        self.name = name
        self.cache_dir= cache_dir
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         df_rem=outlier_process(df,self.cache_dir,num_features,flag=mode)
         logger.debug('{},{}'.format(self.name, df.shape))
         logger.debug(len(num_features))

         df = pd.concat([df[cat_features],df_rem],axis=1)
         del df_rem
         gc.collect()
         gc.collect()
         return df,None

class QuartileFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '四分位差剔除',y_target='label'):
        self.name = name
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         num_features = quartile_filter(num_features,df)
         df = pd.concat([df[cat_features], df[num_features]], axis=1)
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None

class MissValueFillFeatureProcess (UniFeatureProcess):
    def __init__(self,name = '缺失值填充',model_dir=None,y_target='label'):
        self.name = name
        self.model_dir = model_dir
        stats_file = os.path.join(self.model_dir, 'state_file.json')  # 缺失率填充分布序列化文件名
        self.stats_file = stats_file
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         df=process_data(df,num_features,cat_features,stats_file=self.stats_file,mode=mode)
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None

#双变量处理
class IVFeatureProcess (DoubleFeatureProcess):
    def __init__(self,name = 'IV值剔除',bin_method='tree',
                 min_iv_limit=.02,max_iv_limit=.5,
                 max_num_bin = 5,
                 model_dir=None,y_target='label',target_positive='bad|1'):
        self.name = name
        self.bin_method = bin_method
        self.min_iv_limit = min_iv_limit
        self.max_iv_limit = max_iv_limit
        self.model_dir = model_dir
        self.max_num_bin = max_num_bin
        self.y_target=y_target
        self.target_positive=target_positive

    def process(self, df, df1=None, df2=None, mode='train'):
        bins = sc.woebin(df, y=self.y_target, x=None
                         , breaks_list=None
                         , special_values=None
                         , min_perc_fine_bin=0.02  # 初始细分箱，每个箱子等等比例分：0.02>>>>>50箱
                         , min_perc_coarse_bin=0.05  # 最终粗分箱中，每个箱的最小样本占比
                         , stop_limit=0.05  # 分箱临界阀值
                         , max_num_bin=self.max_num_bin  # 最终最大分箱数
                         , positive=self.target_positive
                         , no_cores=None
                         , print_step=0
                         , method=self.bin_method)  # chimerge 分箱方法，决策树或卡方
        var_bin_df = pd.DataFrame()
        for key, value in bins.items():
            var_bin_df = pd.concat([var_bin_df, value], axis=0)
        try:
            logger.debug(var_bin_df.head())
            var_bin_df.to_csv(os.path.join(self.model_dir,'var_bin.csv'))
        except Exception as e:
            logger.error(f'error:{e}')

        with open(os.path.join(self.model_dir,'bins.pkl'), "wb") as f:
            pickle.dump(bins, f)
            logger.debug('bins保存成功!')
        #IV值变量筛选
        iv_features, var_iv_list = iv_filter(var_bin_df, min_iv_limit=self.min_iv_limit, max_iv_limit=self.max_iv_limit,model_dir=self.model_dir)
        x_var_iv_list = [c for c in iv_features]
        iv_features.append(self.y_target)
        df = df[iv_features]
        # iv_df = var_iv_list.reset_index(inplace=True)
        iv_df = var_iv_list.reset_index(inplace=False)
        iv_df.to_csv(os.path.join(self.model_dir,'iv_df.csv'))
        logger.debug('{},{}'.format(self.name, df.shape))
        return df,iv_df

class CorrFeatureProcess (DoubleFeatureProcess):
    def __init__(self,name = 'Pearson相关性系数剔除',corr_threshold=.85,y_target='label'):
        self.name = name
        self.corr_threshold = corr_threshold
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         iv_df = df1
         # iv_features =  [ c for c in list(df.columns) if c not in self.y_target]
         corr_filter_list = corr_filter(df[num_features],num_features,iv_df,corr_threshold=self.corr_threshold)
         corr_filter_features = list(set([c for c in corr_filter_list]))
         cat_features.append(self.y_target)
         df = pd.concat([df[corr_filter_features],df[cat_features]],axis=1)
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None

class CSIFeatureProcess(DoubleFeatureProcess):
    def __init__(self, name = '变量CSI筛选', csi_threshold = .25,
                 model_dir=None,bins=[0.0, 0.2, 0.4, 0.6, 0.8 ,1.0], y_target='label'):
        self.name = name
        self.csi_threshold = csi_threshold
        self.model_dir = model_dir
        self.y_target = y_target
        self.bins = bins

    def process(self, df, df1=None, df2=None, mode='train'):
        if df2 is None:
            logger.info('跨期数据唯恐，跳过CSI筛选特征')
            return df,None
        _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
        #连续型变量
        psi_rem_df = pd.DataFrame()
        df_kq = df2
        y_df = df[self.y_target]
        y_df = y_df.astype(int)
        df_kq_y = df2[self.y_target]
        num_features,psi_rem_df = num_csi_filter(df_rem=df,df_kq=df_kq,num_df_list=num_features,y_df=y_df,
                                                 df_kq_y=df_kq_y,csi_threshold=self.csi_threshold,bins=self.bins)
        cat_features,psi_rem_df = cat_csi_filter(cat_features,df,df_kq,y_df,df_kq_y,psi_rem_df,csi_threshold=self.csi_threshold)

        cat_features.append(self.y_target)
        df=pd.concat([df[num_features],df[cat_features]],axis=1)
        if len(psi_rem_df) > 0:
            logger.info(psi_rem_df.head())
            try:
                psi_rem_df.columns = ['feature','csi']
                psi_rem = psi_rem_df[psi_rem_df['csi']<=self.csi_threshold].copy()
                psi_rem.to_csv(os.path.join(self.model_dir,'csi.csv'))

            except Exception as e:
                logger.error(f'error:{e}')

            logger.info('{},{}'.format(self.name, df.shape))
        return df, None


class ForestFeatureProcess (DoubleFeatureProcess):
    def __init__(self,name = '树模型累计重要性剔除',corr_threhold=.85,miss_rate=.85,
                 forest_select_cumutative_importance=.95,
                 max_depth=5,
                 y_target='label'):
        self.name = name
        self.miss_rate = miss_rate
        self.forest_select_cumutative_importance = forest_select_cumutative_importance
        self.max_depth = max_depth
        self.corr_threhold = corr_threhold
        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
         _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
         y_df = df[self.y_target]
         y_df = y_df.astype(int)
         df_all = df.copy()
         df = pd.concat([df[num_features],df[cat_features]],axis=1)
         df_rem_forest=forest_selection(df,y_df,missing_threshold=self.miss_rate,
                                        correlation_threshold=self.corr_threhold,
                                        cumutative_importance=self.forest_select_cumutative_importance,
                                        max_depth=self.max_depth)
         forest_filter_features=[col for col in list(df_rem_forest.columns) if not col.endswith('_nan')]
         feature_result = forest_filter_features
         feature_result.append(self.y_target)
         df = df_all[feature_result]
         logger.debug('{},{}'.format(self.name, df.shape))
         return df,None


class ImbalanceProcess(DoubleFeatureProcess):
    '''
    数据不均衡处理
    '''
    def __init__(self,name = '数据不均衡处理',imbalance_type='under_sample',desired_ratio=2,k_neighbors=5,
                 under_desired_ratio=1/6,
                 y_target='label',
                 data_dir='./data'):
        self.name = name
        self.imbalance_type = imbalance_type
        #少数类占多数类比例，目标样本浓度
        self.desired_ratio = desired_ratio
        self.k_neighbors = k_neighbors
        self.under_desired_ratio = under_desired_ratio
        self.data_dir = data_dir

        self.y_target=y_target

    def process(self, df, df1=None, df2=None, mode='train'):
        df_balanced = df.copy()
        if self.imbalance_type is None:
            logger.info('no sample strategy,do not process.')
            return df_balanced,None
        if self.imbalance_type == 'under_sample':
            df_balanced = under_sample(df,self.y_target,self.desired_ratio)
        elif self.imbalance_type == 'over_sample':
            df_balanced = oversample_dataframe(df, y_target=self.y_target, positive_class_ratio=self.desired_ratio,data_dir=self.data_dir)
        else:
            df_balanced = mixed_sampling_with_categorical(df,
                                                          self.desired_ratio,
                                                          y_target=self.y_target,
                                                          under_desired_ratio=self.under_desired_ratio,data_dir=self.data_dir)
        logger.debug('{},{},using {} sample strategy'.format(self.name, df.shape,self.imbalance_type))
        return df_balanced,None




