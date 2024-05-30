import os
import gc
import uuid
import string
import random
import pandas as pd
import sys
from typing import List
import pickle
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


from .model.lightgbm_model import LightGBMModel
from .model.logistic_regression_model import LogisticRegressionModel
from .util.my_logging import logger
from .datasets import SQLDataset,Dataset
from .feature_process import FeatureProcess
from .util.utils import merge_dict,prob_distplot,feature_classify,onehot_encode
from .util.utils import CustomScaler,gmm_train,output_list_by_model,catch_sim_or_notsim_cst,load_all_model_features
from .util.utils import plot_corr,plot_histogram,model_psi,jaccard_similarity
from .util.utils import lr_scored_card
from .util.utils import circle_distrib,plot_distribuion,generate_short_uuid
from .feature_process import VarDerivaticeFeatureProcess
from .feature_encoder import FeatureEncoder,OnehotEncoder,WoeEncoder
logger.info(sys.executable)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class MLModel:
    '''
    Args:
        watch_point: 观察点
        oot_watch_point: 超时观察点
        need_reload_data: 是否需要重新加载数据
        params_optimize: 参数优化
        need_random_search: 是否需要随机搜索
        need_grid_search: 是否需要网格搜索
        same_value_rate: 相同值率阈值
        v_threshold: 方差阈值
        cache_dir: 缓存目录
        model_dir_tabletablename: 模型目录表名
        prv_table_tablename：私有表表名
        cpct_big_tablename：公共大表表名
        cpct_reduce_dim：公共大表降维方法
        need_visuz：是否需要可视化
        need_tsne_visuz: 是否需要T-SNE可视化
        need_recall_negative: 是否需要召回负样本
        negative_sim_threshold: 负样本相似度阈值
        min_iv_limit: 最小IV值
        max_iv_limit: 最大IV值
        corr_threshold: 相关系数阈值
        bin_method: 分箱方法 | chimerge 卡方分箱等各种分箱方法
        tortatus_select_importance_type | 重要性类型选择
        forest_select_max_depth: 随机森林最大深度
        warehouse_address：数据仓库地址
        encoding_type：编码类型 OneHot或者WOE
        models：模型列表
        need_tree_filter 是否需要决策树过滤器
        train_dataset 训练集数据源配置项
        oot_dataset OOT数据源配置项
        pred_dataset 预测集数据源配置项 outlist_table_name 黑名单表名 & 白名单表名 user_profile_label_rules 用户画像标签规则
    '''

    def __init__(self,
                 cache_dir='./data/',
                 model_dir='./models/',
                 y_target='label_cd',
                 encoding_type='onehot',
                 need_reduce_dim=False,
                 need_visuz=True,
                 need_tsne_visuz = False,
                 need_recall_negative=False,
                 negative_sim_threshold=0.35,
                 test_ratio=0.2,
                 drop_columns=None,
                 models :List=  [LogisticRegressionModel, LightGBMModel],
                 train_dataset: Dataset = None,
                 oot_dataset: Dataset = None,
                 pred_dataset: Dataset = None,
                 out_dataset: Dataset = None,
                 cst_base_info_dataset: Dataset = None,
                 uni_feature_process: List[FeatureProcess] = None,
                 double_feature_process: List[FeatureProcess] = None,
                 csi_feature_process: List[FeatureProcess] = None,
                 oot_feature_process: List[FeatureProcess] = None,
                 feature_encoder: FeatureEncoder=None,
                 min_lift = 1.5,
                 sheet_name = 'Sheet1',
                 var_en_name = '',
                 var_zh_name = '',
                 need_pre_process = False,
                 data_dict_path = None,
                 user_profile_label_rules = [],
                 n_components=500,
                 index_col='cst_id',
                 model_instance=None):

        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.y_target = y_target
        self.encoding_type = encoding_type
        self.need_reduce_dim = need_reduce_dim
        self.need_visuz = need_visuz
        self.need_tsne_visuz = need_tsne_visuz
        self.need_recall_negative = need_recall_negative
        self.negative_sim_threshold = negative_sim_threshold
        self.test_ratio = test_ratio
        self.drop_columns = drop_columns
        self.n_components = n_components
        self.models = models
        self.train_dataset = train_dataset
        self.oot_dataset = oot_dataset
        self.pred_dataset = pred_dataset
        self.out_dataset = out_dataset
        self.feature_encoder = feature_encoder

        self.cst_base_info_dataset = cst_base_info_dataset
        self.uni_feature_process = uni_feature_process
        self.double_feature_process = double_feature_process
        self.csi_feature_process = csi_feature_process
        self.oot_feature_process = oot_feature_process
        self.min_lift = min_lift
        self.sheet_name = sheet_name
        self.var_en_name = var_en_name
        self.var_zh_name = var_zh_name
        self.need_pre_process = need_pre_process
        self.data_dict_path = data_dict_path
        self.user_profile_label_rules = user_profile_label_rules
        self.index_col = index_col
        self.model_instance = model_instance

        if self.model_instance is None:
            #generate instance for model
            # self.model_instance = uuid.uuid4()
            # self.model_instance = str(uuid.uuid4())[:8]
            self.model_instance = generate_short_uuid()
            print('automatic generate model instance id:{}'.format(self.model_instance))

        self.model_dir = os.path.join(self.model_dir,self.model_instance)
        self.cache_dir = os.path.join(self.cache_dir,self.model_instance)
        if self.models is not None:
            for model in self.models :
                model.model_dir = self.model_dir
                model.cache_dir = self.cache_dir
        if train_dataset is not None:
            train_dataset.model_dir = self.model_dir
            train_dataset.cache_dir = self.cache_dir

        if train_dataset is not None:
            train_dataset.model_dir = self.model_dir
            train_dataset.cache_dir = self.cache_dir

        if oot_dataset is not None:
            oot_dataset.model_dir = self.model_dir
            oot_dataset.cache_dir = self.cache_dir

        if pred_dataset is not None:
            pred_dataset.model_dir = self.model_dir
            pred_dataset.cache_dir = self.cache_dir

        if out_dataset is not None:
            out_dataset.model_dir = self.model_dir
            out_dataset.cache_dir = self.cache_dir

        if cst_base_info_dataset is not None:
            cst_base_info_dataset.model_dir = self.model_dir
            cst_base_info_dataset.cache_dir = self.cache_dir

        if uni_feature_process is not None:
            for feature_process in uni_feature_process:
                feature_process.model_dir = self.model_dir
                feature_process.cache_dir = self.cache_dir

        if double_feature_process is not None:
            for feature_process in double_feature_process:
                feature_process.model_dir = self.model_dir
                feature_process.cache_dir = self.cache_dir

        if csi_feature_process is not None:
            for feature_process in csi_feature_process:
                feature_process.model_dir = self.model_dir
                feature_process.cache_dir = self.cache_dir

        if oot_feature_process is not None:
            for feature_process in oot_feature_process:
                feature_process.model_dir = self.model_dir
                feature_process.cache_dir = self.cache_dir

        if feature_encoder is not None:
            feature_encoder.model_dir = self.model_dir
            feature_encoder.cache_dir = self.cache_dir

        self.onehot_encoder= OnehotEncoder(model_dir=self.model_dir,y_target=self.y_target)
        self.woe_encoder= WoeEncoder(model_dir=self.model_dir)

        if self.feature_encoder is None:
            # default use onehot encoder
            self.feature_encoder = self.onehot_encoder


        self.negative_custom_scaler =CustomScaler(file_name=os.path.join(self.model_dir,'ne_scaler.pkl'))
        self.custom_scaler = CustomScaler(file_name=os.path.join(self.model_dir,'scaler.pkl'))

        self.encoder_type = 'onehot'
        if isinstance(self.feature_encoder, WoeEncoder):
            self.encoder_type = 'woe'

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.need_pre_process:
            self.pre_process()



    def pre_process(self):
        #读取处理数据字典
        self.data_dict_path = merge_dict(self.cache_dir)

    def load_data_map(self):
        if self.data_dict_path is None:
            self.data_map = {}
            self.zh2en_map = {}
        else:
            if self.sheet_name is not None:
                data_dict = pd.read_excel(self.data_dict_path, sheet_name=self.sheet_name)
            else:
                data_dict = pd.read_excel(self.data_dict_path)
            data_dict[self.var_en_name] = data_dict[self.var_en_name].str.lower()
            data_dict_unique = data_dict.drop_duplicates(subset=self.var_en_name)
            data_dict_map_df = data_dict_unique.copy()
            data_map = data_dict_map_df.set_index(self.var_en_name)[self.var_zh_name].to_dict()
            self.data_map = data_map
            self.zh2en_map = {value: key for key, value in data_map.items()}

    def load_model_features(self):
        """
        加载模型特征字段
        :return:
        """
        self.model_features = load_all_model_features(self.model_dir)

    def load_data(self, dataset, mode='pred',source='pred'):
        """
        读取数据
        :param dataset:
        :param mode:
        :param source:
        :return:
        """
        df_all = dataset.read()
        if self.y_target in df_all.columns:
            df_all[self.y_target] = df_all[self.y_target].replace(np.nan,0)

        df_y = None
        if 'list' in mode:
            return  df_all

        if mode != 'pred' and mode != 'predict':
            df_all[self.y_target] = df_all[self.y_target].astype(int)
            df_y = df_all[[self.y_target]]

        if mode != 'train' and source != 'train' and mode !='pred' and mode != 'predict':
            df_all[self.y_target] = df_y[self.y_target]

        if (mode != 'oot'  or mode == 'outtime' or mode == 'kq') and mode !='pred' and mode != 'predict':
            df_all[self.y_target] = df_y[self.y_target]

        if mode !='pred' and mode != 'predict':
            logger.info('sample distribution.')
            logger.info(df_y[self.y_target].value_counts())

        logger.info(df_all.shape)


        if df_y is not None and self.y_target in list(df_y.columns):
            logger.info('Sample Distribution Before Feature Process:')
            y_df = df_y[self.y_target]
            y_distribution = y_df.value_counts()
            logger.info(y_distribution)
            converted = y_distribution.rename_axis('unique_values').to_frame('counts')
            # plot the distribution
            circle_distrib(df_dis=converted.counts, labels=['negative', 'positive'])

        return df_all,df_y


    def train(self, model_type=None,encoder_type='onehot'):
        """
        train model

        """

        logger.info('开始训练...')
        # 加载数据集
        df_all, df_Y = self.load_data(self.train_dataset,mode = 'train', source = 'train')
        # 特征筛选
        iv_df,feature_result, df, y_df = self.feature_process(df=df_all, df_y=df_Y, mode='train')
        del df_all
        gc.collect()
        gc.collect()

        #数据集划分
        X, y,X_test, y_test = self.train_test_split(feature_result, df,y_df)
        del df
        gc.collect()
        gc.collect()
        #特征归一化
        X = self.feature_scaler(X, model='train')
        X_test = self.feature_scaler(X_test, model='test')

        #特征编码
        if  'onehot' in  encoder_type :
            self.feature_encoder = self.onehot_encoder
        # elif 'woe' in  encoder_type :
        #     self.feature_encoder = self.woe_encoder
        else:
            self.feature_encoder = self.woe_encoder

        self.encoder_type = encoder_type

        X = self.feature_encoding(X, model='train')
        X_test = self.feature_encoding(X_test, model='test')

        logger.info('训练集特征维数：')
        logger.info(X.shape)
        logger.info('测试集特征维数：')
        logger.info(X_test.shape)

        gc.collect()
        gc.collect()

        if model_type is not  None:
            filter_models = [model for model in self.models if model.model_type in model_type]
        if model_type is None:
            filter_models = self.models

        logger.info('model numbers:{}'.format(len(filter_models)))

        for model in filter_models:
            logger.info('{} training'.format(model.name))
            model.train(X,y)
            model.persist(self.model_dir,encoder_type=self.encoder_type)
            #model evalueate
            y_score_test,y_pred_test,test_df = model.evaluate(
                X,
                X_test,
                y,
                y_test,
                self.y_target,
                cache_dir=self.cache_dir,
                index_col=self.index_col
            )
            test_df.to_csv(os.path.join(self.cache_dir,model.model_type+'_test_model_probs.csv'),encoding='utf8',index=False)
            #feature importance
            feature_imp,data_dict=model.model_feature_importance(X,self.data_dict_path,iv_df=iv_df,data_dir=self.cache_dir)
            model.feature_explain(X_test=X_test,y_test=y_test,data_map=data_dict)
            gc.collect()
            gc.collect()

    def outtime(self, model_type=None,encoder_type='onehot'):
        ...
        print('模型预测开始...')
        self.load_model_features()
        model_pred_result = []
        df, df_y = self.load_data(self.oot_dataset,mode='oot', source='oot')

        # 特征筛选
        iv_df, df_feature_result, df, y_df = self.feature_process(df=df, df_y=df_y, mode='outtime')


        # 特征归一化
        df = self.feature_scaler(df, model='test')

        if  'onehot' in  encoder_type :
            self.feature_encoder = self.onehot_encoder
        # elif 'woe' in  encoder_type :
        #     self.feature_encoder = self.woe_encoder
        else:
            self.feature_encoder = self.woe_encoder

        self.encoder_type = encoder_type

        # 特征编码
        X = self.feature_encoding(df, model='test')

        gc.collect()

        print(X.shape)


        # 模型预测结果输出
        if model_type is not None:
            filter_models = [model for model in self.models if model.model_type in model_type]
        else:
            filter_models = self.models
        logger.info('number of models of outtime evaluate.')

        # 加载模型
        for model in filter_models:
            logger.info('current model:{}'.format(model.name))
            model.load(self.model_dir,encoder_type=self.encoder_type)

            y_score_test,y_pred_test,test_df  = model.evaluate_test(
                X,
                y_test=y_df, y_target=self.y_target,
                cache_dir=self.cache_dir,flag='oot',index_col=self.index_col)

            test_df.to_csv(os.path.join(self.cache_dir,model.model_type+'_oot_model_probs.csv'),encoding='utf8',index=False)

            #feature importance
            featrue_imp,data_dict = model.model_feature_importance(X,self.data_dict_path,iv_df=iv_df,data_dir=self.cache_dir)

            model.feature_explain(X_test=X,y_test=y_df,data_map=data_dict)
            gc.collect()
            gc.collect()

    def predict_feature_process(self,encoder_type='onehot'):
        """
        数据预处理、编码
        :return: 
        """
        # 加载模型及特征
        pred_df = pd.DataFrame()
        fea_df = pd.DataFrame()

        logger.info('infrence...')
        # 开始加载模型及特征
        self.load_model_features()

        model_pred_result = []
        # 加载全部数据
        df_all, df_y = self.load_data(self.pred_dataset, mode='pred', source='pred')

        pred_df['cs_id'] = list(df_all.index)
        gc.collect()
        gc.collect()

        iv_df, feature_result,df, _ = self.feature_process(df_all, df_y, mode='pred')

        fea_df = df[self.model_features]
        gc.collect()
        gc.collect()

        # 特征归一化
        df = self.feature_scaler(df, model='test')


        # 特征编码
        if  'onehot' in  encoder_type :
            self.feature_encoder = self.onehot_encoder
        elif 'woe' in  encoder_type :
            self.feature_encoder = self.woe_encoder
        else:
            self.feature_encoder = self.onehot_encoder

        X = self.feature_encoding(df, model='test')
        gc.collect()
        gc.collect()
        logger.info(X.shape)
        return X, pred_df, fea_df

    def predict(self, model_type='lgb',encoder_type='onehot'):


        X,pred_df, _ = self.predict_feature_process(encoder_type=encoder_type)
        self.encoder_type = encoder_type

        if model_type is None:
            model_pred_result = {}
            for model in self.models:
                logger.info(f'[{model.name}]预测')
                model.load(self.model_dir,encoder_type=self.encoder_type)
                y_score, y_pred, df = model.predict(X)
                pred_df.index = df.index
                pred_df['prob'] = y_score
                prob_distplot(pred_df, var_name='prob', bins=10)
                model_pred_result[model.model_type] = list(y_pred)
            return model_pred_result

        else:
            for model in self.models:
                logger.info(f'[{model.name}]预测')
                if (model.type == model_type):
                    model.load(self.model_dir,encoder_type=self.encoder_type)
                    y_score, y_pred, df = model.predict(X)
                    pred_df.index = df.index
                    pred_df['prob'] = y_score
                    prob_distplot(pred_df, var_name='prob', bins=10)
            return pred_df

    def feature_encoding(self, X,model='train'):
        '''
        feature encoding
        :param X:
        :param model:
        :return:
        '''
        return self.feature_encoder.encode(X=X,flag=model)

    def feature_scaler(self, X, model='train'):
        """
        特征值归一化
        :param X:
        :param model:
        :return:
        """
        _, _, _, cat_features, num_features = feature_classify(X, target=self.y_target)
        num_df = None
        if model == 'train':
            # num_df = self.custom_scaler.fit_transform(X[num_features])
            with open(os.path.join(self.model_dir, 'custom_scaler_pickle'), 'wb') as f:
                pickle.dump(self.custom_scaler,f)
                logger.info('persist custom_scaler success.')
        else:
            self.custom_scaler = pickle.load(open(os.path.join(self.model_dir, 'custom_scaler_pickle'), 'rb'))
        num_df = self.custom_scaler.fit_transform(X[num_features])

        X = pd.concat([X[cat_features], num_df], axis=1)
        return X

    def negative_sample_gene(self, df, df_y, cat_features, num_features, model_dir, y_target='label_cd',
                             negative_sim_threshold=0.36, flag='train', need_reduce_dim=False, back_filter=True):
        """
        生成负样本
        :param df: 原始数据DataFrame
        :param df_y:
        :param cat_features:
        :param num_features:
        :param model_dir:
        :param y_target:
        :param negative_sim_threshold: 负样本筛选相似度计算阈值
        :param flag:
        :param need_reduce_dim: 是否需要降维
        :param back_filter: 筛选黑名单
        :return:
        """
        # n_cat_oneshot_df = onehot_encode(df[cat_features], model_dir, y_target, flag='train')
        self.feature_encoder.encode(X=df[cat_features], flag='train')

        n_num_df = self.negative_custom_scaler.fit_transform(df[cat_features])
        negative_df = pd.concat([cat_features, n_num_df], axis=1)
        logger.info('dimension of negative sample:{}.'.format(negative_df.shape))
        if need_reduce_dim:
            X_n_sparse = csr_matrix(negative_df)
            n_components = self.n_components
            svd = TruncatedSVD(n_components=n_components)

            x_n_reduced = svd.fit_transform(X_n_sparse)

        else:
            x_n_reduced = np.array(negative_df)

        # caculate similarity
        catch_sim_or_notsim_cst(df_y, x_n_reduced, threshold=negative_sim_threshold, y_target=y_target,
                                back_filter=back_filter, positive_label=1, unlablel_label=-1)
        logger.info(df_y.shape)

        df_y[self.y_target] = df_y[self.y_target].astype(int)
        df[self.y_target] = df_y[self.y_target]

        df_unlabel = df[df[self.y_target] == -1]
        logger.info('shape of unlabel dataframe:{}'.format(df_unlabel.shape) )
        df = df[df[self.y_target] != -1]
        y_df = df[self.y_target]
        y_df = y_df.astype(int)
        logger.info('shape of df dataframe:{}'.format(df.shape) )

    def plot_feature_corr(self,df,feature_result):
        """
        热力图
        :param df:
        :param feature_result:
        :return:
        """
        self.load_data_map()
        zh_filter_features = []
        rename_dict = {}
        for col in feature_result:
            if col in self.data_map:
                rename_dict[col] = self.data_map[col]
                zh_filter_features.append(self.data_map[col])
            else:
                rename_dict[col] = col
                zh_filter_features.append(col)
        df_zh = df.copy()
        del df
        gc.collect()
        gc.collect()
        df_zh.rename(columns=rename_dict,inplace=True)
        plot_corr(df_zh)

    def feature_process(self,df,df_y,mode='train'):
        feature_result = []
        if mode != 'pred' and mode != 'predict':
            y_df = df_y[self.y_target]
        if mode == 'train':
            #uni-variable filter
            if len(self.uni_feature_process) >0:
                for feature_proc_obj in self.uni_feature_process:
                    df,_ = feature_proc_obj.process(df,df1=None,df2=None,mode='train')

            gc.collect()
            gc.collect()

            #filter negative
            if self.need_recall_negative:
                _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
                df,y_df,df_y=self.negative_sample_gene(df,df_y,cat_features,num_features,self.model_dir,
                                                       y_target=self.y_target,negative_sim_threshold=self.negative_sim_threshold,
                                                       flag=mode,need_reduce_dim=self.need_reduce_dim,back_filter=True)

            gc.collect()
            gc.collect()
            #double-variable filter
            if len(self.double_feature_process) >0:
                df1 = None
                df2 = None
                if self.oot_dataset is not None:
                    df2,df2_y = self.load_data(self.oot_dataset,mode='oot',source=mode)
                    # df2=df2[list(df.columns)]
                    df2[self.y_target] = df2_y[self.y_target]
                    csi_var_d_flag = False
                    if len(self.csi_feature_process) >0:
                        for feature_proc_obj in self.csi_feature_process:
                            if isinstance(feature_proc_obj,VarDerivaticeFeatureProcess):
                                csi_var_d_flag = True
                        if not csi_var_d_flag:
                            df2 = df2[list(df.columns)]

                        if len(self.csi_feature_process) >0:
                            for feature_proc_obj in self.csi_feature_process:
                                df2, _ = feature_proc_obj.process(df2, df1=None, df2=None, mode='test')
                                if isinstance(feature_proc_obj, VarDerivaticeFeatureProcess):
                                    df2 = df2[list(df.columns)]

                    gc.collect()
                    gc.collect()

                    # filter negative
                    if self.need_recall_negative:
                        _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
                        df2, y_df, df2_y = self.negative_sample_gene(df, df_y, cat_features, num_features, self.model_dir,
                                                                   y_target=self.y_target,
                                                                   negative_sim_threshold=self.negative_sim_threshold,
                                                                   flag=mode, need_reduce_dim=self.need_reduce_dim,
                                                                   back_filter=True)
                        df2[self.y_target] = df2_y[self.y_target]
                    else:
                        df2[self.y_target] =  df2_y[self.y_target]

                    gc.collect()
                    gc.collect()
                    # df[self.y_target] = y_df
                    # for feature_proc_obj in self.double_feature_process:
                    #     df, df1 = feature_proc_obj.process(df, df1=df1, df2=df2, mode='train')
                    #     gc.collect()
                    #     gc.collect()
                    # feature_result  = list(df.columns)
                    # # if self.y_target in feature_result:
                    # #     feature_result.remove(self.y_target)
                    # y_df = df[self.y_target]
                else:
                    pass
                    # feature_result = list(df.columns)
                        # feature_result.remove(self.y_target)

                df[self.y_target] = y_df
                for feature_proc_obj in self.double_feature_process:
                    df, df1 = feature_proc_obj.process(df, df1=df1, df2=df2, mode='train')
                    gc.collect()
                    gc.collect()
                feature_result = list(df.columns)
                # if self.y_target in feature_result:
                #     feature_result.remove(self.y_target)
                y_df = df[self.y_target]

                if self.y_target in feature_result:
                    feature_result.remove(self.y_target)

        else:
            if mode != 'pred' and mode != 'predict':
                y_df = df[self.y_target]
            else:
                y_df = None

            # df = df[self.model_features]

            csi_var_d_flag = False
            if len(self.oot_feature_process) >0:
                for feature_proc_obj in self.oot_feature_process:
                    if isinstance(feature_proc_obj, VarDerivaticeFeatureProcess):
                        csi_var_d_flag = True
                if not csi_var_d_flag:
                    df = df[self.model_features]

                for feature_proc_obj in self.oot_feature_process:
                    df,_ = feature_proc_obj.process(df,df1=None,df2=None,mode='test')
                    if isinstance(feature_proc_obj, VarDerivaticeFeatureProcess):
                        df = df[self.model_features]
                    gc.collect()
                    gc.collect()

                if mode != 'pred' and mode != 'predict':
                    if self.need_recall_negative:
                        _, _, _, cat_features, num_features = feature_classify(df, target=self.y_target)
                        df, y_df, df_y = self.negative_sample_gene(df, df_y, cat_features, num_features,
                                                                   self.model_dir,
                                                                   y_target=self.y_target,
                                                                   negative_sim_threshold=self.negative_sim_threshold,
                                                                   flag=mode, need_reduce_dim=self.need_reduce_dim,
                                                                   back_filter=True)
                        df[self.y_target] = y_df

                    gc.collect()
                    gc.collect()
                feature_result = self.model_features

            logger.info('number of final features:'.format(len(feature_result)))
            logger.info(feature_result)

        self.plot_feature_corr(df,feature_result)
        #load iv
        iv_df = pd.read_csv(os.path.join(self.model_dir,'iv_df.csv'))


        if y_df is not None and len(y_df)>0:
            logger.info('Sample Distribution After Feature Process:')
            y_distribution = y_df.value_counts()
            logger.info(y_distribution)
            converted = y_distribution.rename_axis('unique_values').to_frame('counts')
            # plot the distribution
            circle_distrib(df_dis=converted.counts, labels=['negative', 'positive'])

        return iv_df,feature_result, df, y_df


    def train_test_split(self,model_features,df,y_df):
        """
        数据集划分
        :param model_features:筛选后的模型特征
        :param df:模型筛选后的DataFrame
        :param y_df:标签DataFrame
        :return:
        """

        y_df = y_df.astype(int)
        train,test,y,y_test =train_test_split(df,y_df,test_size=self.test_ratio,random_state=3,stratify=y_df)
        X = train[model_features]

        X_test = test[model_features]
        X = X.replace(np.nan,0)
        y = y.replace(np.nan,0)

        X_test = X_test.replace(np.nan,0)
        y_test = y_test.replace(np.nan,0)

        logger.info('length of train:{}'.format(X.shape))
        logger.info('length of test:{}'.format(X_test.shape))
        logger.info('train sample distribution:')
        logger.info(y.value_counts())

        with open(os.path.join(self.model_dir,'model_features.pickle'),'wb') as f:
            pickle.dump(model_features,f)
            logger.info('model features persist success.')

        return X,y,X_test,y_test

    def explore(self, feature_names: List[str] = None, rotation=0):
        self.explore(feature_names=feature_names,rotation=rotation,label_bad='bad',
                   label_good='positive', label='bad_rate%')

    def explore(self,feature_names:List[str]=None,rotation=0,label_bad='bad',
                   label_good='positive', label='bad_rate%'):
        """
        特征画像
        :param feature_names:
        :param rotation:
        :return:
        """
        #load model features
        self.load_model_features()
        self.load_data_map()
        #load bin result
        bins = pickle.load(open(os.path.join(self.model_dir,'bins.pkl'),'rb'))
        var_bin_tmp=pd.concat(bins,ignore_index=True)

        if len(self.model_features) == 0:
            logger.info('please pre-train model first.')
        else:
            filter_imp_df=pd.read_csv(os.path.join(self.cache_dir,'lgb_feature_importance.csv'))
            if filter_imp_df is None or len(filter_imp_df) ==0:
                filter_features = self.model_features
            else:
                print(filter_imp_df.head())
                try:
                    filter_features = list(filter_imp_df[self.var_en_name].values)
                except:
                    filter_features = self.model_features

            if feature_names is not None and len(feature_names) >0:
                filter_features = feature_names

            zh_filter_features = []
            for col in filter_features:
                if col in self.data_map:
                    zh_filter_features.append(self.data_map[col])
                else:
                    zh_filter_features.append(col)

            for inx,feature in enumerate(filter_features):
                plot_histogram(var_bin_tmp,feature,col_name=zh_filter_features[inx],rotation=rotation,label_bad=label_bad,
                   label_good=label_good, label=label)

    def oot_model_psi(self,score_name='y_prob',p=10,model_type='lgb'):
        """
        模型PSI
        :param p:
        :param model_type:
        :return:
        """
        for model in self.models:
            if model_type == model.model_type:
                oot_prob_df = pd.read_csv(os.path.join(self.cache_dir,model.model_type+'_oot_model_probs.csv'),encoding='utf8')
                test_prob_df = pd.read_csv(os.path.join(self.cache_dir,model.model_type+'_test_model_probs.csv'),encoding='utf8')
                model_psi_oot = model_psi(test_prob_df[score_name].values,oot_prob_df[score_name].values,p)
                logger.info('outtime model psi:{}'.format(model_psi_oot))
                return model_psi_oot

    def output_list(self, model_type=None, min_lift=None,encoder_type='onehot'):
        """
        模型清单
        :param model_type:
        :param min_lift:
        :return:
        """
        cst_base_info_df=pd.DataFrame()
        X,pred_df,fea_df = self.predict_feature_process(encoder_type=encoder_type)
        self.encoder_type = encoder_type
        actual_min_lift = self.min_lift

        if min_lift is not None:
            actual_min_lift = min_lift
        # Model type is not None
        if model_type is not None:
            filter_models = [model for model in self.models if model.model_type == model_type]

        if model_type is None:
            filter_models = self.models

        logger.info('number of model for output list.')

        for model in filter_models:
            print('{}output list'.format(model.name))
            # Model prediction dataframe copy
            model_pred_df = pred_df.copy()
            model.load(self.model_dir,encoder_type=self.encoder_type)
            # Low and high response values
            y_score, y_pred,df = model.predict(X)
            # Model prediction probabilities and scores
            model_pred_df.index = df.index
            model_pred_df['probs'] = y_score
            model_pred_df= model_pred_df[['probs']].copy()
            #probability distribution
            prob_distplot(model_pred_df,var_name='probs',bins=10)

            cutoff_high_response, cutoff_medium_response,_,_ = output_list_by_model(os.path.join(self.cache_dir,model.model_type+'_test_report.csv'),min_lift=actual_min_lift)
            cut_bins = [-float('inf'),cutoff_medium_response,cutoff_high_response,float('inf')]
            cut_labels = ['低响应','中响应','高响应']

            if cutoff_medium_response == cutoff_high_response:
                cut_bins = [-float('inf'),cutoff_high_response,float('inf')]
                cut_labels = ['低响应', '高响应']

            model_pred_df['response_label'] = pd.cut(model_pred_df['probs'],bins=cut_bins,labels=cut_labels)

            #model report
            stat_model_pred_df=model_pred_df.copy().reset_index()
            stat_model_pred_df.columns = ['客户ID','概率分数','响应级别']
            total=stat_model_pred_df.shape[0]

            model_pred_group_by = stat_model_pred_df[['响应级别','客户ID']].groupby('响应级别').count()
            model_pred_group_by['占比'] = round(model_pred_group_by['客户ID']/total,2)
            model_pred_group_by.columns = ['响应数','占比']
            logger.info(model_pred_group_by[['响应数','占比']].head())
            model_pred_group_by.to_csv(os.path.join(self.model_dir,'grp_cst_score_list.csv'),encoding='utf8',index=False)

            if self.cst_base_info_dataset is not  None:
                try:
                    model_df=model_pred_df.copy()
                    cst_base_info_df = self.load_data(self.cst_base_info_dataset,mode='list',source='list')

                    model_new_df = model_df.reset_index()
                    model_new_df[self.index_col] = model_new_df[self.index_col].astype(int)
                    # cst_base_info_new_df = cst_base_info_df.reset_index()
                    cst_base_info_new_df = cst_base_info_df
                    # cst_base_info_new_df[self.index_col] = cst_base_info_new_df[self.index_col].apply(lambda x:round(x,0))
                    cst_base_info_new_df[self.index_col] = cst_base_info_new_df[self.index_col].astype(int)
                    # cst_base_info_new_df.set_index(self.index_col,inplace=True)
                    model_pred_df = pd.merge(model_new_df,cst_base_info_new_df,on=self.index_col,how='left')
                    # logger.info(cst_base_info_new_df.head())
                    # logger.info(model_pred_df.head())
                except Exception as err:
                    logger.error('please check the index from cst_base_info,{}.'.format(err))
            else:
                model_pred_df = model_pred_df.reset_index()

            try:
                usr_profile_df =pd.read_csv(os.path.join(self.cache_dir,'cluster_usr_profile.csv'),encoding='utf8')
                logger.info('load user profile.')
                # logger.info(usr_profile_df.head().T)
                model_pred_df[self.index_col] = model_pred_df[self.index_col].astype(int)
                usr_profile_df[self.index_col] = usr_profile_df[self.index_col].astype(int)
                model_pred_df = pd.merge(model_pred_df,usr_profile_df,on=self.index_col,how='left')
                logger.info(model_pred_df.head(3).T)
            except Exception as err:
                logger.info(err)

            #write to database
            model_pred_df.to_csv(os.path.join(self.model_dir,'cst_score_list.csv'),encoding='utf8',index=False)
            logger.info('output list:')
            logger.info(model_pred_df.head())
            logger.info('write output list to databse!')
            if self.out_dataset is not None:
                self.out_dataset.write_by_df(model_pred_df)
            return cst_base_info_df,model_df,model_pred_df

    def user_cluster(self, model_type='lgb',gmm_need_retrain=False,model_size=7,encoder_type='onehot'):
        """
        用户分群
        :param model_type:
        :param gmm_need_retrain:
        :param model_size:
        :return:
        """
        X,pred_df, fea_df = self.predict_feature_process(encoder_type=encoder_type)
        self.encoder_type = encoder_type
        cluster_df =fea_df


        models,gmm = gmm_train(model_dir=self.model_dir,X=X,gmm_need_retrain=gmm_need_retrain,model_size=model_size)

        bic_list = [m.bic(X) for m in models]
        aic_list = [m.aic(X) for m in models]

        n_components = np.arange(3, 3+len(models))

        plt.plot(n_components, bic_list, label='BIC')
        plt.plot(n_components, aic_list, label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()

        bic_df = pd.DataFrame(bic_list, columns=['BIC'])
        logger.info(bic_df.head())

        best_cluster_length = bic_df['BIC'].sort_values(ascending=True).index[0]
        gmm = models[best_cluster_length]
        pred_labels = gmm.predict(X)

        X_n_sparse = csr_matrix(X)
        n_components = 2
        svd = TruncatedSVD(n_components=n_components)
        x_n_reduced = svd.fit_transform(X_n_sparse)

        plt.figure(figsize=(16,10))
        plt.scatter(x_n_reduced[:,0],x_n_reduced[:,1],c=pred_labels,s=40,cmap='viridis')
        plt.show()

        cluster_df['cluster'] = pred_labels

        if model_type is not  None:
            filter_model = [model for model in self.models if model.model_type in model_type][0]
        else:
            return 'input model_type please.'

        filter_model.load(self.model_dir,encoder_type=self.encoder_type)
        y_score,y_pred,df = filter_model.predict(X)
        cluster_df['label'] = y_pred
        cluster_df.reset_index().to_csv(os.path.join(self.cache_dir,'user_group_pred.csv'),encoding='utf8')
        return cluster_df

    def cal_tgi(self,cluster_df,tgi_threshold=1.05):
        """
        计算TGI
        :param cluster_df:
        :param tgi_threshold:
        :return:
        """
        _, _, _, cat_features, num_features = feature_classify(cluster_df, target=self.y_target)

        #tgi for numbers
        cluster_mean_df = cluster_df.groupby(['cluster'])[num_features].mean()
        num_tgi_df = cluster_mean_df/cluster_df[num_features].mean()
        logger.info('tgi for number variable')
        logger.info(num_tgi_df.head())
        #tgi for category numbers
        cat_tar_features = cat_features+['label','cluster']
        cat_df = cluster_df[cat_tar_features]

        #category distribution  for every target user
        target_group_ratio = cat_df[cat_df['label']==1].groupby(['cluster'])[cat_features].count()/cat_df[cat_df['label']==1].shape[0]
        # category distribution  for total target user
        total_group_ratio = cat_df.groupby('cluster')[cat_features].count()/cat_df.shape[0]

        total_ratio = cat_df.shape[0]/cat_df.shape[0]
        cat_tgi_df = (target_group_ratio/total_group_ratio)/total_ratio
        logger.info('tgi for category variable.')
        tgi_df = pd.merge(num_tgi_df,cat_tgi_df,left_index=True,right_index=True,how='left')
        tgi_df = tgi_df[self.model_features]
        self.load_data_map()
        rename_dict = {}
        for col in self.model_features:
            if col in self.data_map:
                rename_dict[col] = self.data_map[col]
            else:
                rename_dict[col] = col
        tgi_columns = []
        for inx in range(len(list(tgi_df.index))):
            cluster_nm = inx
            tgi_columns.append(cluster_nm)

        new_tgi_df = tgi_df.T
        new_tgi_df.colulmns = tgi_columns
        logger.info(new_tgi_df.head())

        plt.figure(figsize=(16,10))
        sns.heatmap(tgi_df,annot=True,cmap='RdBu_r')
        plt.show()
        cluster_usr_profile_df = pd.DataFrame(columns=['cluster','profile_features'])
        cluster_usr_profile_list = []

        for col in list(new_tgi_df.columns):
            logger.info('#################################################')
            cluster_tgi_df = new_tgi_df[new_tgi_df[col] > tgi_threshold].sort_values(by=col,ascending=False)
            logger.info(cluster_tgi_df[col])
            target_features = list(cluster_tgi_df[col].index)

            #caculate jaccard similarity
            best_keywords = ''
            best_sim = 0.0
            for rule in self.user_profile_label_rules:
                keywords,label_words = rule[0],rule[1]
                sim = jaccard_similarity(set(keywords),set(target_features))
                logger.info(sim)

                if sim >= best_sim:
                    best_sim = sim
                    best_keywords = ','.join(label_words)

            if best_sim < 1.1 and best_sim > 0.6:
                cluster_usr_profile_list.append([col,best_keywords])
            logger.info('#################################################')
            logger.info(cluster_usr_profile_list)

            if len(cluster_usr_profile_list) >0:
                #user profile labelling finished.
                cluster_usr_profile_df = pd.DataFrame(cluster_usr_profile_list,columns=['cluster','profile_features'])
                cluster_usr_profile_df.to_csv(os.path.join(self.cache_dir,'group_profile.csv'),encoding='utf8')
                logger.info('user profile labelling finished')
                logger.info(cluster_usr_profile_df.head())

            new_tgi_df.reset_index().to_csv(os.path.join(self.cache_dir,'user_importance_feature_tgi.csv'),encoding='utf8')
            return new_tgi_df,num_tgi_df,cat_tgi_df,cluster_usr_profile_df

    def user_profile(self,model_type='lgb',gmm_need_retrain=False,model_size=7,tgi_threshold=1.05):
        """
        用户画像
        :param model_type:
        :param gmm_need_retrain:
        :param model_size:
        :param tgi_threshold:
        :return:
        """
        cluster_df = self.user_cluster(model_type='lgb',gmm_need_retrain=gmm_need_retrain,model_size=model_size)
        cluster_df_c =cluster_df.copy().reset_index()
        new_tgi_df,num_tgi_df,cat_tgi_df,cluster_usr_profile_df = self.cal_tgi(cluster_df,tgi_threshold=tgi_threshold)
        cluster_df_c['cluster'] = cluster_df_c['cluster'].astype(int)
        cluster_usr_profile_df['cluster'] = cluster_usr_profile_df['cluster'].astype(int)
        cluster_df_c = pd.merge(cluster_df_c,cluster_usr_profile_df,on='cluster',how='left')
        logger.info(cluster_df_c.head())
        cluster_df_c[[self.index_col,'cluster','profile_features']].to_csv(os.path.join(self.cache_dir,'cluster_usr_profile.csv'),encoding='utf8',index=True)
        return new_tgi_df,num_tgi_df,cat_tgi_df,cluster_df_c

    def make_scored_card(self,Basepoints=600,PDO=20,odds0=2,encoder_type=None):
        '''
        评分卡制作
        :param self:
        :param Basepoints:
        :param PDO:
        :param odds0:
        :return:
        '''
        model_features = load_all_model_features(self.model_dir)
        if encoder_type is not None:
            self.encoder_type = encoder_type

        lr_model = self.load_model(model_type='lr',encoder_type=self.encoder_type)
        return lr_scored_card(lr_model, self.model_dir, model_features, Basepoints=Basepoints, PDO=PDO,
                       odds0=odds0)


    def model_report(self,model_type='lgb',flag='test'):
        '''
        模型报告
        :param self:
        :param model_type:
        :param flag:
        :return:
        '''
        df = pd.read_csv(os.path.join(self.cache_dir, model_type + '_' + flag + '_report.csv'), encoding='utf8')
        return df


    def load_model(self,model_type='lgb',encoder_type='woe'):
        #加载模型
        best_model = pickle.load(open(os.path.join(self.model_dir,encoder_type+'_'+model_type+'_model.pickle'),'rb'))
        return best_model


    def feature_distribution(self,df,plot_type='all'):
        plot_distribuion(df,y_target=self.y_target,plot_type=plot_type)

