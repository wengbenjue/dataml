import numpy as np
import os
import pickle
import lightgbm as lgb
from typing import List

import pandas as pd

from ..util.utils import lgbm_evaluate
from ..util.utils import model_predict,model_predict_eval,model_test_eval,show_feature_importance
from .base_model import BaseModel
from ..feature_explain import BaseFeatureExplain
from ..util.my_logging import logger


class LightGBMModel(BaseModel):

    def __init__(self, explain_models: List[BaseFeatureExplain],
                 params_optimize=False,
                 name='LightGBM模型',
                 var_en_name=None,
                 sheet_name='Sheet1',
                 var_zh_name=None,params=None):
        self.explain_models = explain_models
        self.params_optimize = params_optimize
        self.model_type = 'lgb'
        self.name = name
        self.var_en_name = var_en_name
        self.var_zh_name = var_zh_name
        self.sheet_name = sheet_name
        self.params = params

        if self.var_en_name is None:
            self.var_en_name = 'index'
        if self.var_zh_name is None:
            self.var_zh_name = 'index'

        if self.params is None:
            self.params = {'application': 'binary', 'num_iterations':100, 'learning_rate':0.05, 'n_estimators':200, 'num_leaves':48, 'device':'cpu',
             'max_depth':7, 'boosting':'gbdt', 'objective':['binary'], 'max_bin':510, 'lambda_l1':5, 'lambda_l2':10, 'metric':'binary_error',
             'subsample_for_bin':200, 'random_state':500, 'colsample_bytree':0.53, 'subsample':0.8, 'min_split_gain':5, 'min_child_weight':0.01,
             'min_child_samples':313, 'reg_alpha': 1, 'reg_lambda':2, 'sumsample':0.88, 'scale_pos_weight':1.0}

        self.model = lgb.LGBMClassifier(
                                    n_jobs=-1
                                    ,learning_rate=self.params['learning_rate']
                                    ,n_estimators=self.params['n_estimators']
                                    ,num_leaves=self.params['num_leaves']
                                    ,max_depth=self.params['max_depth']
                                    ,random_state=42
                                    ,colsample_bytree=self.params['colsample_bytree']
                                    ,subsample=self.params['subsample']
                                    ,min_child_weight=self.params['min_child_weight']
                                    ,min_child_samples=self.params['min_child_samples']
                                    ,reg_alpha=self.params['reg_alpha']
                                    ,reg_lambda=self.params['reg_lambda']
                                    # ,sumsample=params['sumsample']
                                    ,scale_pos_weight=self.params['scale_pos_weight']
                                   )
    def load(self, model_dir,encoder_type='onehot_encoder'):
        if encoder_type is None:
            encoder_type = 'none'
        #加载模型
        self.model_fit = pickle.load(open(os.path.join(model_dir,encoder_type+'_lgb_model.pickle'),'rb'))
        self.model = self.model_fit

    def model_feature_importance(self, X,data_dir_path,iv_df=None, data_dir='.'):
        feature_imp,data_map = show_feature_importance(self.model_fit,cols=list(X.columns),
                                                  data_dict_path=data_dir_path,
                                                  right_on_name=self.var_en_name,
                                                  sheet_name=self.sheet_name,
                                                  var_zh_name=self.var_zh_name)

        if iv_df is not  None:
            feature_imp = feature_imp.merge(iv_df,how='left',left_on=self.var_en_name,right_on='variable')
        logger.info('输出特征重要性')
        try:
            feature_imp[[self.var_en_name,self.var_zh_name,'feature_importance','IV']].to_csv(os.path.join(data_dir,self.model_type+'_feature_importance.csv'))
            logger.debug(feature_imp[[self.var_en_name,self.var_zh_name,'feature_importance','IV']].head(30))
        except Exception as e:
            feature_imp[[self.var_en_name, self.var_zh_name, 'feature_importance']].to_csv(
                os.path.join(data_dir, self.model_type + '_feature_importance.csv'))
            logger.debug(feature_imp[[self.var_en_name, self.var_zh_name, 'feature_importance']].head(30))
        return  feature_imp,data_map


    def feature_explain(self, X_test, y_test, data_map=None):
        """
        特征敏感性分析
        :param X_test:
        :param y_test:
        :param data_map: 中英文数据字典映射
        :return:
        """
        if self.explain_models is None:
            logger.info('没有特征解释器！')
        else:
            for explain_model in self.explain_models:
                logger.debug(explain_model)
                explain_model.var_explain(model_fit=self.model_fit,X=X_test,y=y_test,data_dict=data_map,exlpain_type='tree')

    def train(self, X, y,encoder_type='onehot_encoder'):
        self.hyperparams_search(X,y)
        # self.model_fit = self.model.fit(X,y)
        if encoder_type is None or 'none' in encoder_type:
            cateorical_features = list(X.select_dtypes(include=pd.Categorical).columns)
            # for col in cateorical_features:
            #     X[col] = X[col].astype('category')  # 先转换为类别类型
            #     X[col] = X[col].cat.codes  # 然后使用 cat.codes 编码为整数
            #     X[col] = X[col].replace(-1, X[col].max() + 1)  # 若存在空值，替换为最大类别值+1
            # print(X[cateorical_features].dtypes)
            self.model.fit(X, y,categorical_feature=cateorical_features)
        else:
            self.model.fit(X,y)

        self.model_fit = self.model

    def predict(self, X):
        if self.model_fit is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score,y_pred,df = model_predict(self.model_fit,X)
            return y_score,y_pred,df

    def persist(self, model_dir,encoder_type='onehot_encoder'):
        if encoder_type is None:
            encoder_type = 'none'
        #保存模型
        with open(os.path.join(model_dir,encoder_type+'_lgb_model.pickle'),'wb') as f:
            # pickle.dump(self.model_fit,f)
            pickle.dump(self.model,f)
            logger.debug('LightGBM模型保存成功！')

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'
                      ,encoder_type='onehot'):
        logger.debug(self.name)
        if encoder_type is None:
            encoder_type = 'none'
        if self.model_fit is None:
            logger.debug('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score_test,y_pred_test,test_df = model_test_eval(self.model_fit,
                                                               X_test,
                                                               y_test,
                                                               y_target=y_target,
                                                               pos_label=1,
                                                               cache_dir=cache_dir,
                                                               model_type='lgb',
                                                               flag=flag,
                                                               index_col=index_col,
                                                               encoder_type=encoder_type
                                                               )
            return y_score_test,y_pred_test,test_df

    def evaluate(self,
                 X,
                 X_test,
                 y,
                 y_test,
                 y_target,
                 cache_dir=None,
                 index_col='cst_id'
                 ,encoder_type='onehot'):
        if encoder_type is None:
            encoder_type = 'none'
        logger.info(self.name)
        if self.model_fit is None:
            logger.debug('请先使用self.load(model_dir)加载模型！')
            return None,None,None
        else:
            y_score_test,y_pred_test,test_df=model_predict_eval(self.model_fit,
                                                                X,
                                                                X_test,
                                                                y,
                                                                y_test,
                                                                y_target,
                                                                pos_label=1,
                                                                cache_dir=cache_dir,
                                                                model_type='lgb',
                                                                index_col=index_col,
                                                                encoder_type=encoder_type)
            return y_score_test,y_pred_test,test_df

    def hyperparams_search(self,X,y):

        if self.params_optimize:
            params = {'colsample_bytree': (0.8, 1),
                      'learning_rate': (.01, .02),
                      'num_leaves': (20, 200),
                      'subsample': (0.8, 1),
                      'max_depth': (3, 10),
                      'reg_alpha': (.03, .05),
                      'reg_lambda': (.06, .08),
                      'min_split_gain': (.01, .03),
                      'min_child_weight': (20, 40),
                      'lambda_l2': (0.0, 10.0),
                      'lambda_l1': (0.0, 5.0),
                      'min_child_samples': (50, 1000)
                      }
            try:

                from bayes_opt import BayesianOptimization
                bo = BayesianOptimization(lgbm_evaluate(X,y),**params)
                bo.maximize(init_points=5,n_iter=5)
                self.params = bo.res['max']
            except Exception as e:
                pass
            self.model = lgb.LGBMClassifier(
                                    n_jobs=-1
                                    ,learning_rate=self.params['learning_rate']
                                    ,n_estimators=self.params['n_estimators']
                                    ,num_leaves=self.arams['num_leaves']
                                    ,max_depth=self.params['max_depth']
                                    ,random_state=0
                                    ,colsample_bytree=self.params['colsample_bytree']
                                    ,subsample=self.params['subsample']
                                    ,min_child_weight=self.params['min_child_weight']
                                    ,min_child_samples=self.params['min_child_samples']
                                    ,reg_alpha=self.params['reg_alpha']
                                    ,reg_lambda=self.params['reg_lambda']
                                    # ,sumsample=params['sumsample']
                                    ,scale_pos_weight=self.params['scale_pos_weight']
                                   )

