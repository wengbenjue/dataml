import numpy as np
import os
import pickle
from catboost import CatBoostClassifier
from typing import List

from ..util.utils import lgbm_evaluate
from ..util.utils import model_predict,model_predict_eval,model_test_eval,show_feature_importance
from .base_model import BaseModel
from ..feature_explain import BaseFeatureExplain
from ..util.my_logging import logger
from ..util.utils import feature_classify


class CatBoostModel(BaseModel):

    def __init__(self, explain_models: List[BaseFeatureExplain],
                 params_optimize=False,
                 name='CatBoost模型',
                 var_en_name=None,
                 sheet_name='Sheet1',
                 var_zh_name=None,params=None):
        self.explain_models = explain_models
        self.params_optimize = params_optimize
        self.model_type = 'catboost'
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
            self.params = { 'iterations':100, 'learning_rate':0.05,'depth':7,'class_weights': [1, 5]}

        self.model = CatBoostClassifier(iterations=self.params['iterations'],
                                        depth=self.params['depth'],
                                        learning_rate=self.params['learning_rate'],
                                        cat_features=[0, 1],
                                        class_weights=self.params['class_weights'],
                                        verbose=0)

    def load(self, model_dir,encoder_type=None):
        if encoder_type is None:
            encoder_type = 'none'
        #加载模型
        self.model_fit = pickle.load(open(os.path.join(model_dir,encoder_type+'_catboost_model.pickle'),'rb'))

    def model_feature_importance(self, X,data_dir_path,iv_df=None, data_dir='.'):
        feature_imp,data_map = show_feature_importance(self.model_fit,cols=list(X.columns),
                                                  data_dict_path=data_dir_path,
                                                  right_on_name=self.var_en_name,
                                                  sheet_name=self.sheet_name,
                                                  var_zh_name=self.var_zh_name)

        if iv_df is not  None:
            feature_imp = feature_imp.merge(iv_df,how='left',left_on=self.var_en_name,right_on='variable')
        logger.info('输出特征重要性')
        feature_imp[[self.var_en_name,self.var_zh_name,'feature_importance','IV']].to_csv(os.path.join(data_dir,'catboost_feature_importance.csv'))
        logger.debug(feature_imp[[self.var_en_name,self.var_zh_name,'feature_importance','IV']].head(30))
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

    def train(self, X, y):
        self.hyperparams_search(X,y)
        # self.model_fit = self.model.fit(X,y)
        _, _, _, cat_features, num_features = feature_classify(X, target=y.name)
        self.model = CatBoostClassifier(iterations=self.params['iterations'],
                                        depth=self.params['depth'],
                                        learning_rate=self.params['learning_rate'],
                                        cat_features=cat_features,
                                        class_weights=self.params['class_weights'],
                                        verbose=0)
        self.model.fit(X,y)
        self.model_fit = self.model

    def predict(self, X):
        if self.model_fit is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score,y_pred,df = model_predict(self.model_fit,X)
            return y_score,y_pred,df

    def persist(self, model_dir,encoder_type='none'):
        if encoder_type is None:
            encoder_type = 'none'
        #保存模型
        with open(os.path.join(model_dir,encoder_type+'_catboost_model.pickle'),'wb') as f:
            # pickle.dump(self.model_fit,f)
            pickle.dump(self.model,f)
            logger.debug('CatBoost模型保存成功！')

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'):
        logger.debug(self.name)
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
                                                               model_type='catboost',
                                                               flag=flag,
                                                               index_col=index_col
                                                               )
            return y_score_test,y_pred_test,test_df

    def evaluate(self,
                 X,
                 X_test,
                 y,
                 y_test,
                 y_target,
                 cache_dir=None,
                 index_col='cst_id'):
        logger.info(self.name)
        if self.model_fit is None:
            logger.debug('请先使用self.load(model_dir)加载模型！')
            return None,None,None
        else:
            _, _, _, cat_features, num_features = feature_classify(X, target=y.name)
            _, _, _, cat_features_test, num_features_test = feature_classify(X_test, target=y_test.name)
            logger.debug('train category features:{}'.format(','.join(cat_features)))
            logger.debug('test category features:{}'.format(','.join(cat_features_test)))
            y_score_test,y_pred_test,test_df=model_predict_eval(self.model_fit,
                                                                X,
                                                                X_test,
                                                                y,
                                                                y_test,
                                                                y_target,
                                                                pos_label=1,
                                                                cache_dir=cache_dir,
                                                                model_type='catboost',
                                                                index_col=index_col)
            return y_score_test,y_pred_test,test_df

    def hyperparams_search(self,X,y):

        if self.params_optimize:
            pass

