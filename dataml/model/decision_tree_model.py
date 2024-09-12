import numpy as np
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from typing import List
from sklearn import tree
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

from ..util.utils import model_predict,model_predict_eval,model_test_eval,show_feature_importance
from .base_model import BaseModel
from ..feature_explain import BaseFeatureExplain
from ..util.my_logging import logger


class DecisionTreeModel(BaseModel):

    def __init__(self, explain_models: List[BaseFeatureExplain],
                 params_optimize=False,
                 name='决策树模型',
                 var_en_name=None,
                 sheet_name='Sheet1',
                 var_zh_name=None,
                 params=None,
                 class_names=['0_negative','1_positive']):
        self.explain_models = explain_models
        self.params_optimize = params_optimize
        self.model_type = 'decision_tree'
        self.name = name
        self.var_en_name = var_en_name
        self.var_zh_name = var_zh_name
        self.sheet_name = sheet_name
        self.params = params
        self.class_names = class_names

        if self.var_en_name is None:
            self.var_en_name = 'index'
        if self.var_zh_name is None:
            self.var_zh_name = 'index'

        if self.params is None:
            self.params = {
            'max_depth':3,
            'criterion':'gini', #entropy  gini
            'splitter':'best',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'random_state': None,
            'max_leaf_nodes': None,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'min_impurity_decrease':0.0,
            'min_impurity_split':None,
            'min_weight_fraction_leaf':0.0,
            'presort':False
            }

        self.model  = DecisionTreeClassifier(max_depth=self.params['max_depth'],
                                             criterion=self.params['criterion'],
                                             class_weight=self.params['class_weight'],
                                             max_features=self.params['max_features'],
                                             max_leaf_nodes=self.params['max_leaf_nodes'],
                                             min_impurity_decrease=self.params['min_impurity_decrease'],
                                             min_impurity_split=self.params['min_impurity_split'],
                                             min_samples_leaf=self.params['min_samples_leaf'],
                                             min_samples_split=self.params['min_samples_split'],
                                             min_weight_fraction_leaf=self.params['min_weight_fraction_leaf'],
                                             random_state=42,
                                             splitter=self.params['splitter'])

    def load(self, model_dir,encoder_type='onehot_encoder'):
        #加载模型
        self.model_fit = pickle.load(open(os.path.join(model_dir,encoder_type+'_decision_model.pickle'),'rb'))

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
            logger.debug(feature_imp[[self.var_en_name, self.var_zh_name, 'feature_importance', 'IV']].head(30))
        return  feature_imp,data_map


    def feature_explain(self, X_test, y_test, data_map=None):
        """
        特征敏感性分析
        :param X_test:
        :param y_test:
        :param data_map: 中英文数据字典映射
        :return:
        """

        feature_names = [str(col) for col in X_test.columns]
        text_representation = tree.export_text(self.model_fit, feature_names=feature_names,max_depth=self.params['max_depth'])
        logger.debug(text_representation)

        plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(self.model_fit,
                           feature_names=X_test.columns,
                           class_names=self.class_names,
                           filled=True)
        plt.show()

        if self.explain_models is not None:
            for explain_model in self.explain_models:
                logger.debug(explain_model)
                explain_model.var_explain(model_fit=self.model_fit,X=X_test,y=y_test,data_dict=data_map,exlpain_type='tree')

    def train(self, X, y,encoder_type='onehot_encoder'):
        self.hyperparams_search(X,y)
        # self.model_fit = self.model.fit(X,y)
        self.model.fit(X, y)
        self.model_fit = self.model

    def predict(self, X):
        if self.model_fit is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score,y_pred,df = model_predict(self.model_fit,X)
            return y_score,y_pred,df

    def persist(self, model_dir,encoder_type='onehot_encoder'):
        #保存模型
        with open(os.path.join(model_dir,encoder_type+'_decision_model.pickle'),'wb') as f:
            # pickle.dump(self.model_fit,f)
            pickle.dump(self.model,f)
            logger.debug('决策树模型保存成功！')

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'
                      ,encoder_type='none'):
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
                                                               model_type='decision_tree',
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
        logger.info(self.name)
        if encoder_type is None:
            encoder_type = 'none'
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
                                                                model_type='decision_tree',
                                                                index_col=index_col,
                                                                encoder_type=encoder_type)
            return y_score_test,y_pred_test,test_df

    def hyperparams_search(self,X,y):

        pass

