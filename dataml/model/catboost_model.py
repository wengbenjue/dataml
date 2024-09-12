import numpy as np
import os
import pickle
from catboost import CatBoostClassifier,Pool
from typing import List

from ..util.utils import lgbm_evaluate
from ..util.utils import model_predict,model_predict_eval,model_test_eval,show_feature_importance
from .base_model import BaseModel
from ..feature_explain import BaseFeatureExplain
from ..util.my_logging import logger
from ..util.utils import feature_classify
import pandas as pd


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
            self.params = { 'iterations':100, 'learning_rate':0.05,
                            'depth':7,
                            'class_weights': [1, 5],'eval_metric':'AUC'}

        self.model = CatBoostClassifier(iterations=self.params['iterations'],
                                        depth=self.params['depth'],
                                        learning_rate=self.params['learning_rate'],

                                        class_weights=self.params['class_weights'],
                                        verbose=0,
                                        eval_metric=self.params['eval_metric'],
                                        use_best_model=True,
                                        random_seed=42
                                        )

    def load(self, model_dir, encoder_type='none'):
        if encoder_type is None:
            encoder_type = 'none'
        # 加载模型
        with open(os.path.join(model_dir,'catboost_cate_features.pickle'),'rb') as f:
            self.cat_features = pickle.load(f)
            logger.info('catboost cat_features load success.')

        with open(os.path.join(model_dir, 'catboost_cate_categories.pickle'), 'rb') as f:
            self.cat_categories_dict =  pickle.load(f)
            logger.info('catboost cate_categories load success.')

        model_path = os.path.join(model_dir, encoder_type + '_catboost_model.json')
        # self.model = CatBoostClassifier(cat_features=self.cat_features)
        self.model = CatBoostClassifier(iterations=self.params['iterations'],
                                        depth=self.params['depth'],
                                        learning_rate=self.params['learning_rate'],
                                        cat_features=self.cat_features,
                                        class_weights=self.params['class_weights'],
                                        eval_metric=self.params['eval_metric'],
                                        # use_best_model=True,
                                        random_seed=42,
                                        verbose=0)
        self.model.load_model(model_path,format='json')
        self.model_fit = self.model


        print(self.model.get_params())



        logger.debug('CatBoost模型加载成功！')


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
        if self.explain_models is None:
            logger.info('没有特征解释器！')
        else:
            for explain_model in self.explain_models:
                logger.debug(explain_model)
                explain_model.var_explain(model_fit=self.model_fit,X=X_test,y=y_test,data_dict=data_map,exlpain_type='tree')

    def train(self, X, y,encoder_type='onehot_encoder'):
        self.hyperparams_search(X,y)
        # self.model_fit = self.model.fit(X,y)
        X.replace('Unknown', pd.NA, inplace=True)
        _, _, _, self.cat_features, self.num_features = feature_classify(X, target=y.name)

        self.cat_categories_dict = {}
        for col in self.cat_features:
            X[col] = X[col].astype('category')
            self.cat_categories_dict[col] = X[col].cat.categories
            # X[col] = X[col].cat.codes  # 类别特征编码


        self.model = CatBoostClassifier(iterations=self.params['iterations'],
                                        depth=self.params['depth'],
                                        learning_rate=self.params['learning_rate'],
                                        cat_features=self.cat_features,
                                        class_weights=self.params['class_weights'],
                                        eval_metric=self.params['eval_metric'],
                                        # use_best_model=True,
                                        random_seed=42,
                                        verbose=0)
        self.model.fit(X,y)
        self.model_fit = self.model
        print(self.model.get_params())

    def predict(self, X):
        print(self.model.get_params())
        X.replace('Unknown', pd.NA, inplace=True)
        if self.model is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score,y_pred,df = model_predict(self.model,X)
            return y_score,y_pred,df

    def persist(self, model_dir, encoder_type='none'):
        if encoder_type is None:
            encoder_type = 'none'
        # 保存模型
        model_path = os.path.join(model_dir, encoder_type + '_catboost_model.json')
        self.model.save_model(model_path,format='json')
        print(self.model.get_params())

        with open(os.path.join(model_dir,'catboost_cate_features.pickle'),'wb') as f:
            pickle.dump(self.cat_features,f)
            logger.info('catboost cat_features persist success.')

        with open(os.path.join(model_dir, 'catboost_cate_categories.pickle'), 'wb') as f:
            pickle.dump(self.cat_categories_dict, f)
            logger.info('catboost cate_categories persist success.')

        logger.debug('CatBoost模型保存成功！')

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'
                      ,encoder_type='onehot'):
        logger.debug(self.name)
        print(self.model.get_params())  # 查看模型参数

        if encoder_type is None:
            encoder_type = 'none'
        # 处理未知值
        X_test.replace('Unknown', pd.NA, inplace=True)

        # 获取分类和数值特征
        _, _, _, cat_features, num_features = feature_classify(X_test, target=y_target)

        # 检查类别特征是否在测试集中
        for col in cat_features:
            if col not in X_test.columns:
                raise ValueError(f"Feature '{col}' is in cat_features but is missing in the test set.")
            X_test[col] = X_test[col].astype('category')  # 将类别特征转换为 'category' 类型

        # 确保测试集的列名与训练集一致
        assert all(col in X_test.columns for col in cat_features)
        print("Test cat features: ", cat_features)
        print("Model cat features: ", self.cat_features)

        # 数值特征处理，尝试将其转换为数值类型
        for col in num_features:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

        # 处理类别特征中未见过的类别
        for col in self.cat_features:
            if col in cat_features:
                X_test[col] = X_test[col].astype('category')  # 确保是类别特征

                # 获取训练集的类别
                train_categories = self.cat_categories_dict.get(col, [])
                # 创建新的 Index 对象，并添加 'unknown'
                if 'unknown' not in train_categories:
                    train_categories = list(train_categories) + ['unknown']
                train_categories_index = pd.Index(train_categories)

                X_test[col] = X_test[col].cat.set_categories(train_categories_index)
                # X_test[col] = X_test[col].fillna('unknown')  # 填充缺失值为 'unknown'
                X_test[col] = X_test[col].fillna('unknown')  # 填充缺失值为 'unknown'
                # X_test[col] = X_test[col].cat.add_categories(['unknown']).fillna('unknown')
               #类别特征编码
                # X_test[col] = X_test[col].cat.codes  # 类别特征编码

        # 检查模型是否存在
        if self.model is None:
            logger.debug('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            # 进行模型预测和评估
            y_score_test, y_pred_test, test_df = model_test_eval(self.model,
                                                                 X_test,
                                                                 y_test,
                                                                 y_target=y_target,
                                                                 pos_label=1,
                                                                 cache_dir=cache_dir,
                                                                 model_type='catboost',
                                                                 flag=flag,
                                                                 index_col=index_col,
                                                                 encoder_type=encoder_type
                                                                 )
            return y_score_test, y_pred_test, test_df

    def evaluate(self,
                 X,
                 X_test,
                 y,
                 y_test,
                 y_target,
                 cache_dir=None,
                 index_col='cst_id'
                 ,encoder_type='none'):
        logger.info(self.name)
        # X.replace('Unknown', pd.NA, inplace=True)
        # X_test.replace('Unknown', pd.NA, inplace=True)
        if encoder_type is None:
            encoder_type = 'none'
        if self.model is None:
            logger.debug('请先使用self.load(model_dir)加载模型！')
            return None,None,None
        else:
            _, _, _, cat_features, num_features = feature_classify(X, target=y.name)
            _, _, _, cat_features_test, num_features_test = feature_classify(X_test, target=y_test.name)
            logger.debug('train category features:{}'.format(','.join(cat_features)))
            logger.debug('test category features:{}'.format(','.join(cat_features_test)))
            y_score_test,y_pred_test,test_df=model_predict_eval(self.model,
                                                                X,
                                                                X_test,
                                                                y,
                                                                y_test,
                                                                y_target,
                                                                pos_label=1,
                                                                cache_dir=cache_dir,
                                                                model_type='catboost',
                                                                index_col=index_col,
                                                                encoder_type=encoder_type)
            return y_score_test,y_pred_test,test_df

    def hyperparams_search(self,X,y):

        if self.params_optimize:
            pass

