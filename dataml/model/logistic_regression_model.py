import os
import pickle
import numpy as np
from typing import List
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from ..feature_explain import BaseFeatureExplain
from ..util.utils import model_predict,model_predict_eval,model_test_eval,ab,load_sc_bins,scorecard,load_all_model_features,lr_scored_card
from  .base_model import BaseModel
from ..util.my_logging import logger
class LogisticRegressionModel(BaseModel):
    def __init__(self, explain_models: List[BaseFeatureExplain],
                 class_weight='balanced',max_iter=200,C=0.5,class_weights=[{1: 1.0,0: 1.0}, {1:2.0,0:1.0}],
                 params_optimize=False,Basepoints=600,PDO=20,use_woe=False,model_dir=None):
        self.explain_models = explain_models
        self.class_weight = class_weight
        self.class_weights = class_weights
        self.max_iter = max_iter
        self.C = C
        self.model_type = 'lr'
        self.name = '逻辑回归模型'

        self.Basepoints = Basepoints
        self.PDO = PDO
        self.model_dir = model_dir
        self.use_woe = use_woe


        self.params_optimize = params_optimize
        self.params_new = {}
        best_params = {'C': self.C,
                'penalty': 'l1',
                'solver': 'liblinear',
                'class_weight':self.class_weight,
                'max_iter': self.max_iter}
        self.params_new.update(best_params)
        logger.info(self.params_new)

        self.model = LogisticRegression(random_state=0,C=self.params_new['C'],penalty=self.params_new['penalty'],
solver=self.params_new['solver'],class_weight=self.params_new['class_weight'])

    def load(self, model_dir,encoder_type='onehot_encoder'):
        self.model_fit = pickle.load(open(os.path.join(model_dir,encoder_type+'_lr_model.pickle'),'rb'))
        self.model_features = load_all_model_features(model_dir)

    def make_scored_card(self,y,odds0=None):
        '''
        评分卡制作
        :param y:
        :param odds0:
        :return:
        '''
        if odds0 is None:
            odds0 = y.value_counts()[1] / y.value_counts()[0]  # 好坏比
        return lr_scored_card(self.model_fit, self.model_dir, self.model_features, Basepoints=self.Basepoints, PDO=self.PDO, odds0=odds0)

    def model_feature_importance(self, X,data_dir_path,iv_df=None, data_dir='.'):
        lr_coef = self.model_fit.coef_
        logger.info('输出特征重要性')
        sorted_inx = np.argsort(-lr_coef)
        print('变量名：{}'.format(X.columns[sorted_inx[-5:]]))
        print('系数：{}'.format(-np.sort(-lr_coef)))
        print('截距：{}'.format(self.model_fit.intercept_))
        return None,None

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
                logger.info(explain_model)
                explain_model.var_explain(model_fit=self.model_fit, X=X_test, y=y_test, data_dict=data_map,exlpain_type='linear')

    def train(self, X, y):
            self.hyperparams_search(X,y)
            self.model_fit = self.model.fit(X, y)
            #output score card

            if self.use_woe:
                self.model_features = load_all_model_features(self.model_dir)
                self.make_scored_card(y=y)

    def predict(self, X):
        if self.model_fit is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score, y_pred, df = model_predict(self.model_fit, X)
            return y_score, y_pred, df

    def persist(self, model_dir,encoder_type='onehot_encoder'):
        # 保存模型
        with open(os.path.join(model_dir, encoder_type+'_lr_model.pickle'), 'wb') as f:
            pickle.dump(self.model_fit, f)
            logger.info('逻辑回归模型保存成功！')

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'):
        logger.info(self.name)
        if self.model_fit is None:
            logger.info('please load model first using self.load(model_dir)')
            return None, None, None
        else:
            y_score_test, y_pred_test, test_df = model_test_eval(self.model_fit,
                                                                 X_test,
                                                                 y_test,
                                                                 y_target=y_target,
                                                                 pos_label=1,
                                                                 cache_dir=cache_dir,
                                                                 model_type='lr',
                                                                 flag=flag,
                                                                 index_col=index_col)
            return y_score_test, y_pred_test, test_df

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
            logger.info('请先使用self.load(model_dir)加载模型！')
            return None, None, None
        else:
            y_score_test, y_pred_test, test_df = model_predict_eval(self.model_fit,
                                                                    X,
                                                                    X_test,
                                                                    y,
                                                                    y_test,
                                                                    y_target,
                                                                    pos_label=1,
                                                                    cache_dir=cache_dir,
                                                                    model_type='lr',
                                                                    index_col=index_col)
            return y_score_test, y_pred_test, test_df

    def hyperparams_search(self,X,y):
        if self.params_optimize:
            lr_param = {'C': [0.5, 1, 2], 'class_weight':self.class_weights,
                        'solver': ['liblinear'], 'max_iter': [100, 200]}

            # solver=lbfgs, newton-cg, liblinear, saga
            lr_gsearch = GridSearchCV(
                estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='liblinear',
                                             class_weight={1: 2, 0: 1}), param_grid=lr_param, cv=3, scoring='f1_macro',
                n_jobs=-1, verbose=1)
            lr_gsearch.fit(X, y)
            logger.info("逻辑回归 best_score_为：{}，最好的参数为：{}".format(lr_gsearch.best_score_, lr_gsearch.best_params_))
            self.params_new.update(lr_gsearch.best_params_)
            self.model = LogisticRegression(random_state=0,C=self.params_new['C'],penalty=self.params_new['penalty'],
solver=self.params_new['solver'],class_weight=self.params_new['class_weight'])

