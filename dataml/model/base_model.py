from typing import List
from ..feature_explain import BaseFeatureExplain


class BaseModel:
    def __init__(self, explain_models: List[BaseFeatureExplain]):
        self.explain_models = explain_models

    def load(self, model_dir,encoder_type='onehot_encoder'):
        pass

    def model_feature_importance(self, X,data_dir_path,iv_df, data_dir):
        pass

    def feature_explain(self, X_test, y_test, data_map=None):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def persist(self, model_dir,encoder_type='onehot_encoder'):
        pass

    def evaluate_test(self,
                      X_test,
                      y_test,
                      y_target,
                      cache_dir=None,
                      flag='test',
                      index_col='cst_id'):
        pass

    def evaluate(self,
                 X,
                 X_test,
                 y,
                 y_test,
                 y_target,
                 cache_dir=None,
                 index_col='cst_id'):
        pass

    def hyperparams_search(self,X,y):
        pass
