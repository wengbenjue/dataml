from .base_feature_explain import BaseFeatureExplain
import shap
import matplotlib.pyplot as plt
import pandas  as pd
from ..util.my_logging import logger

class ShapFeatureExplain(BaseFeatureExplain):

    def __init__(self):
        pd.set_option("display.max_columns", None)
        shap.initjs()

    def var_explain(self, model_fit, X, y, data_dict=None, exlpain_type='tree'):
        """
        特征敏感性分析
        :param X:
        :param Y:
        :param data_dict:
        :return:
        """
        zh_features_names = []
        columns = list(X.columns)
        if data_dict is not None:
            for col in columns:
                if col in data_dict:
                    zh_features_names.append(data_dict[col].strip())
                else:
                    zh_features_names.append(col.strip())
            columns = zh_features_names

        if  'tree' in exlpain_type:
            # explainer = shap.TreeExplainer(model_best_modele.best_estimator_, masker=masker)
            explainer = shap.TreeExplainer(model_fit,feature_perturbation="interventional")
            shapvalues = explainer.shap_values(X=X,y=y)
            try:
                shap.summary_plot(shapvalues[1], X, feature_names=columns)
            except:
                try:
                    shap.summary_plot(shapvalues, X, feature_names=columns)
                except Exception as e:
                    logger.error(e)
        else:
            # explainer = shap.LinearExplainer(model_fit, X, feature_perturbation="interventional")
            masker = shap.maskers.Independent(data=X)
            explainer = shap.LinearExplainer(model_fit,masker=masker, feature_perturbation="interventional")
            # explainer = shap.Explainer(
            #     model_fit,seed=43,masker=shap.maskers.Impute(), feature_names=columns,algorithm='linear'
            # )
            shapvalues = explainer.shap_values(X=X)
            shap.summary_plot(shapvalues, X, feature_names=columns)


        # shp.summary_plot(shp_values[0], show=False, plot_type='bar')
        plt.figure(figsize=(10, 6), constrained_layout=True)
        # shap.force_plot(explainer.expected_value[1], shapvalues[2][2], X[3], feature_names=X.columns)
        plt.show()
