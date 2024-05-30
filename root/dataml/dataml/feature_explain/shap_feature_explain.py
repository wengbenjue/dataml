from .base_feature_explain import BaseFeatureExplain
import shap
import matplotlib.pyplot as plt


class ShapFeatureExplain(BaseFeatureExplain):

    def __init__(self):
        pass

    def var_explainModel_fit(self,model_fit, X, Y, data_dict=None):
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

        # explainer = shap.TreeExplainer(model_best_modele.best_estimator_, masker=masker)
        explainer = shap.TreeExplainer(model_fit)
        shap_values = explainer.shap_values(X, Y)

        # shp.summary_plot(shp_values[0], show=False, plot_type='bar')
        plt.figure(figsize=(10, 6), constrained_layout=True)
        shap.summary_plot(shap_values[1], X, feature_name=columns)
        plt.show()
