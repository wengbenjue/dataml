import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset


def stock_prediction(df, model_callback, start_predict_date,forecast_n=6, lag_features=None, target_variable='return',
                     drop_columns=['ts_code', 'trade_date'], need_grid_search=False, lagN=5, need_shap=False):
    """
    对股票数据进行预测的函数。

    :param df: 包含股票数据的DataFrame。
    :param model_callback: 用于预测的回归模型回调函数。
    :param start_predict_date: 开始进行预测的日期（字符串格式，如'2022-01-01'）。
    :param lag_features: 要创建滞后的特征列表。
    :param target_variable: 目标变量名称。
    :param need_grid_search: 是否需要进行网格搜索优化模型。
    :param lagN: 滞后特征的数量。
    :return: 包含模型、预测和评估结果的字典。
    """

    models = {}
    predictions = {}
    evaluations = {}

    # 特征筛选和归一化
    df_filter = df.drop(drop_columns, axis=1)

    ## 特征筛选
    # 设置方差阈值
    selector = VarianceThreshold(threshold=0.01)

    # 应用方差筛选
    selector.fit(df_filter)
    high_variance_features = df_filter.columns[selector.get_support()]

    df_filter = df_filter[high_variance_features]

    scaler = MinMaxScaler()

    normalized_df = pd.DataFrame(scaler.fit_transform(df_filter), columns=df_filter.columns)
    df_cat = df[drop_columns].reset_index()
    df = pd.concat([df_cat, normalized_df], axis=1)
    df.set_index('stock_date', inplace=True)
    if lag_features is None:
        lag_features = list(df_filter.columns)
    print('自回归使用到的特征变量：', lag_features)
    # lag_features = ['open', 'high', 'low', 'close', 'pre_close','change', 'pct_chg', 'vol', 'amount','return']
    # lag_features = ['change', 'pct_chg','return']

    # 创建滞后特征
    for feature in lag_features:
        for lag in range(1, lagN + 1):
            df[f'{feature}_lag_{lag}'] = df.groupby('ts_code')[feature].shift(lag)

    df.dropna(inplace=True)

    grouped = df.groupby('ts_code')

    for stock_code, group in grouped:
        train = group[group['trade_date'] < start_predict_date]
        # test = group[group['trade_date'] >= start_predict_date]

        start_predict_date_dt = pd.to_datetime(start_predict_date)
        new_test_date = start_predict_date_dt + DateOffset(months=forecast_n)
        new_test_date_str = new_test_date.strftime('%Y-%m-%d')

        test = group[(group['trade_date'] >= start_predict_date) & (group['trade_date'] < new_test_date_str)]


        X_train = train[[f'{feature}_lag_{lag}' for feature in lag_features for lag in range(1, lagN + 1)]]
        y_train = train[target_variable]
        X_test = test[[f'{feature}_lag_{lag}' for feature in lag_features for lag in range(1, lagN + 1)]]
        y_test = test[target_variable]

        # 使用模型回调函数
        model = model_callback(X_train, y_train, need_grid_search, need_shap)
        #         model.fit(X_train, y_train)
        models[stock_code] = model

        y_pred = model.predict(X_test)
        predictions[stock_code] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluations[stock_code] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

        # 绘图
        # 将字符串转换为datetime对象
        start_predict_date_dt = pd.to_datetime(start_predict_date)

        # 计算新的开始日期：从start_predict_date减去12个月
        new_start_date = start_predict_date_dt - DateOffset(months=12)
        new_start_date_str = new_start_date.strftime('%Y-%m-%d')

        real_hist = group[group['trade_date'] >= new_start_date_str]
        real_test = real_hist[target_variable]

        real_test = group[target_variable]
        new_start_date = group['trade_date'].min()

        # 创建日期范围：对于真实值，从new_start_date开始，直到预测的结束日期
        # 注意：这里假设y_test的长度恰好等于预测期长度
        date_range_real = pd.date_range(start=new_start_date, periods=len(real_test), freq='D')

        # 创建日期范围：对于预测值，从start_predict_date开始
        date_range_pred = pd.date_range(start=start_predict_date, periods=len(y_pred), freq='D')

        # 绘制图表
        plt.figure(figsize=(18, 10))

        # 真实值曲线
        # 注意：这里我们需要对y_test进行适当的拓展或截取，以匹配新的日期范围
        # 如果y_test包含在一个更大的DataFrame中，你可能需要调整索引来获取正确的数据段

        plt.plot(date_range_real, real_test, label='Real Return')  # 从第12个点开始绘制，以跳过前12个月

        # 预测值曲线
        plt.plot(date_range_pred, y_pred, label='Predicted Return')

        plt.title(f'Stock: {stock_code} - Predicted vs Real Return')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.show()

        print({'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

    return {"models": models, "predictions": predictions, "evaluations": evaluations}


import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt


# 定义LightGBM回调函数
def lightgbm_model(X_train, y_train, need_grid_search=False, need_shap=True):
    # 定义LightGBM模型
    best_model = lgb.LGBMRegressor()

    if need_grid_search:

        # 设置参数网格
        param_grid = {
            'num_leaves': [15, 28, 32],  # 大的 num_leaves 可以增加模型的复杂度
            'learning_rate': [0.01, 0.05, 0.1],  # 学习速率
            'n_estimators': [100, 200, 500],  # 迭代次数，即树的数量
            'max_depth': [5, 8, 15],  # 决策树的最大深度
            'subsample': [0.7, 0.8, 0.9],  # 训练每棵树时使用的数据比例
            'colsample_bytree': [0.7, 0.8, 0.9]  # 训练每棵树时使用的特征比例
        }
        grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print("Best parameters found: ", best_params)
        print("Best model found: ", best_model)
    else:
        best_model.fit(X_train, y_train)
    if need_shap:
        shap.initjs()
        explainer = shap.TreeExplainer(best_model, feature_perturbation="interventional")
        shapvalues = explainer.shap_values(X=X_train, y=y_train)
        shap.summary_plot(shapvalues, X_train, feature_names=X_train.columns)

    return best_model


def ridge_regression_model(X_train, y_train, need_grid_search=False, need_shap=True):
    # 定义岭回归模型
    best_model = Ridge()
    if need_grid_search:
        # 设置参数网格
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],  # alpha是岭回归的正则化强度参数
        }
        grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print("Best parameters found: ", best_params)
        print("Best model found: ", best_model)
    else:
        best_model.fit(X_train, y_train)
    if need_shap:
        shap.initjs()
        masker = shap.maskers.Independent(data=X_train)
        explainer = shap.LinearExplainer(best_model, masker=masker, feature_perturbation="interventional")
        shapvalues = explainer.shap_values(X=X_train)
        shap.summary_plot(shapvalues, X_train, feature_names=X_train.columns)

    return best_model


# 定义SVR回调函数
def svr_model(X_train, y_train, need_grid_search=False, need_shap=False):
    best_model = None
    if need_grid_search:
        param_grid = {
            'kernel': ['rbf', 'linear', 'poly'],  # 考虑不同的核函数
            'C': [0.01, 0.1, 1, 10]  # 不同的C值
        }
        grid_search = GridSearchCV(SVR(), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_svr = grid_search.best_estimator_

        print("Best parameters found: ", best_params)
        print("Best SVR found: ", best_svr)
        best_model = best_svr
    else:
        best_model = SVR(kernel='rbf', C=0.01)
        best_model.fit(X_train, y_train)

    if need_shap:
        shap.initjs()
        explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(shap.sample(X_train, 100))
        shap.summary_plot(shap_values, shap.sample(X_train, 100), feature_names=X_train.columns)

    return best_model


import numpy as np


def compare_plot(result):
    mse_data = {}
    r2_data = {}
    rmse_data = {}
    mae_data = {}
    for model_name, model_result in result.items():
        for stock_code, evaluation in model_result['evaluations'].items():
            if stock_code not in mse_data:
                mse_data[stock_code] = {}
                r2_data[stock_code] = {}
                rmse_data[stock_code] = {}
                mae_data[stock_code] = {}
            mse_data[stock_code][model_name] = evaluation['MSE']
            rmse_data[stock_code][model_name] = evaluation['RMSE']
            mae_data[stock_code][model_name] = evaluation['MAE']
            r2_data[stock_code][model_name] = evaluation['R2']

    # 为每只股票绘制MSE、RMSE、MAE和R²对比图
    for stock_code in mse_data.keys():
        model_mse = mse_data[stock_code]
        model_rmse = rmse_data[stock_code]
        model_mae = mae_data[stock_code]
        model_r2 = r2_data[stock_code]

        n_models = len(model_mse)
        bar_width = 0.2
        index = np.arange(n_models)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # MSE条形图
        ax1.bar(index, model_mse.values(), bar_width, label='MSE', color='blue')

        # RMSE条形图
        ax1.bar(index + bar_width, model_rmse.values(), bar_width, label='RMSE', color='green')

        # MAE条形图
        ax1.bar(index + 2 * bar_width, model_mae.values(), bar_width, label='MAE', color='orange')

        # R²折线图
        ax2 = ax1.twinx()
        ax2.plot(index + bar_width, model_r2.values(), color='red', marker='o', label='R²')

        ax1.set_title(f'MSE, RMSE, MAE, and R² Comparison for Stock: {stock_code}')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Error', color='blue')
        ax2.set_ylabel('R²', color='red')

        ax1.set_xticks(index + bar_width)
        ax1.set_xticklabels(model_mse.keys())

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()





