import os
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

setup(
    name="dataml",
    version="1.1.0",
    description="data analysis machine learning",
    author="",
    url="",
    install_requires=[
        "setuptools",
        "wheel",
        "scikit_learn==0.24.2",
        "matplotlib==3.3.4",
        # "scikit_learn<=1.0.2",
        "seaborn==0.11.2",
        "tqdm==4.64.1",
        "numpy==1.19.5",
        "pandas == 1.1.5",
        "scorecardpy==0.1.9.2",
        "lightgbm==2.3.1",
        "xgboost==0.81",
        # "statsmodels==0.12.2",
        "statsmodels==0.13.2",
        "shap==0.41.0",
        "xlrd==2.0.0",
        "sqlparse==0.4.3",
        "SQLAlchemy==1.4.51",
        # "psycopg2-binary==2.7.6.1",
        # "psycopg2-binary==2.8.4",
        "psycopg2-binary==2.8.6",
        "colorama==0.4.5",
        "bayesian-optimization==1.4.1",
        "imbalanced-learn==0.8.0",
        "wordcloud==1.9.3",
        "colorlog==6.8.2",
        "catboost==0.26"
    ],
    packages=find_packages(),
    long_description_content_type="",
    include_package_data=True,
        # 将.so或者.pyd文件打包进入
        package_data={
            '': ['*.so','*.pyd']
        }
)