import os
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

setup(
    name="dataml",
    version="0.1.4",
    description="data analysis machine learning",
    author="",
    url="",
    install_requires=[
        "setuptools",
        "wheel",
        "shap==0.41.0",
        "scikit_learn==0.22.1",
        "matplotlib==3.1.3",
        "seaborn==0.10.0",
        "tqdm==4.46.0",
        "numpy==1.18.3",
        "numba==0.48.0",
        "pandas == 1.0.1",
        "scorecardpy==0.1.9.2",
        "lightgbm==2.3.1",
        "xgboost==0.81",
        # "statsmodels==0.12.2",
        "statsmodels==0.11.0",
        "xlrd==2.0.0",
        "sqlparse==0.4.3",
        "SQLAlchemy==1.3.13",
        "psycopg2-binary==2.8.4",
        "colorama==0.4.3",
        # "bayesian-optimization==1.4.1",
        "imbalanced-learn==0.6.2",
        "wordcloud==1.9.3"
    ],
    packages=find_packages(),
    long_description_content_type="",
    include_package_data=True,
        # 将.so或者.pyd文件打包进入
        package_data={
            '': ['*.so','*.pyd']
        }
)

