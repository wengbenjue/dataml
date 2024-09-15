from dataml.feature_process import VarDerivaticeFeatureProcess
from dataml.util.my_logging import logger
import  pandas as pd
import numpy as np
class ScoreCardDerivationFeatureProcess(VarDerivaticeFeatureProcess):
    def __init__(self, name: str='自定义特征衍生'):
        self.name = name

    def process(self, df, df1=None, df2=None, mode='train'):

        

        return df,None

