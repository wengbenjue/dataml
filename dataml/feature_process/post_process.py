import os


from ..util.imbance_utils import under_sample_en,oversample_encoded,mixed_sampling_with_categorical,mixed_sampling_with_categorical_encoded

import warnings


warnings.filterwarnings('ignore')
import pandas as pd

from ..util.my_logging import logger

pd.set_option('display.max_columns', None)


class PostProcess:
    def __init__(self):
        pass

    def process(self, X, y, mode='train'):
        pass


class ImbalanceProcess(PostProcess):
    '''
    数据不均衡处理
    '''

    def __init__(self, name='数据不均衡处理', imbalance_type='under_sample', desired_ratio=2, k_neighbors=5,
                 under_desired_ratio=1 / 6, y_target='label', data_dir='./data'):
        self.name = name
        self.imbalance_type = imbalance_type
        self.desired_ratio = desired_ratio
        self.k_neighbors = k_neighbors
        self.under_desired_ratio = under_desired_ratio
        self.data_dir = data_dir
        self.y_target = y_target

    def process(self, X, y, mode='train'):
        df = X
        if mode != 'train':
            logger.warning('Imbalance process can only be used in train mode. Ignoring it.')
            return X, y

        df_balanced = df.copy()

        if self.imbalance_type is None:
            logger.info('No sampling strategy, processing skipped.')
            return df_balanced, y

        if self.imbalance_type == 'under_sample':
            X,y = under_sample_en(X,y, self.y_target, self.desired_ratio)
        elif self.imbalance_type == 'over_sample':
            X,y = oversample_encoded(X, y, positive_class_ratio=self.desired_ratio)
        else:
            X, y = mixed_sampling_with_categorical_encoded(
                X, y, self.desired_ratio, y_target=self.y_target,
                under_desired_ratio=self.under_desired_ratio, data_dir=self.data_dir
            )

        logger.debug(f'{self.name}, {X.shape}, using {self.imbalance_type} sample strategy')
        return X, y

    def __call__(self, X, y, mode='train'):
        '''
        使对象可以像函数一样调用，并自动调用process方法处理数据
        '''
        return self.process(X, y, mode)




