from pandas import DataFrame
import pickle
import scorecardpy as sc
import pandas as pd

from ..util.utils import onehot_encode,load_sc_bins,feature_classify
from ..util.my_logging import logger

class FeatureEncoder:
    def __init__(self,model_dir,name='feature encoder'):
        self.name = name
        self.model_dir = model_dir
        logger.info(self.name)

    def encode(self, X, flag='train', prefix='all_train') -> DataFrame:
        pass

    def persist(self,X,y,target='target',model_type='train'):
        all_df = X.copy()
        all_df[target] = y
        all_df = all_df.reset_index()
        all_df.to_csv(self.model_dir + '/all_features_{}.csv'.format(model_type), index=False)
        logger.info('feature save successfully,path:{}'.format(self.model_dir + '/all_features_{}.csv'.format(model_type)))
        logger.debug(all_df.shape)


class OnehotEncoder(FeatureEncoder):
    def __init__(self, model_dir,name=' one feature encoder',y_target='target'):
        self.name = name
        self.model_dir = model_dir
        self.y_target = y_target
        # super.__init__(name=name)

    def encode(self, X, flag='train', prefix='all_train_') -> DataFrame:
        _, _, _, cat_features, num_features = feature_classify(X, target=self.y_target)
        X[cat_features] = X[cat_features].astype(str)
        # one-hot encoding for train
        cat_onehot_df = None
        if flag == 'train':
            cate_onehot_df = onehot_encode(X[cat_features],
                                           self.model_dir,
                                           flag='train', prefix=prefix)
        elif flag == 'inverse_transform':
            cate_onehot_df = onehot_encode(X,
                                           self.model_dir,
                                           flag='inverse_transform', prefix=prefix)
        else:
            # one-hot encoding for test
            cate_onehot_df = onehot_encode(X[cat_features],
                                           self.model_dir,
                                           flag='test', prefix=prefix)
        logger.info(cate_onehot_df.shape)
        X = pd.concat([cate_onehot_df, X[num_features]], axis=1)
        return X



class WoeEncoder(FeatureEncoder):
    def __init__(self, model_dir,name='woe feature encoder',binpath=None):
        self.name = name
        self.model_dir = model_dir
        self.binpath = binpath
        # super.__init__(name=name)

    def encode(self, X, flag='train', prefix='all_train') -> DataFrame:
        #load bins for woe
        bins = None
        if self.binpath is None:
            bins = load_sc_bins(self.model_dir)
        else:
            with open(self.binpath, 'rb') as readfile:
                bins = pickle.load(readfile)

        data_woe_df = sc.woebin_ply(X, bins)
        logger.info(data_woe_df.shape)
        # logger.info(data_woe_df.head())
        return data_woe_df



