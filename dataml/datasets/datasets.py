from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer
import psycopg2.extras

import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

from ..util.utils import read_data_by_query
from ..util.sql_utils import data2dbbase



# ... until with utils import data2database

from ..util.my_logging import logger

class Warehouse:
    def __init__(self):
        pass

    def connect(self):
        pass


class DefaultWarehouse(Warehouse):
    def __init__(self, warehouse_address, need_reload_data=False):
        self.warehouse_address = warehouse_address
        self.need_reload_data = need_reload_data

    def connect(self, write=True):
        '''
        连接到数据仓库
        :param write:
        :return:
        '''
        engine = None

        if write:
            engine = create_engine(self.warehouse_address)
        else:
            if self.need_reload_data and self.warehouse_address is not None:
                engine = create_engine(self.warehouse_address)
        logger.info('engine:{}'.format(engine) )
        return engine

class Dataset:
    def __init__(self):
        pass

    def read(self) -> DataFrame:
        pass

    def write_by_df(self, df):
        pass

class LocalDataset(Dataset):
    '''
    本地数据集加载
    '''
    def __init__(self,data_path,index_col=None,
                 dtype=None,file_name='cst_score_list',
                 data_type='csv',model_dir=None,y_target='target'):
        self.data_path = data_path
        self.index_col = index_col
        self.data_type = data_type
        self.file_name = file_name
        self.model_dir = model_dir
        self.dtype = dtype
        self.y_target = y_target

    def read(self) -> DataFrame:
        df = None
        if isinstance(self.data_path,DataFrame):
            return self.data_path

        if self.data_type == 'csv':
            if self.index_col is not None:
                if self.dtype is None:
                    df = pd.read_csv(self.data_path,encoding='utf8',index_col=self.index_col)
                else:
                    df = pd.read_csv(self.data_path,encoding='utf8',index_col=self.index_col,dtype=self.dtype)
            else:
                if self.dtype is not None:
                    df = pd.read_csv(self.data_path,encoding='utf8',dtype=self.dtype)
                else:
                    df = pd.read_csv(self.data_path,encoding='utf8')

        else:
            logger.info('no implements.')
        if self.y_target in df.columns:
            df =df[df[self.y_target].notnull()]

        df = df[~df.index.duplicated(keep='first')]
        return df

    def write_by_df(self,df):
        wirte_path = os.path.join(self.model_dir, '{}.csv'.format(self.file_name))
        df.to_csv(wirte_path, encoding='utf8', index=False)
        logger.debug('写入：{}.csv成功！'.format(wirte_path))

class SQLDataset(Dataset):
    def __init__(self, warehouse,
                 sqls=None,
                 need_reload_data=False,
                 dtype=None,
                 index_col='cst_id',
                 cache_dir=None,
                 schema=None,
                 outlist_table_name=None
                 ):
        self.warehouse = warehouse
        self.sqls = sqls
        self.need_reload_data = need_reload_data
        self.dtype = dtype
        self.index_col = index_col
        self.cache_dir = cache_dir
        self.schema = schema
        self.outlist_table_name = outlist_table_name
        self.write_engine = warehouse.connect(write=True)

    def read(self):
        df = None
        if len(self.sqls) ==1:
            sql = self.sqls[0]
            df = read_data_by_query(self.engine,sql,self.cache_dir,need_reload_data=self.need_reload_data,dtype=self.dtype,index_col=self.index_col)
        else:
            for inx,sql in enumerate(self.sqls):
                logger.debug('{},{}'.format(inx+1,sql))
                tmp_df=read_data_by_query(self.engine,sql,self.cache_dir,need_reload_data=self.need_reload_data,dtype=self.dtype,index_col=self.index_col)
                if df is not None:
                    df = pd.merge(df,tmp_df,left_index=True,right_index=True,how='left')
                else:
                    df = tmp_df
                del tmp_df

        if self.rename_columns is not None:
            df.rename(columns=self.rename_columns,inplace=True)
        logger.debug(df.shape)
        return df

    def write_by_df(self,df,if_exists='replace'):
        data2dbbase(df,self.engine,self.outlist_table_name,schema=self.schema)
        logger.info('写入表：{}.{}成功！'.format(self.schema,self.outlist_table_name))




