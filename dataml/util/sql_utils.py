import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import io
import math

from math import sqrt
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import joblib
import csv
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

from sqlalchemy import types


dtype = {'name': types.NVARCHAR(length=20)}

def has_table(engine, tablename, schema):
    flag = True
    try:
        with engine.connect() as connection:
            with connection.connection.cursor() as cursor:
                cursor.execute("explain select * from {}.{}".format(schema, tablename))
                connection.connection.commit()
    except:
        flag = False
    return flag

def drop_table(engine, tablename, schema):
    flag = True
    try:
        with engine.connect() as connection:
            with connection.connection.cursor() as cursor:
                cursor.execute("drop table {}.{}".format(schema, tablename))
                connection.connection.commit()
    except:
        flag = False
    return flag

def truncate_table(engine,tablename,schema):
    flag = True
    try:
        with engine.connect() as connection:
            cursor = connection.connection.cursor()
            cursor.execute("TRUNCATE TABLE {}.{}}".format(schema, tablename))
            connection.connection.commit()
    except Exception as e:
        flag = False
    return  flag

def data2dbbase(df, engine, table_name, schema=''):
    if_exists = 'replace'
    string_data_io = io.StringIO()
    df.to_csv(string_data_io, sep='|', index=False)

    pd_sql_engine = pd.io.sql.pandasSQL_builder(engine)
    table = pd.io.sql.SQLTable(table_name,pd_sql_engine,frame=df
                               ,index=False,if_exists=if_exists,schema=schema,dtype=dtype)
    exists_flag = has_table(engine,table_name,schema)
    if exists_flag:
        if if_exists == 'fail':
            raise ValueError("Table '%s' already exits."% table_name)
        elif if_exists == 'replace':
            drop_table(engine.table_name, schema)
            table.execute_create()
        elif if_exists == 'overwrite':
            truncate_table(engine,table_name,schema)
        elif if_exists == 'append':
            pass
        else:
            raise ValueError("'{0}' is not valid for is_exists".format(if_exists))
    else:
        table.execute_create()
    string_data_io.seek(0)
    with engine.connect() as connection:
        with connection.connection.cursor() as cursor:
            copy_cmd = f"COPY {schema}.{table} FROM STDIN HEADER DELIMITER '|' CSV"
            cursor.copy_expert(copy_cmd,string_data_io)
        connection.connection.commit()





