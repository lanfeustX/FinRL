# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:13:39 2022

@author: ut34u3
"""

import numpy as np 
import pandas as pd 

def process_isin_old(df, zscore=False):
    if min(df.spread)<1:
        df.spread = df.spread - min(df.spread) + 1
    df.sort_values('date', inplace=True)
    df['log_ret'] = np.log(df.spread) - np.log(df.spread.shift(1))
    if zscore == True:
        m = df['log_ret'].ewm(1).mean()
        std = df['log_ret'].ewm(1).std()
        df['log_ret'] = (df['log_ret']-m)/std
    return df

def process_isin(df):
    df.sort_values('date', inplace=True)
    df['variation'] = df.spread - df.spread.shift(1)
    df['variation'] = (df['variation']-df['variation'].mean())/df['variation'].std()
    return df