# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:58:44 2022

@author: ut34u3
"""

import pandas as pd
import numpy as np
import f 
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")

sectors = ["iBoxx € Austria",
           "iBoxx € Belgium", 
           "iBoxx € Finland",
           "iBoxx € France 1-10",
           "iBoxx € France 10+",
           "iBoxx € Germany 1-10",
           "iBoxx € Germany 10+",
           "iBoxx € Greece",
           "iBoxx € Ireland",
           "iBoxx € Italy 1-10",
           "iBoxx € Italy 10+",
           "iBoxx € Luxembourg",
           "iBoxx € Netherlands",
           "iBoxx € Portugal",
           "iBoxx € Slovakia",
           "iBoxx € Spain 10+"]

file_names_simple = ['iBoxx € Austria.xlsx',
                     'iBoxx € Belgium.xlsx',
                     'iBoxx € Finland.xlsx',
                     'iBoxx € France 1-10.xlsx',
                     'iBoxx € France 10+.xlsx',
                     'iBoxx € Germany 1-10.xlsx',
                     'iBoxx € Germany 10+.xlsx',
                     'iBoxx € Greece.xlsx',
                     'iBoxx € Ireland.xlsx',
                     'iBoxx € Italy 1-10.xlsx',
                     'iBoxx € Italy 10+.xlsx',
                     'iBoxx € Luxembourg.xlsx',
                     'iBoxx € Netherlands.xlsx',
                     'iBoxx € Portugal.xlsx',
                     'iBoxx € Slovakia.xlsx',
                     'iBoxx € Spain 10+.xlsx']

file_names = ['spreadiBoxx € AustriaiBoxx € Belgium.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Finland.xlsx',
                 'spreadiBoxx € AustriaiBoxx € France 1-10.xlsx',
                 'spreadiBoxx € AustriaiBoxx € France 10+.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Germany 1-10.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Greece.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Ireland.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Portugal.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € AustriaiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Finland.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € France 1-10.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € France 10+.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Germany 1-10.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Greece.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Ireland.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Portugal.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € BelgiumiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € FinlandiBoxx € France 1-10.xlsx',
                 'spreadiBoxx € FinlandiBoxx € France 10+.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Germany 1-10.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Greece.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Ireland.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Portugal.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € FinlandiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € France 10+.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Germany 1-10.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Greece.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Ireland.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Portugal.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € France 1-10iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Germany 1-10.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Greece.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Ireland.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Portugal.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € France 10+iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Germany 10+.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Greece.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Ireland.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Portugal.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € Germany 1-10iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Greece.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Ireland.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Portugal.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € Germany 10+iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Ireland.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Portugal.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € GreeceiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Italy 1-10.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Portugal.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € IrelandiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Italy 10+.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Portugal.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € Italy 1-10iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € Italy 10+iBoxx € Luxembourg.xlsx',
                 'spreadiBoxx € Italy 10+iBoxx € Netherlands.xlsx',
                 'spreadiBoxx € Italy 10+iBoxx € Portugal.xlsx',
                 'spreadiBoxx € Italy 10+iBoxx € Slovakia.xlsx',
                 'spreadiBoxx € Italy 10+iBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € LuxembourgiBoxx € Netherlands.xlsx',
                 'spreadiBoxx € LuxembourgiBoxx € Portugal.xlsx',
                 'spreadiBoxx € LuxembourgiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € LuxembourgiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € NetherlandsiBoxx € Portugal.xlsx',
                 'spreadiBoxx € NetherlandsiBoxx € Slovakia.xlsx',
                 'spreadiBoxx € NetherlandsiBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € PortugaliBoxx € Slovakia.xlsx',
                 'spreadiBoxx € PortugaliBoxx € Spain 10+.xlsx',
                 'spreadiBoxx € SlovakiaiBoxx € Spain 10+.xlsx']

q = 0.9
n = 2
nan=np.nan

dfs = []
for file in file_names:
    dfs.append(pd.read_excel(file, index_col=(0)).dropna())
    
dfes = []
for df in dfs: 
    df['pastspread'] = df.spread.diff(1)
    std = df.std()['pastspread']
    mean = df.mean()['pastspread']
    df['pastspread'] = (df['pastspread']-mean)/std
    
    df.spread = -df.spread.diff(-n)
    std = df.std()['spread']
    mean = df.mean()['spread']
    df['spread'] = (df['spread']-mean)/std
    
    df.T.index.name = 'client'
    
    df = f.fe(df, 'spread', n).dropna()
    qu = df['spread'].quantile(q)
    qd = df['spread'].quantile(1-q)
    df.spread = df.spread.apply(f.value_to_class, qu=qu, qd=qd)
    df = df[df.spread!=0.5]
    
    dfes.append(df)
    
    
dfeeee = []
a = dfes[0].index[100]
for df in dfes:
    dfeeee.append(df[df.index<a])
dfes = dfeeee


classifiers = {#'XGB': XGBClassifier(
                             # learning_rate =0.1,
                             # n_estimators=1000,
                             # max_depth=5,
                             # min_child_weight=1,
                             # gamma=0,
                             # subsample=0.8,
                             # colsample_bytree=0.8,
                             # objective= 'binary:logistic',
                             # nthread=4,
                             # seed=27),
               'LGB': lgb.LGBMClassifier(),
               #"Nearest Neighbors": KNeighborsClassifier(3),
               #"Linear SVM":SVC(kernel="linear", C=0.025),
               #"RBF SVM":SVC(gamma=2, C=1),
               #"Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0)),
               #"Decision Tree": DecisionTreeClassifier(max_depth=5),
               #"Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               #"Neural Net":MLPClassifier(alpha=1, max_iter=1000),
               #"AdaBoost":AdaBoostClassifier(),
               #"Naive Bayes":GaussianNB(),
               #"QDA":QuadraticDiscriminantAnalysis()
            }





dtrain = []
for i in range(len(dfes)):
    dtrain.append(dfes[i])
    
    
predictelem = 'spread'
dft = pd.concat(dtrain)
dft.replace([np.inf, -np.inf], np.nan, inplace=True)
dft.dropna(inplace=True)
feat = list(dft.columns)
feat.remove(predictelem)
Xt = dft[feat]
yt = dft[predictelem]
for name, clf in classifiers.items():
    clf.fit(Xt, yt)

    






dfs = []
for file in file_names:
    dfs.append(pd.read_excel(file, index_col=(0)).dropna())
    
dfes = []
for df in dfs: 
    #df = pd.DataFrame(df.spread)
    #df['all'] = df['all'].shift(1)
    
    df['pastspread'] = df.spread.diff(1)
    std = df.std()['pastspread']
    mean = df.mean()['pastspread']
    df['pastspread'] = (df['pastspread']-mean)/std
    
    df.T.index.name = 'client'
    
    df = f.fe(df, 'spread', n).dropna()
    
    dfes.append(df[df.index>=a])
    
    
s = {}

feat = list(dfes[0].columns)
feat.remove('spread')
for i in range(len(dfes)):
    Xv = dfes[i][feat]
    yv = dfes[i]['spread']
    s[file_names[i]] = pd.DataFrame(clf.predict(Xv), index=yv.index)
    
    
score = pd.DataFrame(columns=sectors, index=yv.index)
for i in score.index:
    for j in score.columns:
        score[j][i]=0
for i in range(len(dfes)):
    num = file_names[i][14:19]
    den = file_names[i][-10:-5]
    for sec in sectors:
        if sec[8:13]==num:
            num=sec
        if sec[-5:]==den:
            den=sec
    
    for j in score[num].index:
        try:
            if s[file_names[i]][0][j] == 1.0:
                score[num][j] = score[num][j]+1
                score[den][j] = score[den][j]-1
            if s[file_names[i]][0][j] == 0.0:
                score[num][j] = score[num][j]-1
                score[den][j] = score[den][j]+1
        except:
            True
    
dfss={}
for file in file_names_simple:
    b = pd.read_excel(file, index_col=(0)).dropna()
    b = b.iloc[1:,:]
    b.columns = ['date', 'del', 'spread']
    del b['del']
    b.set_index('date', inplace=True)
    dfss[file[:-5]] = b
    

perf = pd.DataFrame(columns=['spread'], index=list(score.index))
cur=0
for i in score.index[:-1]:
    long = list(score[score.index==i].rank(axis=1).sort_values(i, axis=1).columns[:1])
    short = list(score[score.index==i].rank(axis=1).sort_values(i, axis=1).columns[-1:])
    try:
        for l in long:
            cur = cur + dfss[l][dfss[l].index>i]['spread'].iloc[0] - dfss[l]['spread'][i]
        for l in short:
            cur = cur - dfss[l][dfss[l].index>i]['spread'].iloc[0] + dfss[l]['spread'][i]
        perf['spread'][i] = cur
    except: 
        True
    
    
perf.plot()
    
    
    