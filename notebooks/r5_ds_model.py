import pandas as pd
import json
import csv 
import re
import glob
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

#SKLEARN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_auc_score ,cohen_kappa_score, f1_score, roc_curve,auc, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from scipy.stats import ks_2samp

#XGBOOST
import xgboost as xgb
from xgboost import cv
from xgboost import XGBClassifier

#bayes
#!pip install fastai wwf bayesian-optimization -q --upgrade
from bayes_opt import BayesianOptimization

#pickle
from pickle import dump
import pickle


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_validate #método para evaluar varios particionamientos de C-V
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, LeaveOneOut #Iteradores de C-V
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso #modelamiento
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score #métricas de evaluación
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from datetime import date
from datetime import datetime, timedelta
from keras.regularizers import L1L2

import pickle
from data_modeling import data_encoder

#%%
X = data_encoder.drop(columns=['fraudfound_p'])
y = data_encoder['fraudfound_p']

def decile_metrics(data=None, formatting=False, num_decile=10):
    '''Calcula métricas de un modelo basadas en deciles, dados los scores evaluados sobre un dataset.
    
    Intenta calcular deciles. Si la distribución está demasiado concentrada en uno de los extremos,
    busca el mejor número de percentiles que se adapte a la distribución (menos de 10 buckets).
    
    Recibe un dataframe los scores estimados por el modelo (columna "prob") y 
    la clase verdadera para cada instancia (columna "target")

    Retorna un dataframe con el detalle del cálculo del KS de cada decil (u otro percentil).
    ''' 
    continuar = True

    while continuar:
        try:
            data['target0'] = 1 - data["target"]
            data['bucket'] = pd.qcut(data["prob"], num_decile)
            grouped = data.groupby('bucket', as_index = False)
            metrics = pd.DataFrame()
            metrics['min_prob'] = grouped.min()["prob"]
            metrics['max_prob'] = grouped.max()["prob"]
            metrics['events']   = grouped.sum()["target"]
            metrics['nonevents'] = grouped.sum()['target0']

            metrics = metrics.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
            metrics['event_rate'] = (metrics.events / data["target"].sum()).apply('{0:.2%}'.format)
            metrics['nonevent_rate'] = (metrics.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
            metrics['cum_eventrate']=(metrics.events / data["target"].sum()).cumsum()
            metrics['cum_noneventrate']=(metrics.nonevents / data['target0'].sum()).cumsum()
            metrics['KS'] = np.round(metrics['cum_eventrate']-metrics['cum_noneventrate'], 3) * 100

            metrics["bad_rate"]=metrics.events / (metrics.events + metrics.nonevents)
            metrics["cum_bad_rate"]=metrics.apply(
                lambda x: 
                metrics[metrics.max_prob <= x.max_prob].events.sum() / \
                    (metrics[metrics.max_prob <= x.max_prob].events.sum() +    \
                     metrics[metrics.max_prob <= x.max_prob].nonevents.sum()),
                axis=1
            )

            continuar = False
        except Exception as e:
            num_decile = num_decile -1

    #Formatting in a more readable form, if the resultaing table is to be communicated
    if formatting:
        metrics['cum_eventrate']=metrics['cum_eventrate'].apply('{0:.2%}'.format)
        metrics['cum_noneventrate']=metrics['cum_noneventrate'].apply('{0:.2%}'.format)
        
    metrics.index = range(1,1+num_decile)
    metrics.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
                
    return(metrics)


def decile_metrics_test_from_train(metrics_train=None, preds_test=None):
    '''Calcula métricas de un modelo basadas en los deciles (u otro particionamiento) 
    previamente establecidos para otro dataset de entrenamiento.
    
    Recibe un dataframe con el detalle de métricas como la generada por el método decile_metrics() y un 
    dataframe con las probabilidades estimadas por el modelo (columna "prob") y 
    la clase verdadera para cada instancia (columna "target")

    Retorna un dataframe con el detalle del cálculo de las métricas de cada decil (u otro percentil)
    con las mismas características que la retornada por la función decile_metrics(().
    ''' 

    metrics = pd.DataFrame()
    metrics['min_prob'] = metrics_train.min_prob
    metrics['max_prob'] = metrics_train.max_prob

    metrics["events"]=metrics.apply(
        lambda x: ((preds_test["target"]==1) & (preds_test["prob"]>= x.min_prob) & (preds_test["prob"]< x.max_prob)).sum(),
        axis=1
    )
    metrics["nonevents"]=metrics.apply(
        lambda x: ((preds_test["target"]==0) & (preds_test["prob"]>= x.min_prob) & (preds_test["prob"]< x.max_prob)).sum(),
        axis=1
    )

    metrics['event_rate'] = metrics.events/metrics.events.sum()
    metrics['nonevent_rate'] = metrics.nonevents/metrics.nonevents.sum()
    metrics['cum_eventrate'] = metrics['event_rate'].cumsum()
    metrics['cum_noneventrate'] = metrics['nonevent_rate'].cumsum()
    metrics['KS'] = np.round(metrics['cum_eventrate']-metrics['cum_noneventrate'], 3) * 100

    metrics["bad_rate"]=metrics.events / (metrics.events + metrics.nonevents)
    metrics["cum_bad_rate"]=metrics.apply(
        lambda x: 
        metrics[metrics.max_prob <= x.max_prob].events.sum() / \
            (metrics[metrics.max_prob <= x.max_prob].events.sum() +    \
             metrics[metrics.max_prob <= x.max_prob].nonevents.sum()),
        axis=1
    )
    
    metrics.index = range(1,1+len(metrics_train))
    metrics.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 10)
    return metrics

def graph_ks(metrics):
    '''Crea el gráfico de probabilidades acumuladas para las clases event y nonevent,
    mostrando el punto donde se obtiene el MAX KS.
        
    Recibe un dataframe con las mismas características que la retornada por la función
    método decile_metrics()
    ''' 

    decileKS = np.argmax(metrics.KS) +1
    
    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(metrics.cum_eventrate, color="red", label="event")
    plt.plot(metrics.cum_noneventrate, color="green", label="nonevent")
    plt.vlines(decileKS, metrics.cum_noneventrate[decileKS], metrics.cum_eventrate[decileKS], linestyles='dashed')
    plt.text(decileKS+0.2, (metrics.cum_noneventrate[decileKS]+metrics.cum_eventrate[decileKS])/2, "Max KS")
    plt.xlabel("Decil")
    plt.ylabel("Proporción acumulada")
    plt.title('Max KS {:.2f} at decile {}'.format(max(metrics['KS']), decileKS))
    plt.legend(loc="lower right")
    
def train_and_evaluate_AUC(gamma, mcw, lr, max_depth,reg_lambda,reg_alpha,subsample,colsample_bytree):
    """Function we want to maximize (Black box)

    It first trains a model with the training set using the received hyper 
    parameterts, and then evaluatesand returns the Max KS over the test set.
    """
    
    
    gamma = int(round(gamma))
    mcw = int(round(mcw))
    max_depth = int(round(max_depth))
    gamma = int(round(gamma))
    
    
    model = XGBClassifier(n_jobs=-1,
                       gamma=gamma, 
                       min_child_weight=mcw, 
                       learning_rate=lr,
                       subsample = subsample,
                       colsample_bytree = colsample_bytree,
                       max_depth=max_depth,
                       reg_lambda = reg_lambda,
                       n_estimators=2000,
                       reg_alpha = reg_alpha,
                       objective='binary:logistic')

    model.fit(X_train,y_train,verbose=True)#,early_stopping_rounds=10,eval_metric="auc",eval_set=[(X_val_sc, y_val)], verbose=False)
   
    preds_train = pd.DataFrame(
    {"prob":model.predict_proba(X_train)[:, 1],
     "target":y_train
    })

    preds_test = pd.DataFrame(
    {"prob":model.predict_proba(X_val)[:, 1],
     "target":y_val
    })

    metrics_train = decile_metrics(data=preds_train)
    metrics_test  = decile_metrics_test_from_train(metrics_train,preds_test)

    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train , model.predict_proba(X_train)[:,1])
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_val , model.predict_proba(X_val)[:,1])

    auc_pr_train = auc(recall_train, precision_train)*100
    auc_pr_test = auc(recall_test, precision_test)*100
  
    metrics_test_auc = roc_auc_score(y_val , model.predict_proba(X_val)[:,1])
    metrics_train_auc = roc_auc_score(y_train , model.predict_proba(X_train)[:,1])

    beta= 2

    pr_ks_train = ((1+beta**2)* auc_pr_train * max(metrics_train['KS']) )/( (beta**2*auc_pr_train) + max(metrics_train['KS']) )
    pr_ks_test = ((1+beta**2)*auc_pr_test * max(metrics_test['KS']))/( (beta**2*auc_pr_test) + max(metrics_test['KS']) )
    
    target = pr_ks_test*(1-(abs(pr_ks_train-pr_ks_test)/pr_ks_test))


    dicta={'model':[model.get_params()],"metric": target }
    test = pd.DataFrame(dicta)
    global df
    df = df.append(test)
    
    print(" target es: {},PR_KS train/test {}/{},PR_AUC train/test {}/{},AUC train/test {}/{},KS train/test {}/{}".
          format(target,pr_ks_train,pr_ks_test,auc_pr_train,auc_pr_test,metrics_train_auc,metrics_test_auc, max(metrics_train['KS']), max(metrics_test['KS'])))

    return target

#%%

pbounds = {'gamma': (0, 5), 'mcw': (1, 15), 'lr': (0.01, 0.1), 'max_depth':(1,6) , "reg_lambda" :(0.001, 5),"reg_alpha" :(0.001, 5) ,"subsample":(0.4,0.8),"colsample_bytree":(0.4,0.8)}

from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f=train_and_evaluate_AUC,
    pbounds=pbounds,
    random_state=1,
    verbose=2
)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, random_state=42, test_size=0.3,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, random_state=42,test_size=0.5,stratify=y_rem)

df = pd.DataFrame()
optimizer.maximize(init_points=10, n_iter=40)
df.to_csv("Optimizer_V1.csv")

optimizer_v1=pd.read_csv("Optimizer_V1.csv")
optimizer_v1

optimizer_v1["metric"].max()

param1= optimizer_v1[optimizer_v1["metric"] == 38.534462983468806]["model"][59]
param1

from numpy import nan
dict_str=eval(param1)
dict_str

modelo = XGBClassifier(**dict_str)

modelo.fit(X_train, y_train)

preds_train = pd.DataFrame(
    {"prob":modelo.predict_proba(X_train)[:, 1],
     "target":y_train
    })

preds_test = pd.DataFrame(
    {"prob":modelo.predict_proba(X_val)[:, 1],
     "target":y_val
    })

metrics_train = decile_metrics(data=preds_train)
metrics_test  = decile_metrics_test_from_train(metrics_train,preds_test)
#print(metrics_test)
graph_ks(metrics_train)
graph_ks(metrics_test)

metrics_test_auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:,1])
metrics_train_auc = roc_auc_score(y_train , modelo.predict_proba(X_train)[:,1])
print(metrics_train_auc)
print(metrics_test_auc)


preds = modelo.predict_proba(X_train)[:, 1]
preds


#%% Evaluando en test

X_train_2 = pd.concat((X_train, X_val), axis=0)
y_train_2 = pd.concat((y_train, y_val), axis=0)


modelo = XGBClassifier(**dict_str)

modelo.fit(X_train_2, y_train_2)

preds_train = pd.DataFrame(
    {"prob":modelo.predict_proba(X_train_2)[:, 1],
     "target":y_train_2
    })

preds_test = pd.DataFrame(
    {"prob":modelo.predict_proba(X_test)[:, 1],
     "target":y_test
    })

metrics_train = decile_metrics(data=preds_train)
metrics_test  = decile_metrics_test_from_train(metrics_train,preds_test)
#print(metrics_test)
graph_ks(metrics_train)
graph_ks(metrics_test)


metrics_test_auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1])
metrics_train_auc = roc_auc_score(y_train_2 , modelo.predict_proba(X_train_2)[:,1])
print(metrics_train_auc)
print(metrics_test_auc)

precision_train, recall_train, thresholds_train = precision_recall_curve(y_train_2 , modelo.predict_proba(X_train_2)[:,1])
precision_test, recall_test, thresholds_test = precision_recall_curve(y_test , modelo.predict_proba(X_test)[:,1])

auc_pr_train = auc(recall_train, precision_train)*100
auc_pr_test = auc(recall_test, precision_test)*100

print(auc_pr_train)
print(auc_pr_test)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1])

from sklearn.metrics import roc_auc_score
sns.distplot(preds_test[preds_test.target == 1].prob, color='r')
sns.distplot(preds_test[preds_test.target == 0].prob, color='b')

modelo.score(X_test,y_test)

preds=modelo.predict_proba(X_test)[:,1]
preds

preds_bin = np.where(preds > 0.14, 1, 0)
preds_bin

from sklearn.metrics import classification_report
print(classification_report(y_test, preds_bin))

# Save model para poner en producción
import joblib 
joblib.dump(modelo, 'Model_Fraude.pkl')

pickle.dump(modelo, open('C:/Users/Acer/Desktop/enginner_r5/r5-ds-challenge/models/Model_Fraude.pkl', 'wb'))

import dalex as dx
exp = dx.Explainer(modelo, X_test, y_test)

mp = exp.model_performance(model_type = 'classification')
mp.result

