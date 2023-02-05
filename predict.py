import psycopg2
import category_encoders as ce
import pickle as pickle
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


encoder_load = pickle.load(open('encode.pkl', "rb"))
X = encoder_load.drop(columns=['fraudfound_p'])
Model = pickle.load(open('models/Model_Fraude.pkl', 'rb'))
result = Model.predict_proba(X)[:, 1]
print(result)