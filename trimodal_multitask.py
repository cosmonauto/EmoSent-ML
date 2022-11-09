import numpy as np, json
import pickle, sys, argparse
import keras
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
global seed
seed = 1337
np.random.seed(seed)
import gc
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
#=============================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#=============================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
#=============================================================

def attention(att_type, x, y):

    if att_type == 'simple':
        m_dash = dot([x, y], axes=[2,2])
        m = Activation('softmax')(m_dash)
        h_dash = dot([m, y], axes=[2,1])
        return multiply([h_dash, x])

    elif att_type == 'gated':
        alpha_dash = dot([y, x], axes=[2,2])
        alpha = Activation('softmax')(alpha_dash)
        x_hat = Permute((2, 1))(dot([x, alpha], axes=[1,2]))
        return multiply([y, x_hat])

    else:
        print ('Attention type must be either simple or gated.')

def emotionClass(testLabel):
    trueLabel     = []
    for i in range(testLabel.shape[0]):
        maxLen       = []
        for j in range(testLabel.shape[1]):
            temp = np.zeros((1,7),dtype=int)[0]
            pos  = np.nonzero(testLabel[i][j])[0]
            temp[pos] = 1
            maxLen.append(temp)
        trueLabel.append(maxLen)
    trueLabel = np.array(trueLabel)
    return trueLabel

def seventhClass(inputLabel, mask):
    updatedLabel = np.zeros((inputLabel.shape[0],inputLabel.shape[1],7), dtype ='float')
    for i in range(inputLabel.shape[0]):
        for j in range(list(mask[i]).count(1)):
            suM = np.sum(inputLabel[i][j])
            if suM == 0:
                updatedLabel[i][j][6] = 1
            else:
                updatedLabel[i][j][0:6] = input