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
                updatedLabel[i][j][0:6] = inputLabel[i][j]
                updatedLabel[i,j,np.nonzero(updatedLabel[i][j])[0]]=1
    return updatedLabel

def featuresExtraction():
    global train_text, train_audio, train_video, senti_train_label, emo_train_label, train_mask
    global valid_text, valid_audio, valid_video, senti_valid_label, emo_valid_label, valid_mask
    global test_text, test_audio, test_video, senti_test_label, emo_test_label, test_mask
    global max_segment_len

    text          = np.load('MOSEI/text.npz',mmap_mode='r')
    audio         = np.load('MOSEI/audio.npz',mmap_mode='r')
    video         = np.load('MOSEI/video.npz',mmap_mode='r')

    train_text    = text['train_data']
    train_audio   = audio['train_data']
    train_video   = video['train_data']

    valid_text    = text['valid_data']
    valid_audio   = audio['valid_data']
    valid_video   = video['valid_data']

    test_text     = text['test_data']
    test_audio    = audio['test_data']
    test_video    = video['test_data']


    senti_train_label   = video['trainSentiLabel']
    senti_valid_label   = video['validSentiLabel']
    senti_test_label    = video['testSentiLabel']

    senti_train_label   = to_categorical(senti_train_label >= 0)
    senti_valid_label   = to_categorical(senti_valid_label >= 0)
    senti_test_label    = to_categorical(senti_test_label >= 0)

    emo_train_label   = video['trainEmoLabel']
    emo_valid_label   = video['validEmoLabel']
    emo_test_label    = video['testEmoLabel']

    train_length  = video['train_length']
    valid_length  = video['valid_length']
    test_length   = video['test_length']

    max_segment_len = train_text.shape[1]

    train_mask = np.zeros((train_video.shape[0], train_video.shape[1]), dtype='float')
    valid_mask = np.zeros((valid_video.shape[0], valid_video.shape[1]), dtype='float')
    test_mask  = np.zeros((test_video.shape[0], test_video.shape[1]), dtype='float')

    for i in xrange(len(train_length)):
        train_mask[i,:train_length[i]]=1.0

    for i in xrange(len(valid_length)):
        valid_mask[i,:valid_length[i]]=1.0

    for i in xrange(len(test_length)):
        test_mask[i,:test_length[i]]=1.0

    #====================== Add 7th class =========================================
    trainL = seventhClass(emo_train_label, train_mask)
    validL = seventhClass(emo_valid_label, valid_mask)
    testL  = seventhClass(emo_test_label, test_mask)

    #=================== Add multilabel class =====================================
    emo_train_label = emotionClass(trainL)
    emo_valid_label = emotionClass(validL)
    emo_test_label  = emotionClass(testL)

#=================================================================================
def calc_valid_result_emotion(result, test_label, test_mask):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(test_label[i,j])
                predicted_label.append(result[i,j])
    true_label      = np.array(true_label)
    predicted_label = np.array(predicted_label)

    return true_label, predicted_label

def calc_valid_result_sentiment(result, test_label, test_mask):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
 