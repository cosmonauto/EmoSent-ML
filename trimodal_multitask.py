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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_sc