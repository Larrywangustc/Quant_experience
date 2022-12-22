import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.models import model_from_json
import keras
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import *
from factor_lib import *
import random
from keras.losses import Loss
import tensorflow as tf
from backtest import backtest
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

i = 0
json_save_path = '%s%s%s_%s%s.json' % (model_save_dir, 'json/', version, model_name, i)
weights_save_path = '%s%s%s_%s%s.json' % (model_save_dir, 'weights/', version, model_name, i)


n_timesteps = 10

print(keras.__version__)
print(tf.__version__)

class CustomMSE(Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor
        self.enlarge_factor = 100.0
    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square((y_true - y_pred)*self.enlarge_factor))
        return mse

def build_model2(layers, neurons, d):
    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[2]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    adam = keras.optimizers.Adam(decay=0.001)
    model.compile(loss=CustomMSE(), optimizer=adam,  metrics=['accuracy'])
    model.summary()
    return model





if __name__ == '__main__':
    x_train = np.load(x_train_path, encoding="latin1", allow_pickle=True)
    y_train = np.load(y_train_path, encoding="latin1", allow_pickle=True)

    x_traintest = x_train[np.where(x_train[:, -1, 5] < testdate)]
    y_traintest = y_train[np.where(y_train[:, 1] < testdate)]
    x_backtest = x_train[np.where(x_train[:, -1, 5] >= testdate)]
    y_backtest = y_train[np.where(y_train[:, 1] >= testdate)]

    zz500 = pd.read_pickle('%sday_index.pkl' % (data_dir))
    zz500 = zz500[['date', 'open', 'close']]
    zz500 = zz500[zz500['date'] >= testdate]
    zz500 = zz500[zz500['date'] < train_end_date]
    zz500['chg_ratio'] = zz500['close'] / zz500.iloc[0, 1]
    zz500['chg_ratio'] = zz500['chg_ratio'].shift(-1)  # 往前移动一天

    t = 1
    k = 0

    order = np.arange(x_traintest.shape[0])
    while k < 50:
        '''N = range(x_traintest.shape[0])
        sample = np.random.choice(N, size=x_traintest.shape[0], replace=True)'''
        state = np.random.get_state()
        np.random.shuffle(x_traintest)
        np.random.set_state(state)
        np.random.shuffle(y_traintest)
        X_train = x_traintest[:, :, 6:]  # all data until day m
        amount_of_features = X_train.shape[2]
        shape = [amount_of_features, timesteps, 1]
        model = build_model2(shape, neurons, d)
        Y_train = y_traintest[:, 2]
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.02, patience=3, verbose=2,
                          mode='min', baseline=None, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, verbose=2, mode='min', min_delta=0.04,
                              min_lr=0.0001),
            ModelCheckpoint(filepath=weights_save_path, verbose=1, save_weights_only=True, save_best_only=True)
        ]

        model.summary()
        history = model.fit(
            X_train,
            Y_train,
            batch_size=4096,
            epochs=epochs,
            validation_split=0.1,
            verbose=1,
            callbacks=callbacks
        )
        json_string = model.to_json()
        open(json_save_path, 'w').write(json_string)
        model.load_weights(weights_save_path)
        pre_y = model.predict(x_backtest[:, :, 6:], batch_size=1024)
        data = np.c_[y_backtest, pre_y]
        pd.DataFrame(data).to_csv(predata_save_dir)
        result1, retracement1, pick_code1 = backtest(data, 10)
        result2, retracement2, pick_code2 = backtest(data, 20)
        result3, retracement3, pick_code3 = backtest(data, 30)
        result1 = pd.merge(result1, zz500, on='date')
        result2 = pd.merge(result2, zz500, on='date')
        result3 = pd.merge(result3, zz500, on='date')
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.plot(result1['value_sum'], label='pick10')
        plt.plot(result2['value_sum'], label='pick20')
        plt.plot(result3['value_sum'], label='pick30')
        plt.plot(result1['chg_ratio'], label='zz500')
        plt.title('{}epoch{} cumulative profit'.format(version, i))
        plt.legend()
        plt.savefig('%s%stest%s_plot.png' % (model_result_dir, version, i))
        print('Final result of 10 is:', result1.iloc[-1, 2])
        print('max_retracement of 10 is {}'.format(retracement1))
        print('Final result of 20 is:', result2.iloc[-1, 2])
        print('max_retracement of 20 is {}'.format(retracement2))
        print('Final result of 30 is:', result3.iloc[-1, 2])
        print('max_retracement of 30 is {}'.format(retracement3))
        result3.to_csv(result_save_dir)
        t = result3.iloc[-1, 2]
        if t > 1.8:
            i = i + 1
        k = k + 1








