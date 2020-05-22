import logging

import numpy as np
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal import get_window
from librosa.core import amplitude_to_db
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

logger = logging.getLogger('Adityatb Model')


def compute_soft_mask(y1, y2):
    y1 = np.abs(y1)
    y2 = np.abs(y2)
    m1 = np.divide(y1, np.add(y1, y2))
    m2 = np.divide(y2, np.add(y1, y2))
    # m2 = 1 - m1
    return [m1, m2]


def soft_masking(y):
    input = y[0]
    y1_hat = y[1]
    y2_hat = y[2]
    s1, s2 = compute_soft_mask(y1_hat,y2_hat)
    y1_tilde = np.multiply(s1, input)
    y2_tilde = np.multiply(s2, input)
    return [y1_tilde, y2_tilde]


def masked_out_shape(shape):
    shape_0 = list(shape[0])
    shape_1 = list(shape[1])
    return [tuple(shape_0),tuple(shape_1)]


class Adityatb:
    def __init__(self, checkpoint=None):
        self.batch_size = 32
        self.reg = 0.05
        self.learning_rate
        self.n_units = 600
        self.decay = 1e-3
        self.input_sampling_rate = 11025
        self.n_samples_window = 1024
        self.overlap = 0.5
        self.
        if checkpoint:
            logger.info('Model checkpoint input obtained')
            self.__model = keras.models.load_model(checkpoint)
        else:
            logger.info('Creating new model')
            self.__createModel()

    def __createModel(self):
        regularizer = l2(self.reg)
        input = Input(shape=shape)
        hid1 = LSTM(self.n_units, return_sequences=True, activation='relu')(input)
        dp1 = Dropout(0.2)(hid1)
        hid2 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp1)
        dp2 = Dropout(0.2)(hid2)
        hid3 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp2)
        y1_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]),
                                 name='y1_hat')(hid3)
        out1 = Lambda(soft_masking, masked_out_shape, name='softMask')([input, y1_hat])

        self.__model = Model(inputs=input, outputs=out1)
        self.__model.summary()

        opt = Adam(lr=self.learning_rate, decay=self.decay)
        self.__model.compile(loss='kullback_leibler_divergence',
                             optimizer=opt,
                             metrics=['acc', 'mse'])

    def __save(self, filename):
        logger.info('Save model to file ' + filename)
        self.__model.save(filename)

    def __resample(self, input_signal, input_sampling_rate):
        factor = input_sampling_rate/self.input_sampling_rate
        logger.info('Input sampling rate is different from the expected by the model.\n' +
                    '\rInput sampling rate: ' + str(input_sampling_rate) + '\n' +
                    '\rModel sampling rate: ' + str(self.input_sampling_rate) + '\n' +
                    'Resampling input signal by factor: ' + str(factor))
        in_signal = decimate(input, factor)
        return in_signal

    def __prepateInput(self, input_signal, sampling_rate):
        if sampling_rate != self.input_sampling_rate:
            input_signal = self.__resample(input_signal, sampling_rate)
        freq, time, stft = spectrogram(
            input_signal, fs=int(),
            window=get_window('hann', self.n_samples_window),
            # nperseg=None,
            noverlap=int(self.overlap*self.n_samples_window), nfft=self.n_samples_window,
            # detrend='constant',
            return_onesided=True, scaling='spectrum', axis=-1, mode='complex')
        db_values = amplitude_to_db(np.abs(stft))
        return db_values

    def __prepareOutput(self, output):
        pass

    def train(self, xTrain, yTrain, sampling_rate):

        self.__model.fit(xTrain, yTrain,
                         batch_size=self.batch_size,
                         epochs=1,
                         validation_split=0.1)
