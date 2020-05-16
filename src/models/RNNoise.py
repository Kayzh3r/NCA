import logging

from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.constraints import min_max_norm

logger = logging.getLogger('RNNoise')


def rnnCrossentropy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


def mask(y_true):
    return K.minimum(y_true + 1., 1.)


def msse(y_true, y_pred):
    return K.mean(mask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


def rnnCost(y_true, y_pred):
    return K.mean(mask(y_true) * (10 * K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(
        K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01 * K.binary_crossentropy(y_pred, y_true)), axis=-1)


def rnnAccuracy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class RNNoise:
    def __init__(self, checkpoint=None):
        self.batch_size = 32
        self.reg = 0.000001
        self.constraint = WeightClip(0.499)
        if checkpoint:
            logger.info('Model checkpoint input obtained')
            self.__model = keras.models.load_model(checkpoint)
        else:
            logger.info('Creating new model')
            self.__createModel()

    def __createModel(self):
        mainInput = Input(shape=(None, 42), name='main_input')
        tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=self.constraint,
                    bias_constraint=self.constraint)(mainInput)
        vadGRU = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru',
                     kernel_regularizer=regularizers.l2(self.reg), recurrent_regularizer=regularizers.l2(self.reg),
                     kernel_constraint=self.constraint, recurrent_constraint=self.constraint,
                     bias_constraint=self.constraint)(tmp)
        vadOutput = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=self.constraint,
                          bias_constraint=self.constraint)(vadGRU)
        noiseInput = keras.layers.concatenate([tmp, vadGRU, mainInput])
        noiseGRU = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru',
                       kernel_regularizer=regularizers.l2(self.reg), recurrent_regularizer=regularizers.l2(self.reg),
                       kernel_constraint=self.constraint, recurrent_constraint=self.constraint,
                       bias_constraint=self.constraint)(noiseInput)
        denoiseInput = keras.layers.concatenate([vadGRU, noiseGRU, mainInput])
        denoiseGRU = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                         name='denoise_gru', kernel_regularizer=regularizers.l2(self.reg),
                         recurrent_regularizer=regularizers.l2(self.reg), kernel_constraint=self.constraint,
                         recurrent_constraint=self.constraint, bias_constraint=self.constraint)(denoiseInput)

        denoiseOutput = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=self.constraint,
                              bias_constraint=self.constraint)(denoiseGRU)

        self.__model = Model(inputs=mainInput, outputs=[denoiseOutput, vadOutput])
        self.__model.compile(loss=[rnnCost, rnnCrossentropy],
                             metrics=[msse],
                             optimizer='adam', loss_weights=[10, 0.5])

    def __save(self, filename):
        logger.info('Save model to file ' + filename)
        self.__model.save(filename)

    def __prepateInput(self, input):

    def __prepareOutput(self, output):

    def train(self, xTrain, yTrain, voiceAvailable):

        self.__model.fit(xTrain, [yTrain, voiceAvailable],
                         batch_size=self.batch_size,
                         epochs=1,
                         validation_split=0.1)
