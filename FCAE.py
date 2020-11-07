# -*- coding: utf-8 -*-
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Input, Lambda, Permute, MaxPooling2D, concatenate, Dense, Conv2D, MaxPooling2D, UpSampling2D, \
    Reshape
from keras.models import Model
from keras import backend as K
from sklearn.metrics import auc
from sklearn import metrics
import numpy as np
import shutil, os
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import keras.callbacks
from keras import activations
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import namedtuple, defaultdict, OrderedDict

from keras import initializers
import pandas as pd


def thresholds_calculate(fpr, tpr, thresholds):
    data_length = len(fpr)
    min_thresholds = 1
    min_index = 0
    for index in range(data_length):
        temp_num = (1 - tpr[index]) ** 2 + (0 - fpr[index]) ** 2

        if temp_num < min_thresholds:
            min_thresholds = temp_num
            min_index = index
    return thresholds[min_index]

def slice(x, w1, w2):
    """ Define a tensor slice function
    """
    x = x[:, :, :, w1:w2]

    return x


gc_element = defaultdict(list)
gc_element_name= ['ag_mean', 'as_mean', 'au_mean', 'cu_mean', 'pb_mean', 'sb_mean', 'zn_mean', 'cd_mean', 'nao_mean', 'ti_mean',
          'p_mean', 'cu_ture']

for ys in ysname:
    gc_element[ys].append(pd.read_excel("cu1_guiyihuan_input1.xlsx", header=None, sheet_name=ys))


def fcae(f, b):
    '''

    :param f:the number of fine-tuned iterations
    :param b:convolution window size
    :return: AUC
    '''
    input1 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x1 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv1')(input1)
    x1 = MaxPooling2D((3, 2), padding='same', name='Maxpool1')(x1)
    input2 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x2 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv2')(input2)
    x2 = MaxPooling2D((3, 2), padding='same', name='Maxpool2')(x2)
    input3 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x3 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv3')(input3)
    x3 = MaxPooling2D((3, 2), padding='same', name='Maxpool3')(x3)
    input4 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x4 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv4')(input4)
    x4 = MaxPooling2D((3, 2), padding='same', name='Maxpool4')(x4)
    input5 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x5 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv5')(input5)
    x5 = MaxPooling2D((3, 2), padding='same', name='Maxpool5')(x5)
    input6 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x6 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv6')(input6)
    x6 = MaxPooling2D((3, 2), padding='same', name='Maxpool6')(x6)
    input7 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x7 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv7')(input7)
    x7 = MaxPooling2D((3, 2), padding='same', name='Maxpool7')(x7)
    input8 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x8 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv8')(input8)
    x8 = MaxPooling2D((3, 2), padding='same', name='Maxpool8')(x8)
    input9 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x9 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv9')(input9)
    x9 = MaxPooling2D((3, 2), padding='same', name='Maxpool9')(x9)
    input10 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x10 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv10')(input10)
    x10 = MaxPooling2D((3, 2), padding='same', name='Maxpool10')(x10)
    input11 = Input(batch_shape=(1, 57, 92, 1))  # adapt this if using `channels_first` image data format
    x11 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv11')(input11)
    x11 = MaxPooling2D((3, 2), padding='same', name='Maxpool11')(x11)

    merge = concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=3)
    merge = Conv2D(16, (1, 1), activation='relu', padding='same', name='Convm1')(merge)
    merge = Conv2D(176, (1, 1), activation='relu', padding='same', name='Convm2')(merge)
    x_1 = Lambda(slice, arguments={'w1': 0, 'w2': 16})(merge)
    x_1 = UpSampling2D((3, 2), name='Ups_1')(x_1)
    x_1 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_1')(x_1)
    x_1 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_1d')(x_1)

    x_2 = Lambda(slice, arguments={'w1': 16, 'w2': 32})(merge)
    x_2 = UpSampling2D((3, 2), name='Ups_2')(x_2)
    x_2 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_2')(x_2)
    x_2 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_2d')(x_2)

    x_3 = Lambda(slice, arguments={'w1': 32, 'w2': 48})(merge)
    x_3 = UpSampling2D((3, 2), name='Ups_3')(x_3)
    x_3 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_3')(x_3)
    x_3 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_3d')(x_3)

    x_4 = Lambda(slice, arguments={'w1': 48, 'w2': 64})(merge)
    x_4 = UpSampling2D((3, 2), name='Ups_4')(x_4)
    x_4 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_4')(x_4)
    x_4 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_4d')(x_4)

    x_5 = Lambda(slice, arguments={'w1': 64, 'w2': 80})(merge)
    x_5 = UpSampling2D((3, 2), name='Ups_5')(x_5)
    x_5 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_5')(x_5)
    x_5 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_5d')(x_5)

    x_6 = Lambda(slice, arguments={'w1': 80, 'w2': 96})(merge)
    x_6 = UpSampling2D((3, 2), name='Ups_6')(x_6)
    x_6 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_6')(x_6)
    x_6 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_6d')(x_6)

    x_7 = Lambda(slice, arguments={'w1': 96, 'w2': 112})(merge)
    x_7 = UpSampling2D((3, 2), name='Ups_7')(x_7)
    x_7 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_7')(x_7)
    x_7 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_7d')(x_7)

    x_8 = Lambda(slice, arguments={'w1': 112, 'w2': 128})(merge)
    x_8 = UpSampling2D((3, 2), name='Ups_8')(x_8)
    x_8 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_8')(x_8)
    x_8 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_8d')(x_8)

    x_9 = Lambda(slice, arguments={'w1': 128, 'w2': 144})(merge)
    x_9 = UpSampling2D((3, 2), name='Ups_9')(x_9)
    x_9 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_9')(x_9)
    x_9 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_9d')(x_9)

    x_10 = Lambda(slice, arguments={'w1': 144, 'w2': 160})(merge)
    x_10 = UpSampling2D((3, 2), name='Ups_10')(x_10)
    x_10 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_10')(x_10)
    x_10 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_10d')(x_10)

    x_11 = Lambda(slice, arguments={'w1': 160, 'w2': 176})(merge)
    x_11 = UpSampling2D((3, 2), name='Ups_11')(x_11)
    x_11 = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_11')(x_11)
    x_11 = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_11d')(x_11)
    model_2 = Model(inputs=[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11],
                    outputs=[x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11])

    model_2.compile(optimizer='adadelta', loss='mean_squared_error', loss_weights=[1,
                                                                                    1,
                                                                                    1,
                                                                                    1, 1, 1, 1, 1, 1, 1, 1
                                                                                    ])
    for i in ['ag_mean', 'as_mean', 'au_mean', 'cu_mean', 'pb_mean', 'sb_mean', 'zn_mean', 'cd_mean', 'nao_mean',
              'ti_mean', 'p_mean']:
        model_2.load_weights('my_model_weights' + i + '.h5', by_name=True)

    model_2.load_weights('my_model_weights2.h5', by_name=True)

    input_data = []
    for i in ['ag_mean', 'as_mean', 'au_mean', 'cu_mean', 'pb_mean', 'sb_mean', 'zn_mean', 'cd_mean', 'nao_mean',
              'ti_mean', 'p_mean']:
        input_data.append(gc_element[i][0].values.astype(float).reshape(1, 57, 92, 1))
    model_2.fit(input_data,
                input_data,
                epochs=f, batch_size=1

                )

    cu_ppint = gc_element['cu_point'][0]
    model_2.save_weights("temp/my_model_weights_final.h5")
    result = model_2.predict(x=input_data)
    true_mineral = cu_ppint
    pre_result = 0
    for i in range(len(result)):
        pre_result += ((result[i].reshape(5244, 1) - input_data[i].reshape(5244, 1)) ** 2)
    pre_result = pd.DataFrame(pre_result)
    pre_result = np.sqrt(pre_result)
    true_mineral = true_mineral.values
    true_mineral = true_mineral.reshape(1, 57, 92, 1)
    true_mineral = true_mineral.reshape(5244, 1)
    true_mineral = true_mineral.astype(int)
    true_mineral = pd.DataFrame(true_mineral)
    true_mineral[true_mineral > 0] = 1
    aaa = pre_result.values
    aaa = aaa.reshape(-1)
    true_mineral = true_mineral.values
    true_mineral = true_mineral.reshape(-1)
    fpr, tpr, thresholds = metrics.roc_curve(true_mineral, aaa)

    AUC = auc(fpr, tpr)

    return AUC


def train(e, b, f):
    '''e is the number of pre-trained iterations,
     b convolution window size, 
     f is the number of fine-tuned iterations'''
    auc_fae_list = []

    os.mkdir("temp")
    input = Input(shape=(57, 92, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv1')(input)
    x = MaxPooling2D((3, 2), padding='same', name='Maxpool1')(x)
    x = UpSampling2D((3, 2), name='Ups1')(x)
    x = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_1')(x)
    decoded = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_1d')(x)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    dataframe = gc_element['ag_mean'][0]
    dataset = dataframe.values
    X = dataset.astype(float)
    X = X.reshape(1, 57, 92, 1)
    true_aa = X.reshape(5244, 1)
    true_aa = pd.DataFrame(true_aa)
    autoencoder.fit(X, X,
                    epochs=e,

                    batch_size=1,

                    )

    autoencoder.save_weights("temp/my_model_weightsag_mean.h5")

    # Get the output of the 1st pooling layer
    maxpool1_layer_model = Model(autoencoder.input, outputs=autoencoder.get_layer('Maxpool1').output)
    aa = pd.DataFrame(autoencoder.predict(X).reshape(5244, 1))
    bbb = (true_aa - aa) * (true_aa - aa)

    maxpool1_output = maxpool1_layer_model.predict(X)
    maxpool1_output_sum = maxpool1_output
    # Get the output of the other pooling layers

    cov_index = 2
    elements = ['as_mean', 'au_mean', 'cu_mean', 'pb_mean', 'sb_mean', 'zn_mean', 'cd_mean', 'nao_mean', 'ti_mean',
                'p_mean']
    for element in elements:

        dataframe = gc_element[element][0]
        dataset = dataframe.values
        X = dataset.astype(float)
        X = X.reshape(1, 57, 92, 1)

        true_aa = X.reshape(5244, 1)
        true_aa = pd.DataFrame(true_aa)

        input = Input(shape=(57, 92, 1))  # adapt this if using `channels_first` image data format
        x = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv' + str(cov_index))(input)
        x = MaxPooling2D((3, 2), padding='same', name='Maxpool' + str(cov_index))(x)
        x = UpSampling2D((3, 2), name='Ups' + str(cov_index))(x)
        x = Conv2D(16, (b, b), activation='relu', padding='same', name='Conv_' + str(cov_index))(x)
        decoded = Conv2D(1, (1, 1), activation='sigmoid', name='Conv_' + str(cov_index) + 'd')(x)
        autoencoder = Model(input, decoded)
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        autoencoder.fit(X, X,
                        epochs=e,

                        batch_size=1,

                        )

        autoencoder.save_weights("temp/my_model_weights" + element + ".h5")


        maxpool1_layer_model = Model(autoencoder.input,
                                     outputs=autoencoder.get_layer('Maxpool' + str(cov_index)).output)
        maxpool1_output = maxpool1_layer_model.predict(X)
        aa = pd.DataFrame(autoencoder.predict(X).reshape(5244, 1))
        bbb += ((true_aa - aa) * (true_aa - aa))
        maxpool1_output_sum = np.concatenate([maxpool1_output_sum, maxpool1_output], axis=3)
        cov_index = cov_index + 1

    input_merge = Input(shape=(19, 46, 176))
    merge = Conv2D(16, (1, 1), activation='relu', padding='same', name='Convm1')(input_merge)
    out_merge = Conv2D(176, (1, 1), activation='relu', padding='same', name='Convm2')(merge)
    autoencoder2 = Model(input_merge, out_merge)
    autoencoder2.compile(optimizer='adadelta', loss='mean_squared_error')
    input_data = maxpool1_output_sum
    autoencoder2.fit(input_data, input_data,
                     epochs=e,

                     batch_size=1,

                     )

    autoencoder2.save_weights("temp/my_model_weights2.h5")
    auc_fae = fcae(f, b)
    auc_fae_list.append(auc_fae)

    shutil.move("temp","result_"+str(auc_fae))


if __name__ == '__main__':
    train(300, 23, 50)
