import cv2
import numpy as np
import string
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K

from spellchecker import SpellChecker
import tensorflow as tf

spell = SpellChecker()






char_list = string.ascii_letters+string.digits


def preprocess_img(img):
    h,w=img.shape
    new_width=0
    new_height=0
    if w>128:
        img=cv2.resize(img,(128,h))
        new_width=128
    else:
        new_width=w
    if h>32:
        img=cv2.resize(img,(new_width,32))
        new_height=32
    else:
        new_height=h
    if w<128:
        add_ones=np.ones((new_height,128-new_width))*255
        #print(f" First block Image shape={img.shape} Pad shape={add_ones.shape}")
        img=np.concatenate((img,add_ones),axis=1)
        new_width=128
    if h<32:
        add_ones=np.ones((32-new_height,new_width))*255
        #print(f"Second block Image shape={img.shape} Pad shape={add_ones.shape}")
        img=np.concatenate((img,add_ones))
        new_height=32
    
    
    # Normalize each image
    img = img/255
    img = np.expand_dims(img , axis = 2)
    img=np.expand_dims(img,axis=0)
    return img


def predict_word(img):
    pred=act_model.predict(img)
    out = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1],
                         greedy=True)[0][0])
    word=[]
    for char in out[0]:
      word.append(char_list[char])

    return spell.correction(''.join(word))
 



inputs = Input(shape=(32,128,1))
 

conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)

pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)

batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 

blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)


act_model = Model(inputs, outputs)

act_model.load_weights('CRNN_model.hdf5')



def recognize_word(word_img):
    word_img=preprocess_img(word_img)
    word=predict_word(word_img)
    return word




