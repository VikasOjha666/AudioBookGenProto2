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
from PIL import Image

spell = SpellChecker()






char_list = string.ascii_letters+string.digits+',.?:;'


def find_dominant_color(image):
        #Resizing parameters
        width, height = 150,150
        image = image.resize((width, height),resample = 0)
        #Get colors from image object
        pixels = image.getcolors(width * height)
        #Sort them by count number(first element of tuple)
        sorted_pixels = sorted(pixels, key=lambda t: t[0])
        #Get the most frequent color
        dominant_color = sorted_pixels[-1][1]
        return dominant_color

def preprocess_img(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]]) 
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) # INTER_CUBIC interpolation best approximate the pixels image
                                                               # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

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
    word_img=preprocess_img(word_img,(128,32))
    word_img=np.expand_dims(np.expand_dims(word_img,axis=0),axis=-1)
    word=predict_word(word_img)
    return word




