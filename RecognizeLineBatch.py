import numpy as np
import cv2
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import math
from RecognizeWord import recognize_word
os.environ['CUDA_VISIBLE_DEVICES']='-1'


spell = SpellChecker()

def pad_img(img):
    old_h,old_w=img.shape[0],img.shape[1]

    #Pad the height.

    #If height is less than 512 then pad to 512
    if old_h<512:
        to_pad=np.ones((512-old_h,old_w))*255
        img=np.concatenate((img,to_pad))
        new_height=512
    else:
    #If height >512 then pad to nearest 10.
        to_pad=np.ones((roundup(old_h)-old_h,old_w))*255
        img=np.concatenate((img,to_pad))
        new_height=roundup(old_h)

    #Pad the width.
    if old_w<512:
        to_pad=np.ones((new_height,512-old_w))*255
        img=np.concatenate((img,to_pad),axis=1)
        new_width=512
    else:
        to_pad=np.ones((new_height,roundup(old_w)-old_w))*255
        img=np.concatenate((img,to_pad),axis=1)
        new_width=roundup(old_w)-old_w
    return img

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs,conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model

model=unet()
model.load_weights('./word_seg_model.h5')

def sort_word(wordlist):
    wordlist.sort(key=lambda x:x[0])
    return wordlist

def recognize_line(line_img_arr):
    processed_line_imgs=[]
    corresponding_ori_imgs=[]
    pred_arrays=[]

    for im in line_img_arr:
        img=pad_img(im)
        ori_img=img.copy()
        ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
        img=cv2.resize(img,(512,512))
        img=np.expand_dims(img,axis=-1)
        img=img/255

        processed_line_imgs.append(img)
        corresponding_ori_imgs.append(ori_img)
    processed_line_imgs=np.array(processed_line_imgs)


    pred_arrays=model.predict(processed_line_imgs)
    del processed_line_imgs

    for i in range(len(pred_arrays)):
        pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)
        pred=cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.threshold(pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,pred)
        contours, hier = cv2.findContours(pred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        (H, W) = corresponding_ori_imgs[i].shape[:2]
        (newW, newH) = (512, 512)
        rW = W / float(newW)
        rH = H / float(newH)

        coordinates=[]

        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a white rectangle to visualize the bounding rect
            # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
            coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))

        coordinates=sort_word(coordinates)
        word_counter=0

        # for (x1,y1,x2,y2) in coordinates:
        #     temp_img=ori_img[y1:y2,x1:x2]
        #     cv2.imwrite(str(word_counter)+'.jpg',temp_img)
        #     word_counter+=1

        sentence=''
        
        for (x1,y1,x2,y2) in coordinates:
            temp_img=corresponding_ori_imgs[i][y1:y2,x1:x2]
            word=recognize_word(temp_img)
            word=spell.correction(word)
            sentence+=word+' '
        print(sentence)





