#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:42:13 2020

@author: willstith
"""

import numpy as np
from tensorflow import keras
from keras.applications import mobilenet_v2
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def prepare_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = mobilenet_v2.preprocess_input(x)
    return x

def predict(image_path):
    
    y_labels = {0: 'Dark_eyed_Junco',
             1: 'Brown_Thrasher',
             2: 'Common_Tern',
             3: 'Chipping_Sparrow',
             4: 'American_Crow',
             5: 'Pileated_Woodpecker',
             6: 'American_Goldfinch',
             7: 'Red_bellied_Woodpecker',
             8: 'Red_winged_Blackbird',
             9: 'Cardinal',
             10: 'Great_Crested_Flycatcher',
             11: 'Forsters_Tern',
             12: 'Common_Raven',
             13: 'House_Wren',
             14: 'Belted_Kingfisher',
             15: 'Cedar_Waxwing',
             16: 'White_breasted_Nuthatch',
             17: 'House_Sparrow',
             18: 'Scarlet_Tanager',
             19: 'Red_headed_Woodpecker',
             20: 'Downy_Woodpecker',
             21: 'Indigo_Bunting',
             22: 'Warbling_Vireo',
             23: 'Rose_breasted_Grosbeak',
             24: 'Barn_Swallow',
             25: 'Ring_billed_Gull',
             26: 'Gray_Catbird',
             27: 'Song_Sparrow',
             28: 'Northern_Flicker',
             29: 'Baltimore_Oriole',
             30: 'Blue_Jay',
             31: 'Tree_Swallow',
             32: 'White_crowned_Sparrow',
             33: 'Rock_Wren',
             34: 'Ruby_throated_Hummingbird',
             35: 'Mallard'}
    
    #load previously trained and saved model
    model = keras.models.load_model('first_streamlit_model/')
    
    #convert jpg file to numpy array for modeling
    image = np.array(prepare_image(image_path).reshape(224,224,3))
    image = np.expand_dims(image, axis=0)

    # predict the probability across all output classes
    yhat = model.predict(image)
    
    # return species name and probability for the most likely class
    label = y_labels[np.argmax(yhat[0])]
    label_prob = sorted(list(yhat[0]))[-1]
    return (label, label_prob)
