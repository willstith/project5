#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 20:28:58 2020

@author: willstith
"""

#imports
import numpy as np
import seaborn as sns
sns.set_style('dark')
import os


import keras
from keras.applications import mobilenet_v2
from keras.preprocessing import image

### Code to create folders for each species
# root_path = '/Users/willstith/Desktop/sorted_bird_pics/'
# folders = []
# for k, v in y_labels.items():
#     folders.append(v)
    
# for folder in folders:
#     os.mkdir(os.path.join(root_path,folder))
###


def prepare_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = mobilenet_v2.preprocess_input(x)
    return x


def predict(image_path):
    
#extend this list as needed to include more species
#REMEMBER: must also modify training data and final layer of model
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
    model = keras.models.load_model('MNV2_species/')
    
    #convert jpg file to numpy array for modeling
    image = np.array(prepare_image(image_path).reshape(224,224,3))
    image = np.expand_dims(image, axis=0)

    # predict the probability across all output classes
    yhat = model.predict(image)
    
    # return species name and probability for the most likely class
    label = y_labels[np.argmax(yhat[0])]
    label_prob = sorted(list(yhat[0]))[-1]
    return (label, label_prob)

def sort_image(path, file):
    starting_folder = path + file
    species = predict(starting_folder)[0]
    destination = '/Users/willstith/Desktop/sorted_bird_pics/' + species + '/' + file
    #'/'.join(starting_folder.split('/')[:-1]) + species
    os.rename(starting_folder, destination)
    print(destination)
    
unsorted_folder = '/Users/willstith/Desktop/unsorted_bird_pics/'

for idx, file in enumerate(os.listdir(unsorted_folder)):
    
    sort_image(unsorted_folder, file) 
    print(file)
    
print('Done sorting')