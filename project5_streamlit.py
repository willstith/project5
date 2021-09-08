#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:39:40 2020

@author: willstith
"""

from io import BytesIO, StringIO
import streamlit as st 
from PIL import Image
from streamlit_funcs import predict
import numpy as np


st.title("What's This Bird?")

file = st.file_uploader("Choose an image...", type=["jpg"])
#text_io = io.TextIOWrapper(file)
#data = text_io.read()

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Your Image', use_column_width=True)
    st.write("")
    
if isinstance(file, BytesIO):    
    st.write("Classifying...")
    label = predict(np.array(file))
    #rewrite line below so that label[1] is the string name of the class and label[2]*100 is % certainty
    st.write('%s (%.2f%%)' % (label[0], label[1]*100))
