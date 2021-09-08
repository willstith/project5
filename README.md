# project5

## Classifying Bird Images Using CNNs and Transfer Learning

**Created by Will Stith**  
**09/17/2020**

**Introduction**

This folder contains files comprising my fifth and final project for Metis. The aim of the project was to use convolutional neural networks (CNNs) and transfer learning to create a program which could sort my bird photos into folders based on the species of bird captured in the photo. The Caltech-UC Berkeley 200-2011 labelled bird image dataset, obtained from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html, comprised the bulk of the training data. In addition, some personal photos were added to some classes to expand the training set. The modeling process involved first building a ground-up CNN using Keras on a simply binary classification between two species. The model was improved iteratively by increasing the number of classes, adjusting layers of the network, and most importantly, incorporating transfer learning to use the MobileNetV2 model architecture pretrained on ImageNet as a base for my final network. The final model (MNV2_species) achieved a 61.9% accuracy score across 36 species. Using this final model's predictions, I also created a program which will sort bird photos into folders corresponding to the species identified in the photo.

**Requirements**

Python 3.?
Necessary modules (listed below)

**Necessary modules**

- pickle
- pandas
- matplotlib
- seaborn
- numpy
- sklearn
- keras
- collections
- os

**Data**

The data was downloaded from Caltech's Computer Vision Laboratory website at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html. The dataset includes 11,788 high quality bird images labeled across 200 different classes (species). Classes were nearly perfectly balanced, with around 60 images per class. Also included are bounding boxes and parts annotations for each image, but these were not used in this project. Some personal photos were added to bolster the training set, and the idea is to continue improving this model by more such additions, including eventually adding new classes.

In addition, a MobileNetV2 model pretrained on ImageNet served as the base for the final model. This model is found in the Keras library, but more information on it and on ImageNet can be found at the links below:
MobileNetV2: https://arxiv.org/abs/1801.04381
ImageNet: http://www.image-net.org/

**Contents**

*project5_workbook.ipynb* - contains code used to build the various versions of my models, including the final model (MNV2).  
*bird_sorter.py* - the primary functional output of the project. This python script will automatically sort bird photos uploaded to a specified folder into folders corresponding to the species identified within them by the model saved from project5_workbook.ipynb.

**Instructions**

To recreate this project, first download image data from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and move image folders to .../Desktop/birdup/birds_by_species on your local system. Run the project5_workbook to create and save the final model (Note: sections 3 - 3.2.1.2 should be skipped if only interested in the final model). Next, to use the bird sorter, load images to the computer and move them to .../Desktop/unsorted_bird_pics folder. If using the sorter for the first time, open bird_sorter.py and uncomment the code under the imports, which will create the sorted folders on your system when ran (note: be sure to modify file paths to suit your local system). Run bird_sorter.py to sort photos into their proper locations. This should be followed up by manual corrections to ensure satisfactory accuracy.

If you would like to extend the model to include more classes, three changes must be made. First, create a new folder in .../Desktop/birdup/birds_by_species with the name of the new species and containing example images of that species. Second, the first parameter of the final (dense) layer of the final model (in project5_workbook.ipynb) should match the total number of classes (species) you have. Third, modify the y_labels dictionary within the predict function of bird_sorter.py to match the y_labels dictionary from project5_workbook.ipynb.

**Acknowledgements**

A huge thank you to all my Metis instructors and TAs for their support and guidance for this project and others. As well I'd like to thank my fellow Metis students for contributing to a wonderful learning environment that was collaborative, challenging, and constructive. Additional thanks to Caltech and UC-Berkeley for compiling the training dataset and releasing it for public use, and to all persons who contributed to the many libraries that this project and previous projects depended upon.
