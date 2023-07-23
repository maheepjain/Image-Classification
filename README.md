# Abstract 

## Introduction 

This project will examine the accuracy of different classification models by using the CIFAR-10 data. This dataset is a condensed version of the CIFAR100 dataset that the Canadian Institute for Advanced Research created. This image dataset consists of 60,000, 32x32 images with 10 classes of images and 6000 images per class. 

**Image classification** is a solution to the problem of assigning and analyzing images based on their visual content. Image classification is perhaps the most important part of digital image analysis. Digital image analysis is the key to object recognition, surveillance systems, quality control, and medical diagnosis. With image classification, we can conclude the contents of given images to further aid with these digital image analysis applications. 

## Objectives 

The CIFAR10 dataset is well studied, however, we want to see if we can improve on the methods to quickly and efficiently create high-accuracy classification models. We would like to create a fast and high-accuracy (>90%) model. 

As we have already developed Logistic Regression and Convolutional Neural Networks (CNN), we want to add a Random Forest model to determine whether we can improve our accuracy. 

## Methodology 

In terms of our methodology, we will be focusing on Logistic Regression, Random Forest, and CNN models. 

**Logistic Regression** is a statistical model that can be used for binary classification problems. As data is in 10 classes, we can use logistic regression to identify whether a particular image belongs to a class or not. 

**Random Forest** has the advantages of fast training speed and high classification accuracy which is essential for image classification problems as image recognition needs a lot of time in the training process. 

**CNN** helps reduce the high dimensionality of images without losing any information which can help us achieve accurate results in image classification problems. 


## Installation and Execution

### Installation

To run this project, you will need to install the following packages

```bash
!pip install matplotlib
!pip install keras
!pip install tenserflow
!pip install seaborn
!pip install scikit-learn
```

### Execution

`__main__.py` calls functions from all the scripts for all three models to pre-process the data, train the models, predict the models and visualize them. The plots will be saved on the system once you run them. A total of 4 different pages will be saved to visualize the results of our models.


## Authors
### Abigail Lee - 200469770 
Worked on and created Random Forest Model
### Maheep Jain - 203386460
Worked on and created Logistic Regression Model and CNN Model


    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── __main__.py    <- Main script calling all the functions for all the models
    │
    ├── <- Scripts to download or generate data and pre-process the data
    │   └── pre_processing.py
    │
    ├── <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    |
    ├── <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── <- Scripts to create exploratory and results-oriented visualizations
        └── visualize.py           

