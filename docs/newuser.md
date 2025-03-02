# New Users Guide

Welcome new user! This document will be continually updated based on new user experiences. 

Table of Contents:

 - Introductions for Earth system scientists
 - Introduction for data scientists
 - Main features of PyEarthTools for data loading and processing
 - Main features of PyEarthTools for model definition and training
 - Main features of PyEarthTools for model inference

## Introduction for Earth System Scientists

PyEarthTools will greatly simplify your data access and data transformation code. 

Machine learning models must first be 'trained'. All machine learning models, from simple examples like linear regression to complex multidimensional neural networks (which may require huge computational resources), are based on the same principles. Model input is drawn from the sample data and presented to the model. The model then makes a prediction. This prediction may be correct or incorrect. The prediction is compared to the desired output (sometimes called the target value or truth value). That comparison is scored using a loss function. That loss function is then used to update the model based on the accuracy of the prediction. Sometimes this is done in small batches (e.g. 8 samples at once). This process is called model training.

The majority of the arduous effort in almost all machine learning projects is data preparation. There is also some effort in determining the proper model architecture.

Earth system science is typified by large, fairly open, standardised data sets which are well understood by the community, and will often already be held in institutional repositories. On top of that, there may be novel or project-specific data that can bring in additional sources of information, or be used to fine-tune standard models to provide improved performance in a new context.

A very large part of the purpose of PyEarthTools is to understand complicated Earth system science data, and then stream that data to the machine learning frameworks in matched input/output pairs so that the model can be trained.

## Introduction for Data Scientists

Many models include not only the model architecture and model weights, but also the data preprocessing and normalisation code involved in presenting data to the machine learning model framework for training. PyEarthTools separates the concerns of the data pipeline and the model architecture in a modular fashion. This allows model architectures to be swapped in and out independently from the data processing.

PyEarthTools also presents a somewhat-human-readable pipeline file (which can be both saved and loaded) which can give provenance to the data processing, model architecture, model weights and training strategy used in the production of a final model version. This allows a low-code approach to a reproducible research paradigm which also simplifies data access and management.


## Main Features of PyEarthTools for Data Loading and Processing

The first two features of PyEarthTools that a user is likely to be interested in are basic dataset access, and series-based dataset access. Basic dataset access provides a high-level API to load the data from disk into an in-memory xarray structure. Series-based access re-formulates that data into a batches sampling approach, often based on an iterator through the time dimension, which will process the on-disk data structure and present it in a form useful for a machine learning pipeline. Many users will be used to and expect to have to write their own dataset looping and traversal code, but this is not necessary with PyEarthTools unless a new data source has to be connected. Instead, there is a higher-level declarative API to specifying how to access and process data. The tutorial gallery has a sequence of tutorials which demonstrates all of these concepts, the user guide contains additional information, and there is also API documentation covering the details. Developing new dataset plugins is more complicated and will be covered in its own how-to guide in future.

Here is a short example of accessing the data archives, based on a date of interest, for two datasets (ERA5 and BRAN).

```python
import pyearthtools.data

##Common Date of interest
doi = '2022-02-12T000'

##Access ERA5
ERA5 = pyearthtools.data.archive.ERA5(variable = '2t', level = 'single')
##Access BRAN
BRAN = pyearthtools.data.archive.BRAN(variable = 'ocean_temp', type = 'daily')

## Retrieve Data
ERA5[doi] # Get ERA5 data at doi
BRAN[doi] # Get BRAN data at doi

```

## Main Features of PyEarthTools for Model Definition and Training

> [!NOTE]
> This section of the documentation is currently under development

## Main Features of PyEarthTools for Model Inference	

> [!NOTE]
> This section of the documentation is currently under development


	