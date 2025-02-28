# New Users Guide

Welcome new user! This document will be continually updated based on new user experiences. 

## Introduction for Earth System Scientists

PyEarthTools will greatly simplify your data access and data transformation code. All machine learning models, from simple examples like linear regression to complex multidimensional neural networks which require huge computational resources are based on the same principle. Model input is drawn from the sample data and presented to the model. A prediction, right or wrong, is generated. That prediction is compared to the desired output (sometimes called the target value or truth value). That comparison is scored using a loss function. That loss function is then used to update the model based on the accuracy of the prediction. Sometimes this is done in small batches (like 8 samples at once). That's the whole thing.

The majority of the arduous effort in almost all machine learning projects is data preparation. There is some effort in determining the proper model architecture.

A very large part of the purpose of PyEarthTools is to understand complicated Earth System Science data, and then stream that data to the machine learning frameworks in matched input/output pairs so that the model can be trained.

Earth System Science is typified by large, fairly open, standardised data sets which are well understood by the community, and will often already be held in institutional repositories. On top of that, there may be novel or project-specific data that can bring in additional sources of information, or be used to fine-tune standard models to provide improved performance in a new context.

## Introduction for Data Scientists

Many models include not only the model architecture and model weights, but also the data preprocessing and normalisation code involved in presented data to the machine learning model framework for training. PyEarthTools separates the concepts of the data pipeline and the model architecture in a modular fashion. This allows model architectures to be swapped in and out independently from the data processing.

PyEarthTools also presents a somewhat-human-readable pipeline file (which can be both saved and loaded) which can give provenance to the data processing, model architecture, model weights and training strategy used in the production of a final model version. This allows a low-code approach to a reproducible research paradigm which also simplifies data access and management.
