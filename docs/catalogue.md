# Data Catalogues

Table of Contents

- What, why and how of data cataloguing
- Connecting directly to cloud data storage
- Creating a project-based data catalogue
- Creating an on-disk catalogue
- Using the "Intake" package to facilitate catalogue management
- Separating project data from general open data
- Creating new datasets for sharing and cataloguing

## Introduction

PyEarthTools can connect seamlessly to various cloud data sources. The current best technological choice appears to be data stored in the Zarr format, stored in an open data bucket. Earth system data is distributed in a huge variety of technologies, formats and license agreements. It is not possible to provide a zero-effort-required mechanism to totally abstract data access behind an API.

As such, a few things remain the onus of the user: (1) determining the license conditions and appropriate uses of the data (2) registration, signup and payment for access to data (3) whether to adopt technologies like the Intake package and (4) how to manage privacy and security controls around the data. Fortunately, a lot of data which is of research interest is shared openly and freely, so getting started with a useful data set is feasible for individual researchers on self-managed equipment, although the largest use cases will require HPC facilities and hundreds of terabytes (or even petabytes) of storage. There is no sign of any slowdown in data volume growth, so probably we will soon be talking about exabytes, zettabytes and whatever may come next.

The good news is that even though there is no end to data volume increases, there is also cutting-edge research being done on data sets of 1 terabyte or less, which is well within grasp of most research groups to source, store and manage.

## The role of data in science and machine learning

Data is essential in three ways. 

Firstly, access to open, trusted data sets is essential to being able to commence an academic research project into machine learning. Most researchers will get their start on data the can access easily. There are few things which have more impact than publishing high-quality open research benchmark datasets for stimulating research into a particular field or area. 

Secondly, machine learning systems which are only trained on closed data sources are unlikely to be trusted by fellow researchers, leading to a lack of research interest in the field. Research thrives on re-use of prior research and prior data, inspiring people to investigate, compete and innovate. While not all data needs to be made public, providing enough to enable methodological research is important, particularly for validation and verification purposes.

Thirdly, publishing research (or even operational/official) data from a new model allows others to evaluate, compare and assess the efficacy of models. 

## Connecting directly to cloud storage

It is technically possible to access all data via a network, without involving a local on-disk cache. For model inference, this is reasonably efficient. For model training which required repeated access of the relevant data, using cloud storage directly is very inefficient. 

## Creating an on-disk catalogue

1. Manually creating an on-disk catalogue through dataset replication, and connecting to it with PyEarthTools
2. Connecting to a cloud-based or remote data store, and using PyEarthTools to create an on-disk cache

