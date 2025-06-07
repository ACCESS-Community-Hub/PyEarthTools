# New Project Guide

## A Quick Hands-On Approach

This guide is suitable for scientists or anyone else who wants to start trying things quickly to establish their first model and make a first attempt. More detail is provided below with more detail on the nuances and alternatives for each step.

1. Use [https://pyearthtools.readthedocs.io/en/latest/notebooks/tutorial/FourCastMini_Demo.html](https://pyearthtools.readthedocs.io/en/latest/notebooks/tutorial/FourCastMini_Demo.html) as a template for what to do.
1. Determine the parameters you want to model, such as `temperature` or `wind`. When these become part of the neural network, they will be called *channels*.
2. Determine the data source they come from, such as ERA5 or another model or re-analysis source
3. Develop a `pipeline` which includes data normalisation
4. Using a bundled model, configure that model to the size required. This may only required the adjustment of `img_size`, `in_channels` and `out_channels` to match the size of your data. The grid dimension must be a multiple of four for this model, so you may need to crop or regrid your data to match. In future, a standard approach without this limitation will be added. 
5. Run some number of training steps (using the `.fit` method) and visualise the outputs. Visualising predictions from the trained model every 3000 steps or so provides useful insight into the training process as well as helping see when the model might be fully trained. *There is no definite answer to how much training will be required. If your model isn't showing any progress at all after a couple of epochs, there may be a problem. Some models will start to show progress after 3000 steps.*

This approach should be a usable starting point for any gridded inputs and outputs. The example is based on global modelling, but could reasonably be applied to nowcasting, observational data, limited area modelling, or just anything you can represent in an xarray on a grid. You could even add a grid containins data from a weather station at each grid point and see what happens.

Getting a neural network to perform well and make optimal predictions is very hard, with many nuances. Getting started should be reasonably simple.

The sections below go into more detail on how to treat source data, how to develop the most suitable pipeline for your project, how to use alternative neural network architectures, how to manage the training process, and how to perform a more thorough evaluation of the outputs.

## Metholodogical Information

This guide offers a simple, repeatable process for undertaking a machine learning project. Experts in machine learning will recognise this as a standard approach, but of course it can be adapted as required in the project. Completing a project (whether using PyEarthTools or not) comprises the following steps:

1. Identify the sources of data that you wish to work with
2. Fetch the training, testing and validation data
3. Connect PyEarthTools to local data, forming the data catalog
4. Transform the data into a normal form suitable for machine learning
5. Define your neural network (or other machine learning architecture)
6. Perform network training using the training and validation data
7. Test the data using out-of-sample test data
8. Evaluate the model using an evaluation scorecard

Note this page is a work in progress! In time, a tutorial for each step of this process will be added.

## Identifying sources of data

Step 1 is up to you, and will vary enormously by project, with an almost totally open-ended set of possibilities. This includes both input and target data. Many people working in weather model will have in mind a symmetrical model which runs auto-regressively. This is the case for many physical models, whereby the physical initial conditions contain an identical set of variables to the output of the model, but this is not necessarily the case in either physical modelling and even less so in machine learning.

If you aren't sure where to start, work through the tutorials, which will introduce several interesting data sets. If you would like some additional ideas, take a look at the [project ideas](projectideas.md) page and see if anything sparks your creativity.

# Fetch the training, testing and validation data

Machine learning models will typically use all of the data available, and will access that data many times. For that reason, it is usually much more efficient to replicate data onto the local disk of the machine you are working on, rather than relying exclusively on network-accessible data.

Additional functionality will come in time to ease the process of fetching data. Our main user base operates at research facilities with on-site data already in place. However, even in that setting, additional novel data is often required for a new project.

## Connect PyEarthTools to local data, forming the data catalogue

Connecting PyEarthTools to data sources is a significant part of the functionality of the package. For well-known community data sets, PyEarthTools requires only simple configuration and offers a significant benefit to users. The following data sets are well-understood by PyEarthTools:

- ERA5
- Himawari 8
- CMIP5
- ACCESS model data

In addition to simple loading of data, PyEarthTools undertakes the following services:

- Renaming of variables to a standard and compatible naming convention
- Re-projecting all datasets onto a common latitude and longitude reference grid
- Derives variables on-the-fly where relevant
- Provides a common method to access variables from any data source

## Transform the data into a normal form suitable for machine learning

This step is often considered to be part of the "model" in a machine learning project. Neural networks generally prefer input data which ranges between approximately -1 to +1, with some caveats and variations applying. An example of a common approach to this would be to divide all temperature values observed by the maximum observed temperature, applying a normalisation approach which can be easily reversed (so long as you keep track of that max observed temperature). A more statistical approach might be to subtract the mean from each value and divide the result by the standard deviation.

These approaches may not always be suitable for a variety of reasons, and may result in incompatibilities or undesirable and arbitrary differences between models resulting from whatever choices happen to have been made. PyEarthTools seeks to remove the arbitrary nature of many of these choices, offering standardised normalisation factors which can then improve interoperability between data sources and model architectures.

For users working with standard data sources which are already supported by PyEarthTools, this is an out-of-the-box experience, saving a significant amount of time and difficulty which is not always apparent at the start of a project. For users wishing to connect a new data source, or publish a new benchmark data set, integrating the normalisation code into PyEarthTools will allow all users to easily adopt new data sources for machine learning.

## Define your neural network (or other machine learning architecture)

(This section coming soon)

## Perform network training using the training and validation data

(This section coming soon)

## Test the data using out-of-sample test data

(This section coming soon)

The ultimate tests for a machine learning model are its ability to effectively transfer its learned skill onto new data, and maintain its performance over time. For this reason, some data should be reserved for evaluation which is never used for training.

## Evaluate the model using an evaluation scorecard

(This section coming soon)

All physical models should ideally be compared to the following benchmarks:

1. Performance against "persistence"
2. Performance against "climatology"
3. Performance against a physical model
4. Performance against a simple multilayer perceptron or other simplified architecture
5. Performance against a best-in-class ML model (if one is available)

In addition, a range of evaluative metrics should be used.
