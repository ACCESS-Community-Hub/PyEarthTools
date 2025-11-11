# New Project Guide

To get started with PyEarthTools, please see the [new user guide](newuser.md).

This page is not about getting started, but is about how to undertake a larger project such as developing a new model.

## Overview

**Step One:** Set up a place for your project (e.g. a folder on disk) to hold the source code, data and notebooks you will need. Consider using [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) which will automate the process of creating a standard data science project layout on disk including creating a Python package structure for your code. Doing so will give all of your projects a consistent layout and structure. It's fine to do things differently, but this is a way to get started consistently with a documentated approach.

**Step Two:** Load and visualise your data. Read [the data API how-to](/api/data/data_how_to.md) for more information fetching and adding data.

**Step Three:** Set up a data pipeline to load your data, get the data onto a common grid, and normalise them. Read [the pipeline API how-to](/api/pipeline/pipeline_how_to.md) for more information.

**Step Four:** Train an initial model to establish a baseline. There are several reference architectures bundled in the framework and one of them should do for starters. Read the [models how-to guide](api/models/models_how_to.md) for more information.

**Step Five:** Review the standard evaluation scorecard for your baseline. Read the [evaluation how-to guide](api/evaluation/evaluation_how_to.md) for more information.

## An Example

Let's imagine you want to improve the temperature predictions at your location.

You will train the model using historical model data and historical weather station data. To obtain and integrate station data, see the [HadISD tutorials](project:/notebooks/Gallery.md#Working-with-Station-Data-(Medium-Hardware-Requirements)). To obtain and integrate model (or, technically, reanalysis) data, you can use the NCI site archive. If you are working elsewhere, see [Downloading ERA5 Data](project:/notebooks/tutorial/Downloading_ERA5.ipynb). Note, the ERA5 tutorial demonstrates downloading low-resolution data, there may be a more accurate source of model data for your location.

From there, configure the data accessors. The tutorial examples demonstrate how to do this for the downloaded data.

Then, make a pipeline. Work out which variables you want, subset the grid points you want, and normlise the data. Take a look at the tutorial on [Working with Multiple Data Sources](./notebooks/tutorial/MultipleSources) and [MLX Demo](./notebooks/tutorial/MLX-Demo-Custom-Arch) to see how to approach constructing the pipeline, and refer to [api/pipeline/howto.md](api/pipeline/pipeline_how_to.md) for a more in-depth how-to guide on this process.

Visualise some of the samples from the pipeline, and make sure the data looks right. Maybe do a plot of the historical difference between the gridded value and the point value, to see how the two things are different.

There are a number of ways to train the baseline model. One of the easiest is to use the XGBoost framework, because it's robust and computationally lightweight. There is no tutorial example of this yet, but you might like to look at the [MLX Demo](./notebooks/tutorial/MLX-Demo-Custom-Arch) and the [CNN Demo](./notebooks/tutorial/CNN-Model-Training) for inspiration. There are a lot of nuances here for how to manage a large project where you might be running dozens or hundreds of experiments, but the easiest place to start is a single model trained in a Jupyter Notebook. Experiment management information will be added later.

Evaluating the model is up to you at this point. They PyEarthTools roadmap includes the development of standard scorecards for an out-of-the-box experience, but for now check out the [scores](https://scores.readthedocs.io/) framework for verification.
