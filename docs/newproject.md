# New Project Guide

## Overview

**Step One:** Set up a place for your project. Consider using [Cookie Cutter Data Science](https://cookiecutter-data-science.drivendata.org). This will give all of your projects a consistent layout and structure regardless of what kind of data science project is done. Install this and use their instructions to start a new project. It's fine to do things differently, but this is a way to get started consistently with a documentated approach. 

**Step Two:** Load and visualise your data. Read [the data API how-to](/api/data/data_how_to.md) for more information fetching and adding data.

**Step Three:** Set up a data pipeline to load things, get them onto a common grid, and normalise them. Read [the pipeline API how-to](/api/pipeline/pipeline_how_to.md) for more information.

**Step Four:** Train an initial model to establish a baseline. There are several reference architectures bundled in the framework and one of them should do for starters. Read the [models how-to guide](api/models/models_how_to.md) for more information.

**Step Five:** Review the standard evaluation scorecard for your baseline. Read the [evaluation how-to guide](api/evaluation/evaluation_how_to.md) for more information.

## An Example

Let's imagine you want to improve the temperature predictions at your location. 

You will train the model using historical model data and historical point data. If you are working at NCI, these things are available from the standard data accessors in the NCI site archive. If you haven't obtained any data yet, check out [https://herbie.readthedocs.io/en/stable/](https://herbie.readthedocs.io/en/stable/) for the means to download historical model data, and see [https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) or [https://www.metoffice.gov.uk/hadobs/hadisd/](https://www.metoffice.gov.uk/hadobs/hadisd/) for access to weather station data. If you don't have the data on disk yet, put it into your data/raw directory. Take note, there is a lot of data here, so you will probably want to work out how to download only what you need.

From there, configure the data accessors. The PyEarthTools accessors are currently being extended so that there are a range of accessors for fetching cloud data and working with standard datasets, but for now look at the examples in the tutorial for how to create one for your own data.

Then, make a pipeline. Work out which variables you want, subset the grid points you want, and normlise the data. Take a look at the tutorial on [Working with Multiple Data Sources](./notebooks/tutorial/MultipleSources) and [MLX Demo](./notebooks/tutorial/MLX-Demo-Custom-Arch) to see how to approach constructing the pipeline, and refer to [api/pipeline/howto.md](api/pipeline/pipeline_how_to.md) for a more in-depth how-to guide on this process.

Visualise some of the samples from the pipeline, and make sure the data looks right. Maybe do a plot of the historical difference between the gridded value and the point value, to see how the two things are different. 

There are a number of ways to train the baseline model. One of the easiest is to use the XGBoost framework, because it's robust and computationally lightweight. There is no tutorial example of this yet, but you might like to look at the [MLX Demo](./notebooks/tutorial/MLX-Demo-Custom-Arch) and the [CNN Demo](./notebooks/tutorial/CNN-Model-Training) for inspiration. There are a lot of nuances here for how to manage a large project where you might be running dozens or hundreds of experiments, but the easiest place to start is a single model trained in a Jupyter Notebook. Experiment management information will be added later.

Evaluating the model is up to you at this point. They PyEarthTools roadmap includes the development of standard scorecards for an out-of-the-box experience, but for now check out the [scores](https://scores.readthedocs.io/) framework for verification.