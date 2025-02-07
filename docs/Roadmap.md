# PyEarthTools Roadmap

Code health (may take some time):
 - Improve test coverage
 - Improve docstring coverage
 - Add badges to GitHub

Project setup:
 - Set up the “main” branch for release preparation
 - PyPI versions
 - Write a contributor guide
 - Write a new user guide
 - Re-evaluate the tech setup used to build the current docs and see if we can easily move it over to a sphinx-base stack
 - Readthedocs site (probably just track develop rather than the latest release, I guess). Figure out whether each submodule gets its own readthedocs or whether there’s just one. Maybe go with just the one.
 - CI/CD setup
 - Reproduce installation and tutorials

Immediate goals:
 - Get the tutorial sorted and working
 - Reproduce the tutorial from the main repo in some more places
 - Bundle the tutorial model as as part of the default model registry

Community goals:
 - Get PyEarthTools set up at NCI in a supported environment
 - Get some NRI contributors onboarded for development
 - Write much clearer new user documentation
 - See if NRI want to create an AussieWeatherBench

Roadmap:
 - Create some more tutorials, including for different model architectures, perhaps look at point-based NN example, perhaps do a simplified site prediction model with XGBoost
 - Create a new user tutorial or doc covering how to download initial data for predictions, analysis, point obs and maybe radar/satellite. Could look at WeatherReal for point data.
 - Bundle FourCastNeXt into the default model registry as a low-cost training architecture
 - Create an XGBoost example to show how to work with non-NN models and alternative frameworks
 - Expand the support for fetching and managing datasets
 - Improve pipelines so there can be pipelines-of-pipelines, reverse pipelines, and staged pipelines (e.g. data load, train, inference, verify, kfold, withhold key cases, split-by-region).
 - Write a “new project” guide for using PyEarthTools on novel problems
 - Write a developer guide
 - Include general documentation on neural networks for geosciences
 - Give examples of how to fine-tune models for new data
 - Implement drift detection routines for model assessment post-training, e.g. 6-months later or a year later
 - Write an integration guide for MLOps
 - Add support for inference debugging in applied contexts

Extended goals:
 - Work out which models to bundle and maintain
 - See if we can integrate with various “benches” easily (e.g. easy submission to WeatherBench, ExtremeWeatherBench)
 - Assess code-level overlap with Anemoi to see if would be possible to create common shared libraries
 - Attempt to do an integration test of an Anemoi-generated model into PyEarthTools
 - Develop a collection of open-weight foundation models which can be downloaded from a registry like HuggingFace into the default model registry

Scientific goals:
 - Set up standard scaling factors for use across models for standard variables

Speculative ideas:
 - Integrate adverserial testing and other computer-science based testing regimens into a verification pipeline
 - Create a default verification pipeline