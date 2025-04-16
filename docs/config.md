# Configuration Guide

The default configuration file location is ~/.config/pyearthtools. This directory may contain multiple .yaml files which control different aspects of the configuration. The directory is known as the configuration directory.

```{Note} 
The rest of this document describes the **intended** functionality of the configuration system and served to capture requirements only. The current behaviour is not documented and should be considered deprecated. Once the indended functionality is sufficiently well-documented, work will commence towards implementing the requirements.
```

- By default, the configuration directory is ~/.config/pyearthtools  
- This location may changed by setting the environment variable PYEARTHTOOLS_CONFIG_DIRECTORY  
- The location may also be changed at run-time by calling `pyearthtools.config.load_config(custom_directory)`

The following files may be created in the configuration directory:

|  Filename      | Configuration Purpose               | Notes  |
|----------------|-------------------------------------|--------|
| data.yaml      | File formats and standard caching configuration  | Defaults are sensible
| logger.yaml    | Log file information                | Will log to the workspace directory by default
| pipeline.yaml  | Parallelisation and dask config     | Will use dask by default but will not start client workers
| workspace.yaml | Working directories, experiments, and over-rides | Will cache to ~/petproject/ by default
| project.yaml   | Project-specific data sets (may be outside the workspace) | Empty by default
| site.yaml      | Standard data sources for a given computing facility (likely to be outside the workspace) | Will be created by site admins
| default.yaml   | Default directories and cloud dataset URLs | Will be populated on first setup with standard data URLs



