# Models

Using `pyearthtools`'s data, pipeline and model tools, it has been possible to create 'built' models for easy experimentation and testing.

Currently implemented are:

| Model | Source |
| ----- | ------ |
| SFNO  | NVIDIA |
| GraphCast | Google |
| PanguWeather | Huawei Cloud |
| Fuxi | Fudan University|
| FengWu | Shanghai Artificial Intelligence Laboratory |

See [here](/pyearthtools/models/commands/) for a detailed breakdown of the commands.

!!! Warning "Internet Connection"
    If using the 'Live' sources, or running for the first time, an internet connection is normally required.

    Therefore, running the `--data` version of the command on an internet enabled node may be neccessary before prediction can be done. This will ensure the data and model is correctly downloaded.

!!! Note "Normalisation"
    Some pipelines require normalisation values which are currently stored on GADI in `dx2`.

!!! Note "Usage"
    `pyearthtools.models` can be used either as a command line tool, or as a python package.
    See [here](/pyearthtools/models/commands/) and below for the commands, and [here](/pyearthtools/models/programatic/) for the python package.

## Setup

To use these models, they and the rest of `pyearthtools` must be installed.

A setup guide can be found [here](/pyearthtools/started/installation/).

### Assets

Assets will be automatically downloaded to `~/.pyearthtools/models/assets` or if set environment variable `pyearthtools_MODELS_ASSETS`.

## GADI Module

If you have access to GADI, you can use the prebuilt modules

### Step 1 - Access to `ra02`

Join [`ra02`](https://my.nci.org.au/mancini/project/ra02)

This project contains all the environments and assets needed to quickly get up and running.

### Step 2 - Create a GPU Compute Node

These models must be run on a compute node, ideally a gpu enabled one.

#### Storage Flags

You will need to add a storage flag for the project you wish to save data to, as well as the projects you wish to use data from.

| Dataset | Project | Flag |
| ------- | -------- | ---- |
| ERA5   | [`rt52`](https://my.nci.org.au/mancini/project/rt52) | gdata/rt52 |
| ACCESS | [`wr45`](https://my.nci.org.au/mancini/project/wr45) | gdata/wr45 |

#### Creating a gpu interactive script

Below are two pbs header files. One for the v100's and the other for the dgxa100's.

Depending on gadi's mood one may be quicker then the other.

##### GPUVolta

```bash
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -l jobfs=100GB
#PBS -q gpuvolta
#PBS -P [PROJECT_CODE_HERE]
#PBS -N GPU_Compute
#PBS -l walltime=8:00:00
#PBS -l storage=scratch/ra02+gdata/ra02+[DATA_PATHS_HERE]
#PBS -l wd
#PBS -I
```

##### DXGA100

```bash
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -l jobfs=100GB
#PBS -q dgxa100
#PBS -P [PROJECT_CODE_HERE]
#PBS -N GPU_Compute
#PBS -l walltime=8:00:00
#PBS -l storage=scratch/ra02+gdata/ra02+[DATA_PATHS_HERE]
#PBS -l wd
#PBS -I
```

Copy one of the above headers into a new file on GADI, and fill in the missing parts. Once done, simply `qsub` it. This will create a new interactive compute job.

```shell
nano ~/gpu_file.pbs
```

```shell
qsub ~/gpu_file.pbs
```

???+ info "Live Data"

    `pyearthtools.models` uses the `cdsapi` provided by ECMWF to get live ERA5 data, but this will require some set up.

    Following the steps outlined here [cdsapi](https://github.com/ecmwf/cdsapi)

    Get your user ID (UID) and API key from the CDS portal at the address <https://cds.climate.copernicus.eu/user>
    and write it into the configuration file, so it looks like

    ```txt
    $ cat ~/.cdsapirc
    url: https://cds.climate.copernicus.eu/api/v2
    key: <UID>:<API key>
    verify: 0
    ```

    When selecting Live Data, it must be cached somewhere for the gpu nodes to access it without needing to download it. 
    You will need to add in any storage flags pertaining to where you wish to cache this data.

    Specifying `--data_cache` sets this cache directory.

### Step 3 - Loading Modules

Using the command line within the new interactive job, run the following commands to setup your module environment.

```shell
module use /scratch/ra02/modules
module load pyearthtools/models
```

??? note "Check Module Setup"

    Running `module list` should now show that the modules are loaded

    ```
    module list
    # Currently Loaded Modulefiles:
    #  1) pyearthtools/models/VERSION 
    ```

#### Step 3.1 - Set environment variable

These models require their weights and normalisation factors to be downloaded (which cannot be done on an interactive node).

???+ tip "GADI (USING pyearthtools/models)"
    As the environment is already setup you don't need to worry.

??? warning "GADI (NOT USING pyearthtools/models)"
    To use the predownloaded assets on GADI, run the following on your cmd,

    ```shell
    export pyearthtools_MODELS_ASSETS=/g/data/ra02/.pyearthtools/models/assets/
    ```

??? warning "Other Systems"
    If working on another system, either specify `pyearthtools_MODELS_ASSETS` every time, or just allow the default path to be used.

### Step 4 - Running a Forecast

With an interactive job now running, running ```pyearthtools-models interactive``` will now provide a series of prompts to run a forecast.

See [here](/pyearthtools/models/commands/) for a more detailed breakdown of the commands.

Below is an example of how this can be used,

<terminal-window>
    <terminal-line data="input">pyearthtools-models interactive</terminal-line>
    <terminal-line lineDelay=200 typingDelay=10 data="prompt">Which model would you like to use? ['Global/graphcast', 'Global/sfno']: sfno</terminal-line>
    <terminal-line data="output">INFO Loading sfno</terminal-line>
    <terminal-line typingDelay=10 data="prompt">Which pipeline / data source do you want to use? ['ACCESS', 'ERA5', 'ERA5(Live)']: ERA5 </terminal-line>
    <terminal-line typingDelay=10 data="prompt">Where would you like to save the data?: ~/DATADIRECTORY/ </terminal-line>
    <terminal-line typingDelay=10 data="prompt">Required keyword arg: 'lead_time'? [int | str]: 24 hours</terminal-line>
    <terminal-line typingDelay=10 data="prompt">Other kwargs []? (click format): </terminal-line>
    <terminal-line typingDelay=10 data="prompt">Time to predict for? : 2023-01-25T12</terminal-line>
    <terminal-line lineDelay=5000 data="progress">Predicting</terminal-line>
</terminal-window>

You can also do this as a single line, not using the interactive commands, such as below,

```shell
pyearthtools-models predict 'sfno' --pipeline 'ERA5' --output '~/DATADIRECTORY/' --time '2023-01-25T12' --lead_time '24 hours'
```

The interactive command will also provide the single line command for later use, so you don't have to use the interactive prompts repeatedly.
