# Pipeline Entrypoints

As `PyEarthTools` pipelines propose a generic way to load and prepare various earth system datasets, it is possible to use
a pipeline as a source for [anemoi-datasets](https://anemoi.readthedocs.io/projects/datasets/en/latest/).

## Example

Below is a minimal example of using a `PyEarthTools` pipeline to load data and prepare it for `anemoi`, please see the `anemoi` docs
for more information on the `datasets` config.

### Create the Pipeline in PyEarthTools

.. code-block:: python

    import pyearthtools.data
    import pyearthtools.pipeline

    pipeline = pyearthtools.pipeline.Pipeline(
        pyearthtools.data.download.arcoera5.ARCOERA5(['t2m', 'u10', 'v10']),
        pyearthtools.pipeline.operations.xarray.values.FillNan()
    )
    pipeline.save('/PATH/TO/PIPELINE.yaml')

### Create the anemoi-datasets config

.. code-block:: yaml

    name: pyearthtools_to_anemoi
    description: PyEarthTools Pipeline converted to Anemoi
    attribution: PyEarthTools

    dates:
        start: '2025-11-10T00:00:00'
        end: '2025-11-12T00:00:00'
        frequency: 1h

    input:
        pyearthtools: # Use the pyearthtools input object
            pipeline: /PATH/TO/PIPELINE.yaml

### Run anemoi-datasets

.. code-block:: bash

    anemoi-datasets create /path/to/anemoi/dataset.yaml

## Function Contract

The expected contract and result from the `PyEarthTools` pipeline is to return an `xarray` object of a single time index.

Both tools provide methods to modify the metadata of the data, and should be used accordingly to prepare for downstream uses.
