# NCI User

The Data Science and Emerging Technologies team has set up modules on NCI for `pyearthtools`.

This will require access to project `ra02`,

## Modules

```shell
module use /scratch/ra02/modules
module load pyearthtools
```

This will allow `python` to be used, with `pyearthtools` and its dependencies installed. If other packages are needed, another path will need to be added containing those packages

## ARE

The following shows the config needed for ARE.

### Storage

Include `scratch/ra02`

### Module directories

```shell
/scratch/ra02/modules
```

### Modules

Load the `pyearthtools` module

```shell
pyearthtools
```
