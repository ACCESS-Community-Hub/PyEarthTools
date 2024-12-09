# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Register archives for automatic importing depending on the system being run on.

# pyearthtools.data.archive Registry

To ease complexity for an end user, `pyearthtools.data` archives can be automatically imported if the conditions in a config file are met.

This directory `pyearthtools.data.archives.registered` contains those config files which are checked on `pyearthtools.data` `__init__` and thus can add in the archives.

It is assumed that any `archive` for `pyearthtools.data` is using the `register_archive`, so that when the module is imported, the archives are added.

## Structure

These config files are simple `yaml` files with certain keys,

All non `module` keys will be checked according to below, and if **all** pass, then the module will be imported.

| Key | Purpose | Input Type |
| --- | ------- | ----- |
| `module` | Name of python module of archive which will be attempted to be imported | str |
| `folder` | Check if folder exists | str |
| `file`   | Check if file exists | str |
| `env`    | Get environment variable, and check for str in it. If list, check for existence. | dict[str, str] or list |
| `hostname` | Check if given str is in `hostname` from `socket.gethostname()` | str |
| `any`    | Run any other tests with an or | dict |

If multiple of the same key are needed, add a `[x]` with `x` being any text to mark them as different.

If more advanced checks are needed, feel free to add or request them

## Register

To add a new config file, add it under this directory, and then add the file name to `register.txt`.

## Example

```yaml
#
# Template Register
#
module: MODULENAME_HERE

folder: /this/folder/means/something
file: /so/does/this.file
file[1]: wow/look/another/file.cool

hostname: wow.look.i.am.here
env:
  I_AM_SET
env[1]:
  I_CONTAIN: ME_TEXT

any:
  folder: i/might/need/to/exist
  file: but/if/i/do/it/doesnt

```

## Root Directories

To add more flexibility for the user, the root directories are usually stored as a property of the `FileSystemIndex` and can be overwritten with use  of the `pyearthtools.data.archive.set_root` function.

This root directory dictionary, is sourced from `pyearthtools.data.archive.ROOT_DIRECTORIES`, and thus may need to be provided by any archive module. This can be done by

```python
from pyearthtools.data.archive import register_archive

register_archive('ROOT_DIRECTORIES')(ROOT_DIRECTORIES)

```

## Deprecation

If it is neccessary to deprecate a registered package, setting `deprecated` in the register config file to `true` will raise a warning if a user attempts to use it.

"""
