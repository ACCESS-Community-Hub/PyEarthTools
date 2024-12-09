# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
The Environmental Data Intelligence Toolkit

Imports `utils`, `data`, and `pipeline` by default.

More can be imported with `import pyearthtools.SUBMODULENAME`
"""

__version__ = "0.1.0"

import pyearthtools.utils as utils
import pyearthtools.data as data
import pyearthtools.pipeline as pipeline

__all__ = ["utils", "data", "pipeline"]

def show_versions():
    """Show versions of installed pyearthtools modules"""
    import pyearthtools
    import inspect
    list_of_versions = []
    
    total_size = 20
    
    def add_padding(item: str) -> str:
        length = len(item)
        return item + ''.join([' ' for _ in range(total_size - length)])
        
    
    for module_name, module in inspect.getmembers(pyearthtools, inspect.ismodule):
        if module_name == '_version':
            module_name = 'pyearthtools'
        list_of_versions.append(f"{add_padding(module_name + ':')} \t {getattr(module, '__version__', None)}")
        
    print('pyearthtools - Versions:')
    for vers in list_of_versions:
        print(f"\t{vers}")
