# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
