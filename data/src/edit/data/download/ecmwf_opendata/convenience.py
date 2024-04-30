# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""Convenience classes for OpenData"""

import functools
from edit.data.download.ecmwf_opendata.opendata import OpenData


class AIFS(OpenData):
    @functools.wraps(OpenData.__init__)
    def __init__(self, *args, **kwargs):
        kwargs["model"] = "aifs"
        super().__init__(*args, **kwargs)


class IFS(OpenData):
    @functools.wraps(OpenData.__init__)
    def __init__(self, *args, **kwargs):
        kwargs["model"] = "ifs"
        super().__init__(*args, **kwargs)
