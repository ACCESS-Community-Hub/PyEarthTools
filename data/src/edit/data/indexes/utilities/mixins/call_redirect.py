# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from typing import Any


class CallRedirectMixin:
    """
    Provide overrides for `__func__`'s that redirect to `__call__`
    """

    def __call__(self, *args):
        raise NotImplementedError

    def __matmul__(self, key: Any):
        """@ accessor

        Expands tuple or lists passed
        """
        if isinstance(key, (list, tuple)):
            return self.__call__(*key)

        elif isinstance(key, dict):
            return self.__call__(**key)

        return self.__call__(key)

    def __getitem__(self, idx: Any):
        """[] accessor"""
        return self.__call__(idx)
