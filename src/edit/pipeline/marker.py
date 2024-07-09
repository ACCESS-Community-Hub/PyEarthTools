# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import xarray as xr

from edit.pipeline.operation import PipelineStep


def find_shape(obj):
    if hasattr(obj, "shape"):
        return obj.shape

    if isinstance(obj, xr.Dataset):
        return tuple(obj[d].shape for d in obj.data_vars)


class Marker(PipelineStep):
    """
    Marker in a pipeline

    Useful for graph notes.
    """

    def __init__(self, text: str, shape: str = "note", print: bool = False, print_shape: bool = False):
        """
        Pipeline marker

        Args:
            text (str):
                Text to display in graph
            shape (str, optional):
                Shape for graph. Defaults to 'note'.
            print (bool, optional):
                Whether to print `sample` when running. Defaults to False.
        """
        super().__init__()
        self.record_initialisation()

        self.text = text
        self.shape = shape
        self._print = print
        self._print_shape = print_shape

    def run(self, sample):
        if self._print:
            to_print = sample if not self._print_shape else find_shape(sample)
            print(f"At marker {self.text!r} sample was:\n{to_print}\n")
        return sample
