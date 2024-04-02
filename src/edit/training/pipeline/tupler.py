# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from edit.pipeline import SequentialDecorator, DataOperation


@SequentialDecorator
class Tupler(DataOperation):
    """Make input tuple, with another element"""

    def __init__(self, *args, item=[]):
        super().__init__(*args, apply_func=self._func_apply, undo_func=None)
        self.item = item

    def _func_apply(self, *args):
        return (*args, self.item)
