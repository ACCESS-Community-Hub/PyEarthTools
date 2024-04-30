# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from tempfile import TemporaryDirectory

from edit.data import TimeIndex, AdvancedTimeIndex, DataIndex, CachingIndex, EDITDatetime
from edit.data.time import TimeDelta
from .utils.fakedata import fake_dataset


class FakeCachingIndex(CachingIndex):
    def __init__(self, variables, time, size, fill_value, **kwargs) -> None:
        super().__init__(cache=TemporaryDirectory(), **kwargs)
        self.variables = variables

        self.data = fake_dataset(variables, time, size, fill_value)

    def generate(self, *args, **kwargs):
        return self.data


class FakeDataIndex(TimeIndex, DataIndex):
    def __init__(self, variables, time_offset, size, fill_value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variables = variables
        self.time_offset = time_offset
        self.size = size
        self.fill_value = fill_value

    def get(self, querytime, *args, **kwargs):
        querytime = EDITDatetime(querytime)
        return fake_dataset(
            self.variables,
            [(querytime + TimeDelta(t)).qualified for t in self.time_offset],
            self.size,
            self.fill_value,
        )


class FakeOperatorIndex(AdvancedTimeIndex, DataIndex):
    def __init__(self, variables, time_offset, size, fill_value, data_interval, **kwargs) -> None:
        super().__init__(data_interval=data_interval, **kwargs)
        self.variables = variables
        self.time_offset = time_offset
        self.size = size
        self.fill_value = fill_value

    def get(self, querytime, *args, **kwargs):
        querytime = EDITDatetime(querytime)
        data = fake_dataset(
            self.variables,
            [(querytime + TimeDelta(t)).qualified for t in self.time_offset],
            self.size,
            self.fill_value,
        )
        return data
