
from edit.training.data.sequential import SequentialIterator
from edit.training.data.templates import DataOperation, DataStep

@SequentialIterator
class DataFunctionOperation(DataOperation):
    def __init__(self, index: DataStep | DataOperation):
        super().__init__(index, apply_func=self._apply, undo_func=self._undo, split_tuples=True, doc= "User Defined Operation")
    
    def _apply(self, data):
        raise NotImplementedError()
    def _undo(self, data):
        raise NotImplementedError()