from typing import Tuple

class SlicedTensor:
    def __init__(self, tensors):
        self.tensors = tensors
        shape_0 = sum(map(lambda t : t.shape[0], self.tensors))
        self._shape = (shape_0, ) + self.tensors[0].shape[1:]

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    def get_range(self, idx_start: int, idx_end: int) -> 'SlicedTensor':
        output_list = []
        idx = 0
        for tensor in self.tensors:
            if idx >= idx_start:
                output_list.append(tensor)
            idx += tensor.shape[0]
            if idx == idx_end:
                break
            elif idx > idx_end:
                raise ValueError('Unaligned slice size')

        return SlicedTensor(output_list) 

    def __iter__(self):
        for tensor in self.tensors:
            yield tensor