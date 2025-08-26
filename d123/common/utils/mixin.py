from __future__ import annotations

import copy as pycopy

import numpy as np
import numpy.typing as npt


class ArrayMixin:
    """Abstract base class for geometric entities."""

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """The array representation of the geometric entity."""
        raise NotImplementedError

    def __array__(self, dtype: npt.DtypeLike = None, copy: bool = False) -> npt.NDArray:
        array = self.array
        return array if dtype is None else array.astype(dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Return the length of the array."""
        return len(self.array)

    def __getitem__(self, key):
        """Allow indexing into the array."""
        return self.array[key]

    def __eq__(self, other) -> bool:
        """Equality comparison based on array values."""
        if isinstance(other, ArrayMixin):
            return np.array_equal(self.array, other.array)
        return False

    @property
    def shape(self) -> tuple:
        """Return the shape of the array."""
        return self.array.shape

    def tolist(self) -> list:
        """Convert the array to a Python list."""
        return self.array.tolist()

    def copy(self) -> ArrayMixin:
        """Return a copy of the object with a copied array."""
        obj = pycopy.copy(self)
        obj.array = self.array.copy()
        return obj
