from __future__ import annotations

import numpy as np
import numpy.typing as npt


class ArrayMixin:
    """Mixin class for object entities."""

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], copy: bool = True) -> ArrayMixin:
        """Create an instance from a NumPy array."""
        raise NotImplementedError

    @classmethod
    def from_list(cls, values: list) -> ArrayMixin:
        """Create an instance from a list of values."""
        return cls.from_array(np.array(values, dtype=np.float64), copy=False)

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
        return self.__class__.from_array(self.array, copy=True)
