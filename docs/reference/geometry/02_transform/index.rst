Transforms
----------

Coordinate-frame transformations for SE(2) and SE(3) poses and points.

Each function comes in two variants:

- **Array functions** (suffix ``_array``) operate on raw NumPy arrays and support
  batch dimensions.
- **Typed functions** (no suffix) accept and return typed geometry objects
  (:class:`~py123d.geometry.PoseSE2`, :class:`~py123d.geometry.PoseSE3`,
  :class:`~py123d.geometry.Point2D`, :class:`~py123d.geometry.Point3D`).


.. toctree::
   :maxdepth: 2

   01_transform_2d
   02_transform_3d
