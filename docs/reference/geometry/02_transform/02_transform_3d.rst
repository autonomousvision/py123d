Transforms in 3D
^^^^^^^^^^^^^^^^

Functions for converting SE(3) poses ``(x, y, z, qw, qx, qy, qz)`` and 3D points
``(x, y, z)`` between coordinate frames. Rotations are represented as unit
quaternions. For the 2D counterpart see :doc:`01_transform_2d`.


Convert SE3 poses between frames
---------------------------------

Convert between absolute and relative coordinates:

- :func:`~py123d.geometry.transform.abs_to_rel_se3_array` /
  :func:`~py123d.geometry.transform.abs_to_rel_se3` --
  absolute → relative: :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}`

- :func:`~py123d.geometry.transform.rel_to_abs_se3_array` /
  :func:`~py123d.geometry.transform.rel_to_abs_se3` --
  relative → absolute: :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}`

- :func:`~py123d.geometry.transform.reframe_se3_array` /
  :func:`~py123d.geometry.transform.reframe_se3` --
  re-express poses from one reference frame to another.

.. autofunction:: py123d.geometry.transform.abs_to_rel_se3_array

.. autofunction:: py123d.geometry.transform.abs_to_rel_se3

.. autofunction:: py123d.geometry.transform.rel_to_abs_se3_array

.. autofunction:: py123d.geometry.transform.rel_to_abs_se3

.. autofunction:: py123d.geometry.transform.reframe_se3_array

.. autofunction:: py123d.geometry.transform.reframe_se3

Convert 3D points between frames
---------------------------------

The same absolute/relative/reframe operations, applied to 3D points instead of
SE3 poses:

- :func:`~py123d.geometry.transform.abs_to_rel_points_3d_array` /
  :func:`~py123d.geometry.transform.abs_to_rel_point_3d`

- :func:`~py123d.geometry.transform.rel_to_abs_points_3d_array` /
  :func:`~py123d.geometry.transform.rel_to_abs_point_3d`

- :func:`~py123d.geometry.transform.reframe_points_3d_array` /
  :func:`~py123d.geometry.transform.reframe_point_3d`

.. autofunction:: py123d.geometry.transform.abs_to_rel_points_3d_array

.. autofunction:: py123d.geometry.transform.abs_to_rel_point_3d

.. autofunction:: py123d.geometry.transform.rel_to_abs_points_3d_array

.. autofunction:: py123d.geometry.transform.rel_to_abs_point_3d

.. autofunction:: py123d.geometry.transform.reframe_points_3d_array

.. autofunction:: py123d.geometry.transform.reframe_point_3d


Translation along body-frame axes
----------------------------------

Translate SE3 poses or 3D points along their local (body) coordinate axes.
The orientation is preserved.

.. autofunction:: py123d.geometry.transform.translate_se3_along_body_frame

.. autofunction:: py123d.geometry.transform.translate_se3_along_x

.. autofunction:: py123d.geometry.transform.translate_se3_along_y

.. autofunction:: py123d.geometry.transform.translate_se3_along_z

.. autofunction:: py123d.geometry.transform.translate_3d_along_body_frame
