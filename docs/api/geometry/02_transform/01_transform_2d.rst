Transforms in 2D
^^^^^^^^^^^^^^^^

Functions for converting SE(2) poses ``(x, y, yaw)`` and 2D points ``(x, y)``
between coordinate frames. For the 3D counterpart see :doc:`02_transform_3d`.


Convert SE2 poses between frames
---------------------------------

Convert between absolute and relative coordinates:

- :func:`~py123d.geometry.transform.abs_to_rel_se2_array` /
  :func:`~py123d.geometry.transform.abs_to_rel_se2` --
  absolute → relative: :math:`T_\text{rel} = T_\text{origin}^{-1} \cdot T_\text{abs}`

- :func:`~py123d.geometry.transform.rel_to_abs_se2_array` /
  :func:`~py123d.geometry.transform.rel_to_abs_se2` --
  relative → absolute: :math:`T_\text{abs} = T_\text{origin} \cdot T_\text{rel}`

- :func:`~py123d.geometry.transform.reframe_se2_array` /
  :func:`~py123d.geometry.transform.reframe_se2` --
  re-express poses from one reference frame to another.

.. autofunction:: py123d.geometry.transform.abs_to_rel_se2_array

.. autofunction:: py123d.geometry.transform.abs_to_rel_se2

.. autofunction:: py123d.geometry.transform.rel_to_abs_se2_array

.. autofunction:: py123d.geometry.transform.rel_to_abs_se2

.. autofunction:: py123d.geometry.transform.reframe_se2_array

.. autofunction:: py123d.geometry.transform.reframe_se2


Convert 2D points between frames
---------------------------------

The same absolute/relative/reframe operations, applied to 2D points instead of
SE2 poses:

- :func:`~py123d.geometry.transform.abs_to_rel_points_2d_array` /
  :func:`~py123d.geometry.transform.abs_to_rel_point_2d`

- :func:`~py123d.geometry.transform.rel_to_abs_points_2d_array` /
  :func:`~py123d.geometry.transform.rel_to_abs_point_2d`

- :func:`~py123d.geometry.transform.reframe_points_2d_array` /
  :func:`~py123d.geometry.transform.reframe_point_2d`

.. autofunction:: py123d.geometry.transform.abs_to_rel_points_2d_array

.. autofunction:: py123d.geometry.transform.abs_to_rel_point_2d

.. autofunction:: py123d.geometry.transform.rel_to_abs_points_2d_array

.. autofunction:: py123d.geometry.transform.rel_to_abs_point_2d

.. autofunction:: py123d.geometry.transform.reframe_points_2d_array

.. autofunction:: py123d.geometry.transform.reframe_point_2d


Translation along body-frame axes
----------------------------------

Translate SE2 poses or 2D points along their local (body) coordinate axes.
The yaw / orientation is preserved.

.. autofunction:: py123d.geometry.transform.translate_se2_along_body_frame

.. autofunction:: py123d.geometry.transform.translate_se2_along_x

.. autofunction:: py123d.geometry.transform.translate_se2_along_y

.. autofunction:: py123d.geometry.transform.translate_se2_array_along_body_frame

.. autofunction:: py123d.geometry.transform.translate_2d_along_body_frame
