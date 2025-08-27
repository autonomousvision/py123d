
Geometry
========

Geometric Primitives
--------------------

Points
~~~~~~
.. autoclass:: d123.geometry.Point2D()

.. autoclass:: d123.geometry.Point3D()

Vectors
~~~~~~~
.. autoclass:: d123.geometry.Vector2D()

.. autoclass:: d123.geometry.Vector3D()

Special Euclidean Group
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: d123.geometry.StateSE2()

.. autoclass:: d123.geometry.StateSE3()

Bounding Boxes
~~~~~~~~~~~~~~
.. autoclass:: d123.geometry.BoundingBoxSE2()

.. autoclass:: d123.geometry.BoundingBoxSE3()

Indexing Enums
~~~~~~~~~~~~~~
.. autoclass:: d123.geometry.Point2DIndex()

.. autoclass:: d123.geometry.Point3DIndex()

.. autoclass:: d123.geometry.Vector2DIndex()

.. autoclass:: d123.geometry.Vector3DIndex()

.. autoclass:: d123.geometry.StateSE2Index()

.. autoclass:: d123.geometry.StateSE3Index()

.. autoclass:: d123.geometry.BoundingBoxSE2Index()

.. autoclass:: d123.geometry.BoundingBoxSE3Index()

.. autoclass:: d123.geometry.Corners2DIndex()

.. autoclass:: d123.geometry.Corners3DIndex()


Transformations
---------------

Transformations in 2D
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: d123.geometry.transform.convert_absolute_to_relative_se2_array

.. autofunction:: d123.geometry.transform.convert_relative_to_absolute_se2_array

.. autofunction:: d123.geometry.transform.translate_se2_along_body_frame

.. autofunction:: d123.geometry.transform.translate_se2_along_x

.. autofunction:: d123.geometry.transform.translate_se2_along_y


Transformations in 3D
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: d123.geometry.transform.convert_absolute_to_relative_se3_array

.. autofunction:: d123.geometry.transform.convert_relative_to_absolute_se3_array

.. autofunction:: d123.geometry.transform.translate_se3_along_body_frame

.. autofunction:: d123.geometry.transform.translate_se3_along_x

.. autofunction:: d123.geometry.transform.translate_se3_along_y

.. autofunction:: d123.geometry.transform.translate_se3_along_z

Occupancy Map
-------------
.. autoclass:: d123.geometry.OccupancyMap2D()
