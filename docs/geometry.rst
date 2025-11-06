
Geometry
========

Geometric Primitives
--------------------

Points
~~~~~~
.. autoclass:: py123d.geometry.Point2D()

.. autoclass:: py123d.geometry.Point3D()

Vectors
~~~~~~~
.. autoclass:: py123d.geometry.Vector2D()

.. autoclass:: py123d.geometry.Vector3D()

Special Euclidean Group
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: py123d.geometry.StateSE2()

.. autoclass:: py123d.geometry.StateSE3()

Bounding Boxes
~~~~~~~~~~~~~~
.. autoclass:: py123d.geometry.BoundingBoxSE2()

.. autoclass:: py123d.geometry.BoundingBoxSE3()

Indexing Enums
~~~~~~~~~~~~~~
.. autoclass:: py123d.geometry.Point2DIndex()

.. autoclass:: py123d.geometry.Point3DIndex()

.. autoclass:: py123d.geometry.Vector2DIndex()

.. autoclass:: py123d.geometry.Vector3DIndex()

.. autoclass:: py123d.geometry.StateSE2Index()

.. autoclass:: py123d.geometry.StateSE3Index()

.. autoclass:: py123d.geometry.BoundingBoxSE2Index()

.. autoclass:: py123d.geometry.BoundingBoxSE3Index()

.. autoclass:: py123d.geometry.Corners2DIndex()

.. autoclass:: py123d.geometry.Corners3DIndex()


Transformations
---------------

Transformations in 2D
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: py123d.geometry.transform.convert_absolute_to_relative_se2_array

.. autofunction:: py123d.geometry.transform.convert_relative_to_absolute_se2_array

.. autofunction:: py123d.geometry.transform.translate_se2_along_body_frame

.. autofunction:: py123d.geometry.transform.translate_se2_along_x

.. autofunction:: py123d.geometry.transform.translate_se2_along_y


Transformations in 3D
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: py123d.geometry.transform.convert_absolute_to_relative_se3_array

.. autofunction:: py123d.geometry.transform.convert_relative_to_absolute_se3_array

.. autofunction:: py123d.geometry.transform.translate_se3_along_body_frame

.. autofunction:: py123d.geometry.transform.translate_se3_along_x

.. autofunction:: py123d.geometry.transform.translate_se3_along_y

.. autofunction:: py123d.geometry.transform.translate_se3_along_z

Occupancy Map
-------------
.. autoclass:: py123d.geometry.OccupancyMap2D()
