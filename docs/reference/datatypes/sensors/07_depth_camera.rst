Depth Camera
^^^^^^^^^^^^

A depth camera is a per-pixel metric depth stream, pixel-aligned to a sibling RGB
:class:`~py123d.datatypes.Camera` whose ``camera_id``, projection model, intrinsics, and extrinsics it
shares. It is stored as :attr:`~py123d.datatypes.ModalityType.CAMERA_DEPTH` in its own stream,
``camera_depth.<camera_id>.arrow``, separate from the RGB ``camera.<camera_id>.arrow`` of the same
``camera_id``.

The storage contract — how metric depth is quantized into the stored raster, and the ``max_depth``,
``depth_bits``, ``depth_transform``, ``min_depth``, ``has_invalid`` and ``depth_type`` knobs that
control it — is documented on :class:`~py123d.datatypes.DepthCameraMetadata` below. The examples here
show it end to end.

Reading and writing
-------------------

.. code-block:: python

   from py123d.datatypes import Camera, DepthCameraMetadata

   # Writing: quantize metric depth into the stored raster.
   depth_metadata = DepthCameraMetadata(camera_metadata=rgb_metadata, max_depth=96.0, depth_bits=16)
   log_writer.write_async(
       Camera(
           metadata=depth_metadata,
           image=depth_metadata.encode_depth(depth_in_metres),
           camera_to_global_se3=camera_to_global_se3,
           timestamp=timestamp,
       )
   )

   # Reading: `image` is the raw quantized raster, not metres. `scale` optionally downsamples.
   camera = scene.get_camera_depth_at_iteration(iteration, camera_id=CameraID.PCAM_F0, scale=2)
   depth_in_metres = camera.metadata.decode_depth(camera.image)  # NaN where has_invalid marked no data

   # `rgb_image` colorizes the raster with a TURBO ramp for display.
   preview = camera.rgb_image

The default above (linear, no sentinel, z-depth) matches a simulator that renders a finite depth for
every pixel. A sparse real sensor — e.g. lidar projected into the image — wants the inverse transform
for near-field precision and the invalid sentinel for dropouts:

.. code-block:: python

   depth_metadata = DepthCameraMetadata(
       camera_metadata=rgb_metadata,
       max_depth=120.0,
       depth_bits=16,
       depth_transform="inverse",   # fine near, coarse far
       min_depth=0.5,               # required by the inverse transform
       has_invalid=True,            # code 0 -> NaN for pixels with no measurement
       depth_type="z_depth",        # or "ray_distance" for a native range sensor
   )

The ``scale`` argument on the read accessors (``get_camera_depth_at_iteration`` /
``get_camera_depth_at_timestamp``) is an integer downscale denominator — ``2`` for half size, ``4`` for
quarter — applied at decode time. It resamples with nearest-neighbour, never bilinear: averaging depth
across an occlusion boundary would invent a surface floating between foreground and background.

Depth Camera Metadata
---------------------

.. autoclass:: py123d.datatypes.DepthCameraMetadata
   :members:
   :exclude-members: __init__
   :autoclasstoc:


Depth Colorization
------------------

.. autofunction:: py123d.datatypes.colorize_depth_map
