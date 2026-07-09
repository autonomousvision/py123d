Depth Camera
^^^^^^^^^^^^

A depth camera is a per-pixel metric depth stream, pixel-aligned to a sibling RGB
:class:`~py123d.datatypes.Camera` whose ``camera_id``, projection model, intrinsics and extrinsics it
shares. It is stored as :attr:`~py123d.datatypes.ModalityType.CAMERA_DEPTH` in its own Arrow file,
``camera_depth.<camera_id>.arrow``, alongside — and never colliding with — ``camera.<camera_id>.arrow``.

Storage contract
----------------

Unlike a segmentation class-id map, depth is *continuous*, so it must be quantized before it can be
stored as a lossless integer PNG. The encoding is a clip followed by a linear rescale onto the full
integer range:

.. code-block:: text

   raw     = round(clip(depth_m, 0, max_depth) / max_depth * max_raw)   # encode
   depth_m = raw / max_raw * max_depth                                   # decode

where ``max_raw = 2 ** depth_bits - 1``. Two consequences are worth internalizing before you write a
depth stream:

* **The far plane is a hard clip, not a sentinel.** Anything beyond ``max_depth`` saturates to
  ``max_raw`` and decodes back as exactly ``max_depth``. A simulated sky pixel at 1000 m and a wall at
  ``max_depth`` are indistinguishable once encoded. Choose ``max_depth`` accordingly.
* **``0`` means zero metres, not "no measurement".** There is no invalid sentinel — the full integer
  range encodes depth. This suits simulators, which render a finite depth for every pixel; a real-world
  sensor with dropouts must store its invalid mask separately.

``depth_bits`` trades resolution against file size; ``max_depth`` trades range against resolution. The
worst-case round-trip error is half a quantization step:

==============  =============  ==================  ======================
``depth_bits``  ``max_depth``  resolution          max round-trip error
==============  =============  ==================  ======================
8               50 m           196 mm              98 mm
16              96 m           1.46 mm             0.73 mm
16              1024 m         15.6 mm             7.8 mm
==============  =============  ==================  ======================

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

   # Reading: `image` is the raw quantized raster, not metres.
   camera = scene.get_camera_depth_at_iteration(iteration, camera_id=CameraID.PCAM_F0)
   depth_in_metres = camera.metadata.decode_depth(camera.image)

   # `rgb_image` colorizes the raster with a TURBO ramp for display.
   preview = camera.rgb_image

Downscaling (the ``scale`` argument) resamples with nearest-neighbour, never bilinear: averaging depth
across an occlusion boundary would invent a surface floating between the foreground and the background.

Depth Camera Metadata
---------------------

.. autoclass:: py123d.datatypes.DepthCameraMetadata
   :members:
   :exclude-members: __init__
   :autoclasstoc:


Depth Colorization
------------------

.. autofunction:: py123d.datatypes.colorize_depth_map
