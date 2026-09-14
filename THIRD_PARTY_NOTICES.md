# Third-party notices

`cuda_gravity.py` adapts the tiled all-pairs construction from NVIDIA Warp:
https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_nbody.py
Reviewed source blob: `f5b8e41669766c6851a0fdb7d7a9d807526ad8ac`.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; complete text: `licenses/NVIDIA-Warp-Apache-2.0.txt`.
Modifications: 2D coordinates, arbitrary body counts, per-body masses, self
exclusion, configurable softening/G, external Plummer field, velocity-Verlet,
persistent arrays, PyTorch views, and validation. NVIDIA does not endorse this fork.

The remainder of the original simulator retains its MIT license. Warp is an
optional separately installed dependency with its own bundled third-party terms.

Keyframe41/Videos was reviewed as an educational reference. No animation code or
assets were copied. The hierarchy and sampled error visualizations are original
additions inspired by the concepts illustrated in that video. Other reviewed
projects are recorded in `docs/gpu-solver-review.md`; none were vendored.

The Observatory renderer (`gpu_render.py`), frame controller, worker, automatic
population calibration, and UI are original additions. ModernGL 5.12.0 and
glcontext 3.0.0 are separately installed optional dependencies. No external
textures, fonts, animation code or image assets are bundled by this update.
