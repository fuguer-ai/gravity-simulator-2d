# Observatory performance and visual update

## Guiding goals

More physics steps per second at fixed particle count, force law and timestep;
consistent rendering; a clear, beautiful view of the simulated structure. The
reported multiplier is `steps_per_second * dt / 2.7`, not GPU utilization.
At dt=0.5, the user's original 75x observation corresponds to 405 steps/second.
No measured RTX speedup is asserted from the development environment.

## Default: coordinated frame pacing

`CudaFrameController` owns persistent device arrays until reset, add/remove,
solver change or another explicit state invalidation. Snapshots update public
Body objects after completed batches; rendering never sees partially written
state. Normal frames do not re-upload positions/masses or invalidate acceleration.

`FixedStepClock.advance_batches` chooses short power-of-two batches based on
measured step cost and remaining budget. CUDA graph capture caches sequences of
kernel launches; one synchronization follows a batch, rather than each step.
The graph cache is bounded and invalidated when dt changes. Graph setup and
execution are exercised by the startup checker on the user's CUDA device.

Rendering completes before the next physics frame. Smoothed render cost reserves
space in the 60 FPS pacing target, with 1–12 ms allowed for physics. One very
expensive force evaluation may still exceed the budget; this is not a hard
real-time guarantee. Requested overload is dropped rather than accumulating
catch-up debt. Pause leaves state fixed. Timesteps never change automatically.

`--threaded` is optional and experimental, not the default. It gives a worker sole
ownership of its DeviceState and publishes immutable latest-only snapshots.
Structural edits stop/join and consume its last complete snapshot. There is no
unbounded queue. CUDA and OpenGL still share the same GPU; threading may make
frame timing worse. CPU FMM/Barnes-Hut remain on the main thread.

## Automatic population

`gpu_profile.py` queries actual CUDA device name, total/free VRAM. Warm 2048- and
8192-body (memory permitting) pilots fit `t(N)=a+b*N²`; a candidate targets 0.8 ms
per physics step. A measured candidate check reduces an overly optimistic count.
Extrapolation is capped at 4x the largest pilot, population at 50,000, and memory
at 35% of currently free VRAM after reserving 256 MiB for rendering/driver use,
using a conservative 128 B/body estimate. Solver buffers themselves use about
28 B/body plus tile padding.

This is a graphics-friendly starting population, not a claim of maximum capacity,
60 FPS or sustained simulation speed. Calibration excludes rendering and reflects
current system load. Other applications can subsequently change available VRAM.
`--galaxy-particles 5000` preserves a same-N baseline; explicit counts bypass auto.

## Rendering and meaning

- Instanced OpenGL quads replace per-particle Pygame calls; persistent VBO storage.
- HDR star splats, separable bloom and a softly smoothed stellar mass contribution
  provide cinematic light. The warm central light represents the existing fixed
  bulge. No extra gravity particles or artificial forces are introduced.
- The density view uses Gaussian mass splats normalized to total particle mass,
  a logarithmic color scale and faint isolines. It is a relative display at the
  current smoothing/view scale, not a calibrated surface-density measurement.
- Fading trails are an exponentially weighted screen-space exposure. They freeze
  while paused, clear on pan/zoom/reset and persist when following a body so that
  earlier relative positions remain visible. They are not interpolated physical
  orbit reconstructions; the software renderer retains its original point trails.
- V switches modes; comma/period changes exposure; F1 hides the HUD; Tab expands
  help. H/A retain the spatial hierarchy and on-demand force-error snapshot.
- OpenGL and window event handling stay on the main thread. Arrays still pass
  through host snapshots and a VBO upload; CUDA/OpenGL zero-copy interop is not
  implemented. The HUD remains Pygame-to-texture. These are further optimization
  opportunities, not claims of an entirely GPU-resident application.

## Reproducible checks

`python -m unittest discover -s tests -q` covers prior physics, persistent batches,
public-state edits, pause/solver switches, worker final snapshots/errors, profile
selection/memory guards, budgeted fixed timesteps, shader composition, paused
trail stability and framebuffer resize. Actual CUDA cases run only when a CUDA
device is available; CPU checks do not stand in for GPU execution.

Development validation:

- CUDA kernels compile to architecture-89 PTX using Warp's NVRTC, without an
  NVIDIA device. Graph execution and performance are not validated by compilation.
- Real Pygame/OpenGL window path exercised using SDL offscreen + llvmpipe:
  menu, simulation, mode/trail/pause/HUD switches and resize.
- EGL screenshots inspected for cinematic, density, menu and expanded help.
- Equal-duration timestep comparison and old/new frame benchmark exercised on
  small CPU cases. Those timings are not NVIDIA performance predictions.
- Native Windows launch and actual CUDA runtime/performance remain target-machine
  checks. The pinned launcher runs force/integration/graph checks before starting.

Preview tooling uses optional Pillow: `python benchmarks/render_preview.py
--output <folder>`. Run `benchmark_cuda.py` on the actual NVIDIA machine for
same-N same-dt comparisons; `compare_timesteps.py` measures the separate accuracy
tradeoff. The latter includes setup/capture time in its elapsed-time field.

## Scope boundaries

The force kernel and velocity-Verlet integrator remain the same softened
Newtonian model. No higher-order integrator, GPU FMM, live galaxy-merger model or
continuous potential/streamline view was added. Their scientific and performance
tradeoffs deserve separate validation. The original Keyframe link remains an
educational inspiration; its logarithmic-force animations were not copied into
this inverse-square gravity solver.
