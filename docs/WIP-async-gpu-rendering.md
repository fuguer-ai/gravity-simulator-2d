# WIP: asynchronous CUDA physics and GPU rendering

Stopped at the user's request to preserve remaining credits. This branch is
unfinished and MUST be validated before merging. Main remains the previously
working CUDA integration.

## Implemented so far

- `cuda_gravity.py`: `step_batch(dt,count)` with bounded 0..32-step CUDA graphs,
  per-timestep graph invalidation, CPU direct-loop fallback for validation,
  detached snapshots; fixed block size on integration/background kernels.
  Startup verification extended to graph capture/replay and dt changes.
- `physics_worker.py`: dedicated CUDA-owner thread, coalesced controls, immutable
  latest-only snapshots, measured adaptive power-of-two batches targeting ~4 ms,
  60 Hz snapshot publishing, final snapshot on stop. No per-frame state upload.
- `gpu_render.py`: new ModernGL 3.3 instanced particles, HDR targets, bloom,
  exponential screen-space trails, relative mass-density mode, exposure,
  composited Pygame HUD. OpenGL stays on the main thread.
- `gravity_app.py`: worker lifecycle/snapshots, stop-and-consume before edits and
  solver changes, OpenGL/software selection, compact HUD, Tab help, V view,
  comma/period exposure. Existing CPU physics path retained.
- `requirements-cuda.txt`: pinned ModernGL 5.12.0 and glcontext 3.0.0 alongside
  Warp 1.17.0. Existing launcher installs these requirements.
- CLI additions: `--renderer auto|opengl|software`, `--no-threading`, `--no-graphs`.

## Validation actually completed

- `python -m py_compile gravity_app.py cuda_gravity.py gpu_render.py physics_worker.py`
  passed. This establishes syntax only.
- ModernGL installed locally; standalone EGL context initialized successfully
  using llvmpipe software OpenGL. NEW SHADERS HAVE NOT BEEN COMPILED OR RENDERED.
- No NVIDIA driver/device in this environment. NEW CUDA graph behavior has NOT
  been executed here. Previous main's tests do not validate this WIP batch.

## Remaining work, in priority order

1. Run the existing suite and resolve regressions. Add numerical tests comparing
   batches with repeated Verlet steps; check graph reuse, dt changes, 0/partial
   tiles, background force, no repeated uploads. Re-run sm_89 AOT compilation.
   Actual graph execution still needs an NVIDIA machine or CUDA-equipped CI.
2. Test worker lifecycle with a deterministic CPU/fake state factory: controls,
   pause/resume, speed/dt changes, final snapshot correctness, immutable snapshots,
   bounded mailbox, errors, shutdown, reset/add/remove/solver switches, empty
   systems. Check race cases and that edits never overwrite a newer snapshot.
3. Compile/render every shader via standalone EGL, then inspect screenshots for
   cinematic, density, trails, paused frames, resize, menu/HUD and moving frames.
   Exercise the real Pygame/OpenGL window path (Xvfb if available) and Windows.
4. Review graph capture/worker stream ownership carefully. Main thread performs
   CUDA startup checks and paused diagnostics; ensure it never overlaps a worker
   capture. Confirm capture warmup costs are excluded from meaningful timings.
5. Review app lifecycle/error paths. `stop_worker` joins with a timeout: a hung
   worker must not permit state mutation. Worker init errors must remain visible
   and not cause repeated restart attempts. Current CPU backends remain on the
   main thread (only CUDA is threaded).
6. Review renderer behavior/performance: it still transfers snapshots through
   host RAM (no CUDA/GL zero-copy interop), rebuilds a tuple of Body IDs each draw,
   and uploads a full-window HUD texture. This is GPU drawing, not a fully
   GPU-resident rendering pipeline. Density is relative Gaussian mass splatting,
   not a calibrated physical density. Screen-space trails reset on camera changes;
   following a moving reference may therefore reset history every frame. Resolve
   or document this and preserve usable reference-frame behavior.
7. Check resize handling under Pygame 2 without recreating/loss of GL context;
   ensure framebuffer/texture resources are released and recreated correctly.
   Compact HUD has fixed placements that need narrow-window clipping/layout QA.
8. Update benchmark_cuda.py to measure graph/non-graph batches, persistent worker
   throughput and rendering separately. No speedup/FPS claims are justified yet.
9. Add timestep/accuracy comparison requested in previous turn: fixed-duration
   dt/half-dt runs, energy drift and radial statistics. No new higher-order
   integrator, GPU FMM, potential-field/streamline view, or new dynamic scenarios
   has been implemented in this WIP.
10. Update README, third-party/change notes, controls and launcher documentation;
    run final tests and visual QA before publishing a non-draft PR and merging.

## Resume commands

```powershell
git fetch origin
git switch feat/async-gpu-rendering
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use the working `main` branch for normal use until this branch is validated.
