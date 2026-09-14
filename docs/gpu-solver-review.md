# GPU solver review and integration

## Selection

For this fork's 1,200–10,000 particle target, integrate NVIDIA Warp's tiled
all-pairs kernel first. This is a fit assessment, **not a measured claim that it
is the world's fastest solver**. No NVIDIA device was available in the build
environment. The CUDA module compiled to sm_89 PTX (4080/4090 architecture), and
CPU execution checked the shared interaction/integration code. CUDA execution
must pass a real force/integration check on startup on the user's machine.

| Source reviewed | Useful contribution | Integration decision |
|---|---|---|
| [NVIDIA/warp](https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_nbody.py) | Shared-memory tiles, Python CUDA deployment, framework interoperability | Adapted with attribution; pinned `warp-lang==1.17.0` |
| [keyframe41/Videos](https://github.com/keyframe41/Videos/tree/main/Fast-Multipole-Method) | Hierarchy, multipole translations, force-error explanations | Original tree overlay and sampled-error arrows; its files are Manim scenes, not a GPU simulator |
| [kurtamohler/nbody-tensors](https://github.com/kurtamohler/nbody-tensors) | Tensor force formulation, cached Verlet acceleration | Reviewed; its dense N-by-N intermediates are unsuitable for scaling; no code copied |
| [Barjak/Torch-NBody](https://github.com/Barjak/Torch-NBody) | PyTorch experiment with fourth-order Hermite integration | README reviewed; would change integrator, no measured advantage established |
| [andyljones/pybbfmm](https://github.com/andyljones/pybbfmm) | Actual PyTorch GPU black-box FMM, arbitrary kernels | Source reviewed; not integrated: per-interaction Python loops, old dependencies, and tree has no depth termination guard for over-capacity coincident points |
| [lechebs/nbody](https://github.com/lechebs/nbody) | Fully GPU Barnes-Hut, large-N demonstration and diagnostics | README reviewed; candidate for later scale-up; standalone CUDA C++/OpenGL/Linux app, not a drop-in Python/Windows backend |

Keyframe revision: `8290c3159bb8f320b680477adf55535ed3fefb81`.
pybbfmm revision: `967274940b7068342db9601259fbd674a9a75eac`.
nbody-tensors revision: `b919befeda8771b122cbfaeae37ba10846e060ee`.
NVIDIA source identity is recorded in THIRD_PARTY_NOTICES.md.

## Preserve the physics

The simulator uses a 3D Newtonian kernel with particles constrained to a plane:
`a_i = G sum_j m_j (x_j-x_i)/(||x_j-x_i||^2 + eps^2)^(3/2)`.
Keyframe's complex expansion uses a logarithmic potential. Copying it would
change the force law. GPU FMM remains possible, but requires the correct softened
kernel, moving-particle tree rebuilds, near-field treatment and verified errors.
It is not implemented in this batch; the original CPU FMM remains available.

NVIDIA's example assumes equal masses and full tiles and uses a different
integration update. This adaptation supports unequal/zero masses and partial
tiles, retains the smooth galaxy field, and uses kick-drift-kick Verlet.
Positions, velocities, masses and acceleration occupy O(N) device memory;
force work remains O(N²). After the first force evaluation, resident Verlet steps
reuse acceleration and require one new force evaluation each.

The UI uploads Body data once per physics frame, holds state on the GPU across
steps, and downloads once for rendering. Synchronization per step charges actual
GPU work to the frame budget. Public standalone `sim.step()` still updates Body
objects immediately. Add/remove/reset, solver switches and background changes
are handled at frame boundaries. This is not a fully GPU-rendered app: Pygame,
trails, Python Body copies and diagnostic tree drawing still have CPU costs.

FP32 is deliberate for interactive GeForce performance. It does not promise
roundoff-level momentum cancellation from symmetric pair updates or scientific
long-term orbital accuracy. The CPU FP64 solvers remain useful references.
Positive softening and finite representable coordinates are required for CUDA.

## Diagnostics and benchmarking

- `H`: show a spatial quadtree (depth capped for display). This is a diagnostic
  partition, not a representation of the exact CUDA solver's execution.
- `A`: pause and compare 64 deterministic targets to direct all-source gravity.
  Arrows show approximate force direction with normalized display length; color
  encodes per-target relative error. Reported RMS is ||error||₂/||reference||₂.
  This is a snapshot, not an error bound for all particles.
- The existing S/B solver cycle includes CUDA after enabling it with U or CLI.
- `benchmarks/benchmark_cuda.py` warms up kernels, synchronizes timings, measures
  resident integration separately from transfers, and checks force accuracy.
  `--compare-cpu` reports original FMM/Barnes-Hut force costs and their errors.
  These use their existing theta settings, not matched accuracy tolerances.

Run on the NVIDIA machine:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_cuda.py --compare-cpu --output benchmarks\results\cuda-local.json
```

No GPU FPS or speedup numbers are asserted from CPU tests. A lower FMM
asymptotic exponent alone does not establish a win at a few thousand bodies.

## Validation recorded for this integration

- `python -m unittest discover -s tests -q`: 30 tests, 29 passed, one explicitly
  skipped because no NVIDIA driver/device was present. Includes original physics
  tests, partial tile sizes (0/1/2/63/64/65/129), massless/coincident softened
  bodies, unequal masses, background forces, multi-step Verlet and momentum,
  host edits, solver transitions, and PyTorch memory-sharing/reference checks.
- Warp 1.17.0 ahead-of-time CUDA compilation succeeded for architecture 89.
  Compilation does **not** establish correct GPU execution or performance.
- Headless Pygame smoke: render 5,000-star preset, hierarchy overlay, sampled
  force errors, pause and solver switch; screenshots inspected for layout.
- Benchmark CLI exercised on CPU with 65 stars; force RMS error about 1.18e-7.
  CPU timing is not offered as a prediction of NVIDIA performance.
- Windows batch execution and real GPU runtime checks remain machine-dependent;
  the CUDA launcher performs force and integration checks on the user's GPU.
