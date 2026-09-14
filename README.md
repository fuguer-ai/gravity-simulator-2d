# 2D Gravity Simulator

A cross-platform Newtonian **N-body gravity simulator** written in Python with a Pygame interface.


## NVIDIA CUDA quick start (this fork)

After pulling this fork, run **`run_windows_cuda.bat`**. It creates/uses `.venv`,
installs the pinned CUDA-capable NVIDIA Warp wheel, checks actual GPU forces and
integration, then opens the simulator. Choose **4 / Spiral Galaxy** for **5,000
stars**. Your current NVIDIA driver is used; no separate CUDA Toolkit, compiler,
WSL, or PyTorch installation is required. First launch compiles kernels and takes
longer. CPU solvers remain available through S/B.

```powershell
.\run_windows_cuda.bat
# Optional larger galaxy:
.\run_windows_cuda.bat --galaxy-particles 10000
```

This adds **CUDA exact**, not CUDA FMM. The NVIDIA tiled implementation evaluates
all pairs without allocating an N-by-N matrix. Its FP32 velocity-Verlet state
stays on the GPU between physics steps; the existing Pygame renderer receives
one snapshot per frame. The original CPU FMM and ordinary launcher retain their
default behavior. Actual GPU speed depends on hardware and is not yet measured
in this repository; the startup accuracy check runs on your GPU.

New controls: **U** enables CUDA exact; **H** shows the spatial hierarchy;
**A** pauses and displays sampled force-error arrows. FPS is shown in the HUD.
The hierarchy view is diagnostic and can reduce rendering performance.

See [the source comparison and validation details](docs/gpu-solver-review.md).
To benchmark 1,200 / 5,000 / 10,000 particles after installation:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_cuda.py --compare-cpu --output benchmarks\results\cuda-local.json
```

Optional PyTorch users can call `torch_gravity.accelerations(positions, masses)`
with CPU or CUDA tensors for a blocked exact reference, or use
`DeviceState.torch_views()` for zero-copy access to the Warp state. PyTorch is
not a dependency of the fast interactive path. Do not install a CPU Torch build
expecting it to execute CUDA tensor operations; install your desired supported
CUDA build from [PyTorch](https://pytorch.org/get-started/locally/) separately if
using that optional API. Framework stream synchronization is documented in the
method docstring.

## Features

- Newtonian gravity in 2D
- **Symmetric Cartesian FMM** selected by default in the app
- Switchable Barnes-Hut, exact, and automatic solvers with `S`
- Exact pairwise gravity for small systems (O(n^2))
- Velocity-Verlet integration
- Startup scenario picker
- Solar System preset with realistic relative planetary masses and orbital-distance ratios
- Binary-star circumbinary preset
- Randomized 120-body system
- Near-equilibrium exponential galaxy disk with a smooth bulge/halo and mild evolving arms
- Custom body colors, sizes, and masses
- Selectable moving reference frames centered on any body
- Requested simulation speed from 1/16x to **4096x**, with fixed physics timesteps
- Pause, reset, zoom, pan, and orbital trails
- Windows, macOS, and Linux support with Python 3.10+

## Windows quick start

Clone the repository, then enter it:

```powershell
gh repo clone MODLICENSE/gravity-simulator-2d
cd gravity-simulator-2d
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

You can also launch with:

```powershell
.\run_windows.bat
```

## Startup scenarios

When the program starts, choose one of four systems:

1. **Solar System** — Sun plus all eight planets. Relative masses and semi-major-axis ratios are based on the real Solar System, while visual radii are enlarged so the planets remain visible. Orbits are circularized and the model is 2D.
2. **Binary Star System** — two stars orbit a common barycenter with three circumbinary planets.
3. **Randomized System** — a new 120-body rotating system every time. Uses the selected solver (FMM initially).
4. **Spiral Galaxy** — a seeded 1,200-star exponential disk in a smooth bulge and halo, with mild spiral overdensities. FMM is selected initially; `S` or `B` changes the solver.

Press `M` while simulating to return to the scenario menu.

### Spiral galaxy notes

The galaxy now has a density profile and initial velocities designed for
approximate dynamical equilibrium. Its smooth bulge/halo attracts every star
through the same field in Exact, Barnes-Hut and FMM modes. Arms evolve naturally;
they are not fixed tracks. Resets use the same seed for fair solver comparisons.
Trails start off and have shorter history in this preset.

The full preset remained confined over two reference orbits with both FMM and
Barnes-Hut. Read [the theory, parameters, limitations, and measured results](docs/galaxy-stability.md).
This is still an educational 2D model with a fixed background, not a calibrated
or fully self-consistent live Milky Way model.

### Galaxy appearance

Violet, magenta, blue, and warm stars use cached additive glows. The luminous
center is a visual representation of the **existing smooth bulge**, including
an unresolved-star texture. Those points are not extra force particles. The
dark-matter halo remains invisible. Press **G** to toggle the glow layer.

The purple palette is sci-fi-inspired. The density model is not calibrated to
the Milky Way, whose central bulge is barred rather than our spherical Plummer
model ([ESA overview](https://www.esa.int/ESA_Multimedia/Images/2018/05/Anatomy_of_the_Milky_Way)).
The visual changes preserve every particle's initial position, velocity, and mass.

## Controls

| Control | Action |
|---|---|
| Left click | Add a body using the currently selected color / size / mass |
| Right click | Remove the nearest body |
| Middle mouse drag | Pan camera while in the world frame |
| Mouse wheel | Zoom |
| Space | Pause / resume |
| `+` / `-` | Double / halve simulation speed (1/16x to 4096x) |
| `C` | Cycle the color of newly created bodies |
| `[` / `]` | Make newly created bodies smaller / larger |
| `F`, then left click a body | Lock the moving reference frame to that body |
| `F` while locked | Return to the normal world frame |
| `S` or `B` | Cycle FMM → Barnes-Hut → Exact → Auto |
| `Page Up` / `Page Down` | Double / halve physics timestep (1/4x–16x the preset default) |
| `0` | Restore default timestep |
| `G` | Toggle galaxy bulge/glow rendering |
| `T` | Toggle trails |
| `R` | Reset the current scenario |
| `M` | Return to scenario selection |
| `Esc` or `Q` | Quit |

The HUD displays the current new-body radius, mass, color, reference frame, and timestep. Steps above 2x the preset default are marked COARSE.

## Moving reference frames

Press `F` and then click a body. The selected body becomes the stationary center of the display.

This is implemented as a coordinate transformation rather than by changing the physical state of the simulation. If body `r` is selected as the reference,

```text
x'_i = x_i - x_r
v'_i = v_i - v_r
```

so the selected body has zero displayed position and velocity while every other body's relative velocity is preserved. The underlying Newtonian integration continues in the original coordinates.

Trails are transformed using the reference body's historical positions as well, so they show motion in the selected moving frame rather than simply following the camera.

Press `F` again to release the reference frame.

## High-speed simulation

Speed and timestep are decoupled. The requested multiplier (up to 4096x) controls
how many steps run. Defaults are dt=0.5 for the galaxy and 0.045 for other presets.
Use **Page Up / Page Down** to adjust timestep independently, or **0** to reset it.
The bounded range is 1/4x–16x the default (galaxy: 0.125 to 8). Timestep stays
fixed between explicit changes. Larger steps advance more simulated time per
force calculation but resolve orbits less accurately. A change clears pending
time and the speed estimate to avoid catch-up bursts; scenario resets restore
the default.
A bounded CPU budget takes additional steps when the machine has time. When it
cannot keep up, the HUD displays **achieved/requested** speed instead of taking
automatically enlarging steps. This also prevents a growing catch-up backlog.

## Update to the integrated main branch

From your existing repository folder in PowerShell:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
.\run_windows.bat
```

FMM is active at startup. Press `S` to compare solvers; the HUD shows the actual
solver. Your selection survives scenario changes and resets. The solver itself
uses only Python's standard library: no WSL, Fortran, CMake, CUDA, or compiler is
needed. Pygame runs the UI. NumPy is retained for the experimental interpolation reference.

## Fast multipole method

`fmm.py` implements an original low-order symmetric Cartesian FMM using the
cell-to-cell construction in [Dehnen (2014), Appendix A.1](https://arxiv.org/abs/1405.2255).
It builds source moments (P2M/M2M), computes mutual multipole-to-local (M2L)
interactions with a dual tree walk, translates local expansions downward (L2L),
and evaluates them at particles (L2P). Nearby pairs are evaluated directly.

The potential expansion has total degree 3: central source quadrupoles plus
quadratic local acceleration expansions. Both directions of a cell pair use the
same truncation, preserving total force to floating-point roundoff. The
softened kernel is `1/sqrt(dx*dx + dy*dy + softening**2)`, so this retains the
existing inverse-square Newtonian force in a plane. It does not substitute the
logarithmic potential used by mathematical 2D Laplace FMM libraries.

```python
sim = NBodySimulation(bodies, solver="fmm", fmm_theta=0.5,
                      fmm_leaf_capacity=16)
```

- `fmm_theta`: sum of cell radii divided by separation must be below this value.
  Smaller values are more accurate but slower; `0` forces direct interactions.
  Allowed range is `0 <= theta < 1`. This is an opening criterion, not a promised
  relative error tolerance. Expansion order is fixed.
- `fmm_leaf_capacity`: maximum bodies per ordinary leaf (default 16). A depth cap
  prevents endless subdivision of coincident bodies.
- Finite positions and nonnegative finite masses are required; zero-mass test
  particles work. Coincident bodies require positive softening.
- Library callers retain the previous default `solver="auto"` (exact below 64,
  Barnes-Hut otherwise). The app explicitly selects FMM for this branch.

This pure-Python implementation prioritizes portability and verifiable forces.
FMM's cell interactions can approach linear scaling for well-behaved trees at
fixed accuracy; this tree builder also visits particles on each level, and
pathological clustering can force quadratic direct work. FMM is not guaranteed
to beat Barnes-Hut for these small presets. The fixed timestep controls integration accuracy independently of the force approximation.

### Reproduce the benchmark

```powershell
py benchmarks/benchmark_fmm.py --sizes 200 1000 3000 --repeats 3
```

Example Linux-container run, seed 42, uniform square, softening 1, median of
three force evaluations (no rendering or integration):

| Bodies | Exact ms | Barnes-Hut ms | FMM ms | FMM relative RMS force error |
|---|---:|---:|---:|---:|
| 200 | 6.58 | 3.97 | 5.30 | 0.071% |
| 1,000 | 164.30 | 32.60 | 36.82 | 0.320% |
| 3,000 | 1553.64 | 130.54 | 147.48 | 0.535% |

At 3,000 bodies FMM was 10.5 times faster than exact, slightly slower than
Barnes-Hut, and had lower aggregate error (Barnes-Hut: 2.20%). Parameters were
FMM theta 0.5 and Barnes-Hut theta 0.7, so this is not an equal-accuracy benchmark.
These are not Windows timings or FPS promises. Relative RMS is the norm of the
force-vector error divided by the norm of the exact acceleration vector; it is
not a bound on every particle's relative error.

## Barnes-Hut gravity

In `auto` mode, small systems evaluate every body-body force exactly. For systems with 64 or more bodies by default, it automatically switches to the **Barnes-Hut algorithm**.

Barnes-Hut builds a quadtree and approximates sufficiently distant collections of bodies by their combined mass at their center of mass. This reduces the usual O(n^2) force calculation toward roughly O(n log n).

The main accuracy/speed control is `barnes_hut_theta` in `NBodySimulation`:

- Smaller theta such as `0.4` = more accurate, slower
- Default `0.7` = balanced
- Larger theta = faster, less accurate

`barnes_hut_threshold` controls when the simulator switches from exact gravity to Barnes-Hut; the default is 64 bodies.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
.
├── app.py                    # Entry point
├── gravity_app.py            # UI and controls
├── galaxy.py                 # Galaxy density and velocity initialization
├── simulation_clock.py       # Fixed-step pacing and CPU budget
├── gravity_sim.py            # Physics engine + solver dispatch
├── fmm.py                    # Symmetric Cartesian FMM
├── benchmarks/benchmark_fmm.py
├── tests/
│   ├── test_fmm.py           # Accuracy, edge cases, conservation
│   └── test_gravity_sim.py   # Existing physics tests
├── requirements.txt
├── run_windows.bat
├── LICENSE
└── README.md
```

## Alternative FMM implementation

The other main-branch implementation is preserved in
`experimental/interpolation_fmm.py` for accuracy/performance comparisons. The
production `fmm` selection uses the symmetric Cartesian solver. `solver_mode`
and `cycle_solver()` remain available for existing callers. The old `fmm_order`
and `fmm_max_level` controls belong to the experimental interpolation engine;
the production solver uses fixed expansion order and `fmm_theta`.

## Notes

This is an educational simulator rather than a high-precision astrophysics package. The Solar System preset uses realistic relative masses and orbital-distance ratios, but it is intentionally simplified to 2D circular orbits and uses enlarged display radii.