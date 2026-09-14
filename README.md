# 2D Gravity Simulator

A cross-platform Newtonian **N-body gravity simulator** written in Python with a Pygame interface.


## Observatory: NVIDIA CUDA + OpenGL

Run **`run_windows_cuda.bat`** after pulling this fork. It creates/uses `.venv`
and installs pinned NVIDIA Warp and ModernGL wheels. No separate CUDA Toolkit,
compiler, WSL, or PyTorch installation is required; use your current NVIDIA driver.

```powershell
git switch main
git pull --ff-only origin main
.\run_windows_cuda.bat
```

Startup checks CUDA forces, Verlet integration, graph replay and timestep
changes, then **detects your GPU/VRAM and calibrates a starting particle count**.
Choose **4 / Spiral Galaxy**. Auto selection targets a measured 0.8 ms physics
step, with VRAM headroom and a 50,000-particle cap. It is a starting point for
visual density and useful speed, not a maximum supported count or an FPS promise.
Exact gravity uses O(N) storage but O(N²) work, so free VRAM alone cannot choose N.

The default now uses **one coordinated physics-then-render loop**. CUDA state
persists across frames, short CUDA graph batches replace per-step Python waits,
and OpenGL draws particle instances, bloom, density and fading trails. The frame
budget reserves measured render time. An optional worker exists for experiments;
it is **off by default** because concurrent use of one GPU can worsen pacing.

To compare directly with your previous 5,000-star run, override auto selection:

```powershell
.\run_windows_cuda.bat --galaxy-particles 5000
```

| Control | Effect |
|---|---|
| `V` | Cinematic stars / logarithmic relative mass-density map |
| `T` | Smooth fading trails (screen-space exposure in OpenGL) |
| `G` | Glow and bloom |
| `,` / `.` | Lower / raise exposure |
| `F1` | Hide/show HUD for a clean view |
| `Tab` | Show/hide full controls |
| `H` | Spatial quadtree overlay |
| `A` | Pause and compare sampled forces with direct summation |
| `U` | Enable CUDA exact; `S` / `B` cycles enabled solvers |
| `Page Up` / `Page Down` | Larger / smaller timestep; `0` restores default |

The HUD shows achieved/requested simulated speed, steps/second, FPS and batch
cost. **Keep N and dt fixed when comparing performance.** A larger timestep
increases simulated-time speed while changing integration accuracy.

Diagnostics (optional, after the launcher installs dependencies):

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_cuda.py --sizes 5000 --output benchmarks\results\cuda-local.json
.venv\Scripts\python.exe benchmarks\compare_timesteps.py --output benchmarks\results\timestep-local.json
```

The first compares the old upload/per-step-barrier frame pattern with persistent
batches at the same N, dt and eight steps/frame; rendering is excluded. The
second compares equal simulated durations against a smaller timestep, including
energy and radial statistics. Neither silently changes your timestep.

Advanced fallbacks: `--no-graphs`, `--renderer software`, or experimental
`--threaded`. The normal `run_windows.bat` retains CPU FMM and software fallback.
There is no CUDA FMM in this release. CPU FMM/Barnes-Hut remain available.

Optional PyTorch users can call `torch_gravity.accelerations(positions, masses)`
with CPU/CUDA tensors or use `DeviceState.torch_views()` for zero-copy array
access; observe its documented synchronization rules. Torch is not required by
the fast application path.

See [implementation and validation](docs/observatory-implementation.md) and
[the source comparison](docs/gpu-solver-review.md).

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

The compact HUD shows the solver, timestep, speed and reference frame. Tab opens detailed controls and new-body settings. Increasing the timestep trades temporal accuracy for simulated-time speed.

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
A bounded frame budget takes additional steps when the machine has time. When it
cannot keep up, the HUD displays **achieved/requested** speed instead of automatically enlarging steps. This also prevents a growing catch-up backlog.

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