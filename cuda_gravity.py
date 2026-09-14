# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NVIDIA Warp's example_tile_nbody.py (see THIRD_PARTY_NOTICES.md).

Changes: 2D, unequal masses, arbitrary N, explicit self exclusion, Plummer
background, velocity-Verlet, persistent device buffers, optional torch interop.
The tiled kernel is CUDA-only; the scalar kernel is a CPU validation reference.
"""
import math
import numpy as np
import warp as wp

TILE = wp.constant(64)


@wp.func
def interaction(p: wp.vec2, q: wp.vec2, mass: float, eps2: float):
    r = q - p
    r2 = wp.dot(r, r) + eps2
    inv = (1.0 / wp.sqrt(r2))
    return mass * inv * inv * inv * r


@wp.kernel
def tiled_force(pos: wp.array(dtype=wp.vec2), mass: wp.array(dtype=float),
                out: wp.array(dtype=wp.vec2), n: int, eps2: float, G: float):
    i = wp.tid()
    p = pos[i]
    a = wp.vec2(0.0)
    # Host pads both arrays to a full tile: no barrier-divergent early returns.
    for tile in range((n + TILE - 1) // TILE):
        ps = wp.tile_load(pos, shape=TILE, offset=tile * TILE)
        ms = wp.tile_load(mass, shape=TILE, offset=tile * TILE)
        for k in range(TILE):
            j = tile * TILE + k
            if i < n and j < n and j != i:
                a += interaction(p, ps[k], ms[k], eps2)
    out[i] = G * a


@wp.kernel
def scalar_force(pos: wp.array(dtype=wp.vec2), mass: wp.array(dtype=float),
                 out: wp.array(dtype=wp.vec2), n: int, eps2: float, G: float):
    i = wp.tid()
    a = wp.vec2(0.0)
    for j in range(n):
        if i != j:
            a += interaction(pos[i], pos[j], mass[j], eps2)
    out[i] = G * a


@wp.kernel
def add_background(pos: wp.array(dtype=wp.vec2), acc: wp.array(dtype=wp.vec2),
                   components: wp.array(dtype=wp.vec2), count: int, G: float):
    i = wp.tid()
    p = pos[i]
    a = acc[i]
    for k in range(count):
        c = components[k]
        inv = (1.0 / wp.sqrt(wp.dot(p, p) + c[1] * c[1]))
        a -= G * c[0] * inv * inv * inv * p
    acc[i] = a


@wp.kernel
def drift(pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
          acc: wp.array(dtype=wp.vec2), dt: float):
    i = wp.tid()
    vel[i] = vel[i] + (0.5 * dt) * acc[i]
    pos[i] = pos[i] + dt * vel[i]


@wp.kernel
def kick(vel: wp.array(dtype=wp.vec2), acc: wp.array(dtype=wp.vec2), dt: float):
    i = wp.tid()
    vel[i] = vel[i] + (0.5 * dt) * acc[i]


def cuda_available():
    wp.init()
    return wp.is_cuda_available()


class DeviceState:
    """FP32 device state; CPU mode exists solely for validation.

    Positive softening is required. This avoids singular coincident-particle
    states and makes the CUDA contract explicit. Host edits are loaded via
    upload(); there is no implicit stale-state detection inside a device batch.
    """
    def __init__(self, bodies, G, softening, background=None, device="cuda:0"):
        wp.init()
        self.device = wp.get_device(device)
        self.n = len(bodies)
        self.size = max(TILE, ((self.n + TILE - 1) // TILE) * TILE)
        if not math.isfinite(G) or not math.isfinite(softening) or softening <= 0:
            raise ValueError("CUDA gravity requires finite G and positive softening")
        self.G, self.eps2 = float(G), float(softening * softening)
        self.pos = wp.zeros(self.size, dtype=wp.vec2, device=self.device)
        self.vel = wp.zeros(self.size, dtype=wp.vec2, device=self.device)
        self.mass = wp.zeros(self.size, dtype=float, device=self.device)
        self.acc = wp.zeros(self.size, dtype=wp.vec2, device=self.device)
        components = background.components if background else ()
        self.background = wp.array(np.asarray(components, dtype=np.float32).reshape(-1, 2),
                                   dtype=wp.vec2, device=self.device)
        self.background_count = len(components)
        self._graphs = {}
        self._graph_dt = None
        self.upload(bodies)

    def upload(self, bodies):
        if len(bodies) != self.n:
            raise ValueError("Recreate device state after changing body count")
        state = np.asarray([(b.x, b.y, b.vx, b.vy, b.mass) for b in bodies],
                           dtype=np.float32).reshape(-1, 5)
        if not np.isfinite(state).all() or (state[:, 4] < 0).any():
            raise ValueError("Finite FP32 positions/velocities and nonnegative masses required")
        pos = np.zeros((self.size, 2), dtype=np.float32)
        vel = np.zeros_like(pos)
        mass = np.zeros(self.size, dtype=np.float32)
        pos[:self.n], vel[:self.n], mass[:self.n] = state[:, :2], state[:, 2:4], state[:, 4]
        self.pos.assign(pos)
        self.vel.assign(vel)
        self.mass.assign(mass)
        self.acc_valid = False

    def force(self):
        if self.n:
            kernel = tiled_force if self.device.is_cuda else scalar_force
            wp.launch(kernel, dim=self.size if self.device.is_cuda else self.n,
                      inputs=[self.pos, self.mass, self.acc, self.n, self.eps2, self.G],
                      device=self.device, block_dim=TILE)
            if self.background_count:
                wp.launch(add_background, dim=self.n,
                          inputs=[self.pos, self.acc, self.background, self.background_count, self.G],
                          device=self.device, block_dim=TILE)
        self.acc_valid = True
        return self.acc

    def step(self, dt):
        if not math.isfinite(dt):
            raise ValueError("Finite timestep required")
        if not self.n:
            return
        if not self.acc_valid:
            self.force()
        wp.launch(drift, dim=self.n, inputs=[self.pos, self.vel, self.acc, dt], device=self.device, block_dim=TILE)
        self.force()
        wp.launch(kick, dim=self.n, inputs=[self.vel, self.acc, dt], device=self.device, block_dim=TILE)

    def step_batch(self, dt, count, *, use_graphs=True):
        """Enqueue a bounded batch; caller synchronizes once at its boundary.

        Arrays remain authoritative on device. Graphs capture no uploads or
        downloads. A timestep change invalidates graph arguments, not forces.
        """
        if not math.isfinite(dt) or dt <= 0 or not isinstance(count, int) or not 0 <= count <= 32:
            raise ValueError("Positive finite dt and integer batch size 0..32 required")
        if not count or not self.n:
            return
        if not self.acc_valid:
            self.force()
        if not self.device.is_cuda or not use_graphs:
            for _ in range(count):
                self.step(dt)
            return
        if dt != self._graph_dt:
            self._graphs.clear()
            self._graph_dt = dt
        if count not in self._graphs:
            # Compile before capture; kernels share a fixed launch block size.
            wp.load_module(module=__name__, device=self.device, block_dim=TILE)
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(count):
                    self.step(dt)
            if len(self._graphs) >= 8:
                self._graphs.clear()
            self._graphs[count] = capture.graph
        wp.capture_launch(self._graphs[count])

    def snapshot(self):
        """Detached host arrays; consumers never share mutable device storage."""
        return self.pos.numpy()[:self.n].copy(), self.vel.numpy()[:self.n].copy()

    def synchronize(self):
        wp.synchronize_device(self.device)

    def download(self, bodies):
        if len(bodies) != self.n:
            raise ValueError("Do not edit body count during a device batch")
        pos, vel = self.pos.numpy(), self.vel.numpy()
        for i, b in enumerate(bodies):
            b.x, b.y = map(float, pos[i])
            b.vx, b.vy = map(float, vel[i])

    def torch_views(self):
        """Zero-copy live tensors; requires separately installed PyTorch.

        Synchronize before crossing frameworks. External writes must be followed
        by torch.cuda.synchronize() (on CUDA) and acc_valid=False before stepping.
        """
        self.synchronize()
        return {"positions": wp.to_torch(self.pos)[:self.n],
                "velocities": wp.to_torch(self.vel)[:self.n],
                "masses": wp.to_torch(self.mass)[:self.n]}


def verify_device(device="cuda:0", *, use_graphs=True):
    """Small startup check of unequal masses, a partial tile, force and integration.

    Returns device name after comparison to the existing FP64 Python solver.
    This executes on the user's GPU before the interactive app selects CUDA.
    """
    from gravity_sim import Body, NBodySimulation, PlummerBackground
    rng = np.random.default_rng(2718)
    bodies = [Body(float(p[0]), float(p[1]), 0, 0, float(m))
              for p, m in zip(rng.normal(size=(65, 2)), rng.uniform(.1, 2, 65))]
    bg = PlummerBackground(((3., 2.),))
    ref = NBodySimulation(bodies, solver="exact", softening=.4, background=bg)
    expected = np.asarray(ref.accelerations())
    state = DeviceState(bodies, 1, .4, bg, device=device)
    actual = state.force().numpy()[:65]
    if not np.allclose(actual, expected, rtol=5e-5, atol=5e-5):
        raise RuntimeError("CUDA force startup check failed; CPU solver remains available")
    ref.step(.001)
    state.step(.001)
    expected_pos = np.asarray([(b.x, b.y) for b in bodies])
    expected_vel = np.asarray([(b.vx, b.vy) for b in bodies])
    if not (np.allclose(state.pos.numpy()[:65], expected_pos, rtol=5e-5, atol=5e-5)
            and np.allclose(state.vel.numpy()[:65], expected_vel, rtol=5e-5, atol=5e-5)):
        raise RuntimeError("CUDA integration startup check failed")
    # Exercise graph capture, replay and dt invalidation on actual CUDA devices.
    for dt, count in ((.001, 4), (.001, 4), (.002, 2)):
        state.step_batch(dt, count, use_graphs=use_graphs)
        for _ in range(count):
            ref.step(dt)
        if not (np.allclose(state.pos.numpy()[:65], [(b.x,b.y) for b in bodies], rtol=5e-5, atol=5e-5)
                and np.allclose(state.vel.numpy()[:65], [(b.vx,b.vy) for b in bodies], rtol=5e-5, atol=5e-5)):
            raise RuntimeError("CUDA batch/graph startup check failed")
    return state.device.name
