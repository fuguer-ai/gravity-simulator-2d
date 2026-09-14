from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass
class Body:
    """A point-mass body for the 2D gravity simulation."""

    x: float
    y: float
    vx: float
    vy: float
    mass: float
    radius: float = 6.0
    color: tuple[int, int, int] = (240, 240, 240)
    name: str = "Body"


@dataclass(frozen=True)
class PlummerBackground:
    """Fixed spherical mass components (mass, scale radius), centered at origin.

    Phi = -G sum M/sqrt(r²+a²). This is an external field: disk momentum alone
    is not conserved; total stellar energy includes the background potential.
    """
    components: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if any(not math.isfinite(m) or m < 0 or not math.isfinite(a) or a <= 0
               for m, a in self.components):
            raise ValueError("Background masses must be nonnegative and scales positive")

    def acceleration(self, x, y, G=1.0):
        r2 = x*x+y*y
        factor = -G*sum(m/(r2+a*a)**1.5 for m,a in self.components)
        return factor*x, factor*y

    def potential(self, x, y, G=1.0):
        r2 = x*x+y*y
        return -G*sum(m/math.sqrt(r2+a*a) for m,a in self.components)


class _QuadNode:
    """Internal node for the Barnes-Hut quadtree."""

    __slots__ = (
        "cx",
        "cy",
        "half",
        "depth",
        "mass",
        "com_x",
        "com_y",
        "indices",
        "children",
    )

    LEAF_CAPACITY = 4
    MAX_DEPTH = 32

    def __init__(self, cx: float, cy: float, half: float, depth: int = 0) -> None:
        self.cx = cx
        self.cy = cy
        self.half = half
        self.depth = depth
        self.mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.indices: list[int] = []
        self.children: list[_QuadNode] | None = None

    def contains(self, x: float, y: float) -> bool:
        return (
            self.cx - self.half <= x <= self.cx + self.half
            and self.cy - self.half <= y <= self.cy + self.half
        )

    def _child_index(self, x: float, y: float) -> int:
        east = 1 if x >= self.cx else 0
        south = 2 if y >= self.cy else 0
        return east + south

    def _subdivide(self) -> None:
        q = self.half * 0.5
        self.children = [
            _QuadNode(self.cx - q, self.cy - q, q, self.depth + 1),
            _QuadNode(self.cx + q, self.cy - q, q, self.depth + 1),
            _QuadNode(self.cx - q, self.cy + q, q, self.depth + 1),
            _QuadNode(self.cx + q, self.cy + q, q, self.depth + 1),
        ]

    def insert(self, body_index: int, bodies: list[Body]) -> None:
        body = bodies[body_index]

        new_mass = self.mass + body.mass
        if new_mass != 0.0:
            self.com_x = (self.com_x * self.mass + body.x * body.mass) / new_mass
            self.com_y = (self.com_y * self.mass + body.y * body.mass) / new_mass
        self.mass = new_mass

        if self.children is None:
            if len(self.indices) < self.LEAF_CAPACITY or self.depth >= self.MAX_DEPTH:
                self.indices.append(body_index)
                return

            old_indices = self.indices
            self.indices = []
            self._subdivide()
            assert self.children is not None
            for old_index in old_indices:
                old_body = bodies[old_index]
                child = self.children[self._child_index(old_body.x, old_body.y)]
                child.insert(old_index, bodies)

        assert self.children is not None
        child = self.children[self._child_index(body.x, body.y)]
        child.insert(body_index, bodies)


class NBodySimulation:
    """2D Newtonian N-body simulator with exact, Barnes-Hut and FMM solvers.

    Small systems use exact O(n^2) pairwise gravity. Larger systems switch to
    the Barnes-Hut quadtree algorithm, which is typically O(n log n).
    Velocity-Verlet is used for time integration.
    """

    def __init__(
        self,
        bodies: Iterable[Body] | None = None,
        *,
        gravitational_constant: float = 1.0,
        softening: float = 3.0,
        barnes_hut_theta: float = 0.7,
        barnes_hut_threshold: int = 64,
        solver: str = "auto",
        fmm_theta: float = 0.5,
        fmm_leaf_capacity: int = 16,
        background: PlummerBackground | None = None,
        solver_mode: str | None = None,
    ) -> None:
        if solver_mode is not None:
            solver = solver_mode
        if solver not in ("auto", "exact", "barnes-hut", "fmm", "cuda-exact"):
            raise ValueError("Unknown solver: " + solver)
        self._gpu_state = None
        self._gpu_batch = False
        self.cuda_enabled = solver == "cuda-exact"
        self.background = background
        self.solver = solver
        self.fmm_theta = float(fmm_theta)
        self.fmm_leaf_capacity = fmm_leaf_capacity
        self.bodies = list(bodies or [])
        self.G = float(gravitational_constant)
        self.softening = float(softening)
        self.barnes_hut_theta = float(barnes_hut_theta)
        self.barnes_hut_threshold = int(barnes_hut_threshold)

    def accelerations(self) -> list[tuple[float, float]]:
        """Return each body's acceleration.

        Explicit solver choices override automatic dispatch. In auto mode,
        exact is used below ``barnes_hut_threshold`` and Barnes-Hut above it.
        """
        result = self._self_accelerations()
        if self.background is not None:
            result = [(ax+bx, ay+by) for body, (ax, ay) in zip(self.bodies, result)
                      for bx, by in [self.background.acceleration(body.x, body.y, self.G)]]
        return result

    def _self_accelerations(self) -> list[tuple[float, float]]:
        selected = self.active_solver
        if selected == "cuda-exact":
            from cuda_gravity import DeviceState
            state = DeviceState(self.bodies, self.G, self.softening)
            return [tuple(map(float, a)) for a in state.force().numpy()[:len(self.bodies)]]
        if selected == "exact":
            return self._accelerations_exact()
        if selected == "barnes-hut":
            return self._accelerations_barnes_hut()
        if selected == "fmm":
            from fmm import accelerations
            return accelerations(self.bodies, self.G, self.softening,
                                 self.fmm_theta, self.fmm_leaf_capacity)
        raise ValueError("Unknown solver: " + selected)

    @property
    def solver_mode(self) -> str:
        """Compatibility alias for callers of the earlier main-branch engine."""
        return self.solver

    @solver_mode.setter
    def solver_mode(self, value: str) -> None:
        if value not in ("auto", "exact", "barnes-hut", "fmm", "cuda-exact"):
            raise ValueError("Unknown solver: " + value)
        self.solver = value

    def cycle_solver(self) -> str:
        order = ("auto", "fmm", "barnes-hut", "exact")
        if self.cuda_enabled or self.solver == "cuda-exact":
            order += ("cuda-exact",)
        self.solver_mode = order[(order.index(self.solver_mode)+1) % len(order)]
        return self.solver_mode

    @property
    def active_solver(self) -> str:
        if self.solver == "auto":
            return "exact" if len(self.bodies) < self.barnes_hut_threshold else "barnes-hut"
        return self.solver

    def _accelerations_exact(self) -> list[tuple[float, float]]:
        n = len(self.bodies)
        acc = [[0.0, 0.0] for _ in range(n)]
        eps2 = self.softening * self.softening

        # Symmetric pair update preserves Newton's third law exactly.
        for i in range(n):
            a = self.bodies[i]
            for j in range(i + 1, n):
                b = self.bodies[j]
                dx = b.x - a.x
                dy = b.y - a.y
                r2 = dx * dx + dy * dy + eps2
                inv_r3 = 1.0 / (r2 * math.sqrt(r2))

                factor_i = self.G * b.mass * inv_r3
                factor_j = self.G * a.mass * inv_r3

                acc[i][0] += dx * factor_i
                acc[i][1] += dy * factor_i
                acc[j][0] -= dx * factor_j
                acc[j][1] -= dy * factor_j

        return [(ax, ay) for ax, ay in acc]

    def _build_quadtree(self) -> _QuadNode | None:
        if not self.bodies:
            return None

        min_x = min(body.x for body in self.bodies)
        max_x = max(body.x for body in self.bodies)
        min_y = min(body.y for body in self.bodies)
        max_y = max(body.y for body in self.bodies)

        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        span = max(max_x - min_x, max_y - min_y)
        half = max(0.5 * span * 1.000001, 1e-9)

        root = _QuadNode(cx, cy, half)
        for i in range(len(self.bodies)):
            root.insert(i, self.bodies)
        return root

    def _accelerations_barnes_hut(self) -> list[tuple[float, float]]:
        root = self._build_quadtree()
        if root is None:
            return []

        eps2 = self.softening * self.softening
        theta = self.barnes_hut_theta
        result: list[tuple[float, float]] = []

        for target_index, target in enumerate(self.bodies):
            ax = 0.0
            ay = 0.0
            stack = [root]

            while stack:
                node = stack.pop()
                if node.mass == 0.0:
                    continue

                if node.children is None:
                    # Nearby bodies in leaves are still evaluated exactly.
                    for source_index in node.indices:
                        if source_index == target_index:
                            continue
                        source = self.bodies[source_index]
                        dx = source.x - target.x
                        dy = source.y - target.y
                        r2 = dx * dx + dy * dy + eps2
                        inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                        factor = self.G * source.mass * inv_r3
                        ax += dx * factor
                        ay += dy * factor
                    continue

                dx = node.com_x - target.x
                dy = node.com_y - target.y
                distance_sq = dx * dx + dy * dy
                distance = math.sqrt(distance_sq) if distance_sq > 0.0 else 0.0
                width = 2.0 * node.half

                # A node containing the target cannot be collapsed because its
                # aggregate mass also contains that target's own mass.
                can_approximate = (
                    not node.contains(target.x, target.y)
                    and distance > 0.0
                    and width / distance < theta
                )

                if can_approximate:
                    r2 = distance_sq + eps2
                    inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                    factor = self.G * node.mass * inv_r3
                    ax += dx * factor
                    ay += dy * factor
                else:
                    stack.extend(node.children)

            result.append((ax, ay))

        return result

    def begin_device_frame(self):
        """Upload public Body state once; device steps then run without host copies.

        Call end_device_frame before reading/editing bodies or changing solvers.
        The application brackets each physics frame with this pair.
        """
        if self._gpu_batch:
            raise RuntimeError("Device frame already open")
        if self.active_solver != "cuda-exact":
            return
        from cuda_gravity import DeviceState
        config = (len(self.bodies), self.G, self.softening, self.background)
        if self._gpu_state is None or self._gpu_config != config:
            self._gpu_state = DeviceState(self.bodies, self.G, self.softening, self.background)
            self._gpu_config = config
        else:
            self._gpu_state.upload(self.bodies)
        self._gpu_batch = True

    def end_device_frame(self):
        if self._gpu_batch:
            try:
                self._gpu_state.download(self.bodies)
            finally:
                self._gpu_batch = False

    def step(self, dt: float) -> None:
        """Advance one time step using velocity-Verlet integration."""
        if not self.bodies:
            return

        if self.active_solver == "cuda-exact":
            owns_frame = not self._gpu_batch
            if owns_frame:
                self.begin_device_frame()
            try:
                self._gpu_state.step(dt)
                # Charge actual GPU work to the pacing budget, not just enqueue time.
                self._gpu_state.synchronize()
            finally:
                if owns_frame:
                    self.end_device_frame()
            return

        a0 = self.accelerations()
        half_dt2 = 0.5 * dt * dt

        for body, (ax, ay) in zip(self.bodies, a0):
            body.x += body.vx * dt + ax * half_dt2
            body.y += body.vy * dt + ay * half_dt2

        a1 = self.accelerations()
        half_dt = 0.5 * dt

        for body, (ax0, ay0), (ax1, ay1) in zip(self.bodies, a0, a1):
            body.vx += (ax0 + ax1) * half_dt
            body.vy += (ay0 + ay1) * half_dt

    def total_momentum(self) -> tuple[float, float]:
        px = sum(body.mass * body.vx for body in self.bodies)
        py = sum(body.mass * body.vy for body in self.bodies)
        return px, py

    def total_energy(self) -> float:
        kinetic = sum(
            0.5 * body.mass * (body.vx * body.vx + body.vy * body.vy)
            for body in self.bodies
        )

        potential = 0.0
        eps2 = self.softening * self.softening
        for i, a in enumerate(self.bodies):
            for b in self.bodies[i + 1 :]:
                dx = b.x - a.x
                dy = b.y - a.y
                distance = math.sqrt(dx * dx + dy * dy + eps2)
                potential -= self.G * a.mass * b.mass / distance

        if self.background is not None:
            potential += sum(b.mass*self.background.potential(b.x,b.y,self.G)
                             for b in self.bodies)
        return kinetic + potential
