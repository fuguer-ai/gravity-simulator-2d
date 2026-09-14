from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import pygame
import numpy as np

from galaxy_render import GalaxyLighting
from simulation_clock import FixedStepClock
from gravity_sim import Body, NBodySimulation
from galaxy import spiral_galaxy, BACKGROUND as GALAXY_BACKGROUND, SOFTENING as GALAXY_SOFTENING

WIDTH, HEIGHT = 1440, 900
BACKGROUND = (10, 13, 22)
GRID = (26, 31, 44)
TEXT = (225, 230, 240)
MUTED = (150, 160, 180)
ACCENT = (95, 180, 255)
PANEL = (19, 24, 36)
PANEL_HOVER = (31, 39, 56)
LOCK_COLOR = (255, 235, 120)

PALETTE = [
    (140, 220, 255),
    (255, 150, 120),
    (180, 255, 160),
    (225, 180, 255),
    (255, 220, 110),
    (245, 245, 245),
    (120, 150, 255),
    (255, 110, 180),
]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str


SCENARIOS = [
    Scenario("solar", "Solar System", "Sun + 8 planets with realistic relative masses and orbital spacing."),
    Scenario("binary", "Binary Star System", "Two stars orbiting each other with circumbinary planets."),
    Scenario("random", "Randomized System", "A fresh random star cluster / planetary system every time."),
    Scenario("galaxy", "Spiral Galaxy", "Exponential stellar disk, smooth bulge and halo, evolving spiral structure."),
]


def solar_system() -> list[Body]:
    """Solar System using realistic relative masses and semi-major-axis ratios."""
    au = 18.0
    sun = Body(0, 0, 0, 0, mass=1.0, radius=14, color=(255, 210, 80), name="Sun")
    data = [
        ("Mercury", 0.387, 1.66e-7, 3.0, (170, 165, 155)),
        ("Venus", 0.723, 2.45e-6, 4.5, (220, 185, 105)),
        ("Earth", 1.000, 3.00e-6, 4.8, (70, 145, 255)),
        ("Mars", 1.524, 3.23e-7, 4.0, (215, 95, 65)),
        ("Jupiter", 5.203, 9.54e-4, 8.8, (220, 175, 125)),
        ("Saturn", 9.537, 2.86e-4, 8.0, (225, 200, 135)),
        ("Uranus", 19.191, 4.37e-5, 6.7, (135, 220, 230)),
        ("Neptune", 30.069, 5.15e-5, 6.5, (75, 115, 245)),
    ]
    bodies = [sun]
    phases = [0.2, 1.1, 2.2, 3.1, 4.0, 5.0, 0.9, 2.7]
    for (name, semi_major_au, mass, radius, color), phase in zip(data, phases):
        r = semi_major_au * au
        speed = math.sqrt(sun.mass / r)
        x = math.cos(phase) * r
        y = math.sin(phase) * r
        vx = -math.sin(phase) * speed
        vy = math.cos(phase) * speed
        bodies.append(Body(x, y, vx, vy, mass, radius, color, name))
    return bodies


def binary_system() -> list[Body]:
    """A compact binary-star system with circumbinary planets."""
    m1, m2 = 700.0, 520.0
    separation = 95.0
    total = m1 + m2
    r1 = separation * m2 / total
    r2 = separation * m1 / total
    omega = math.sqrt(total / separation**3)

    star_a = Body(-r1, 0, 0, -omega * r1, m1, 12, (255, 215, 120), "Helios A")
    star_b = Body(r2, 0, 0, omega * r2, m2, 10, (255, 150, 110), "Helios B")
    bodies = [star_a, star_b]

    for idx, (r, mass, radius, color) in enumerate(
        [
            (190.0, 5.0, 5.0, (100, 175, 255)),
            (285.0, 9.0, 6.0, (180, 240, 155)),
            (405.0, 15.0, 7.0, (215, 155, 240)),
        ],
        start=1,
    ):
        phase = 0.8 + idx * 1.7
        speed = math.sqrt(total / r)
        bodies.append(
            Body(
                math.cos(phase) * r,
                math.sin(phase) * r,
                -math.sin(phase) * speed,
                math.cos(phase) * speed,
                mass,
                radius,
                color,
                f"Circumbinary {idx}",
            )
        )
    return bodies


def randomized_system(count: int = 120) -> list[Body]:
    """Generate a randomized rotating system large enough to exercise Barnes-Hut."""
    star_mass = random.uniform(900.0, 1800.0)
    bodies = [Body(0, 0, 0, 0, star_mass, 14, (255, 210, 95), "Primary")]

    for i in range(count):
        r = random.uniform(55.0, 520.0)
        phase = random.random() * math.tau
        tangent = math.sqrt(star_mass / r) * random.uniform(0.78, 1.18)
        radial = random.uniform(-0.15, 0.15)
        mass = 10 ** random.uniform(-1.0, 1.2)
        color = random.choice(PALETTE)
        radius = max(2.5, min(8.5, 2.4 + math.sqrt(mass)))
        bodies.append(
            Body(
                math.cos(phase) * r,
                math.sin(phase) * r,
                -math.sin(phase) * tangent + math.cos(phase) * radial,
                math.cos(phase) * tangent + math.sin(phase) * radial,
                mass,
                radius,
                color,
                f"Random {i + 1}",
            )
        )
    return bodies


def scenario_bodies(key: str, galaxy_particles: int = 1200) -> tuple[list[Body], float, float]:
    if key == "solar":
        return solar_system(), 0.08, 0.58
    if key == "binary":
        return binary_system(), 2.5, 0.95
    if key == "galaxy":
        return spiral_galaxy(star_count=galaxy_particles), GALAXY_SOFTENING, 0.62
    return randomized_system(), 3.0, 0.92


class GravityApp:
    def __init__(self, solver="fmm", galaxy_particles=1200, *, renderer="auto", threaded=False, use_graphs=True) -> None:
        pygame.init()
        pygame.display.set_caption("2D Gravity Simulator")
        self.renderer = None
        if renderer != "software":
            try:
                from gpu_render import GalaxyRenderer
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
                pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                self.renderer = GalaxyRenderer()
            except Exception as exc:
                if renderer == "opengl":
                    raise RuntimeError(f"OpenGL renderer could not start: {exc}") from exc
                print(f"Using software renderer: {exc}")
        if self.renderer is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.threaded = threaded
        self.use_graphs = use_graphs
        self._worker = None
        self._frame_controller = None
        self._snapshot_seq = -1
        self.render_positions = None
        self._render_token = 0
        self.worker_stats = None
        self.show_help = False
        self.show_hud = True
        self._menu_preview = None
        self.gpu_profile = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.big_font = pygame.font.SysFont("segoeui", 46)
        self.title_font = pygame.font.SysFont("consolas", 23, bold=True)

        self.galaxy_particles = galaxy_particles
        self.show_tree = False
        self.diagnostic_points = []
        self.diagnostic_text = "H: spatial tree   A: pause + force error   U: CUDA exact"
        self.galaxy_lighting = GalaxyLighting()
        self.show_glow = True
        self.physics_clock = FixedStepClock()
        self.mode = "menu"
        self.current_scenario = "solar"
        self.sim = NBodySimulation([], softening=3.0, solver=solver)
        self.paused = False
        self.show_trails = True
        self.time_scale = 1.0
        self.effective_time_scale = 1.0
        self.zoom = 1.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.trail_maxlen = 600
        self.trails: list[deque[tuple[float, float]]] = []
        self.drag_start: tuple[float, float] | None = None

        self.reference_body: Body | None = None
        self.reference_pick_armed = False

        self.spawn_color_index = 0
        self.spawn_radius = 6.0
        self.spawn_mass = 8.0
        self.menu_rects: dict[str, pygame.Rect] = {}

    def ensure_trails(self) -> None:
        while len(self.trails) < len(self.sim.bodies):
            self.trails.append(deque(maxlen=self.trail_maxlen))
        if len(self.trails) > len(self.sim.bodies):
            self.trails = self.trails[: len(self.sim.bodies)]

    def reference_index(self) -> int | None:
        if self.reference_body is None:
            return None
        for i, body in enumerate(self.sim.bodies):
            if body is self.reference_body:
                return i
        self.reference_body = None
        return None

    def reference_origin(self) -> tuple[float, float]:
        idx = self.reference_index()
        if idx is None:
            return 0.0, 0.0
        body = self.sim.bodies[idx]
        return body.x, body.y

    def clear_reference(self) -> None:
        if self.renderer:
            self.renderer.reset_history()
        self.reference_body = None
        self.reference_pick_armed = False
        self.camera_x = 0.0
        self.camera_y = 0.0

    def load_scenario(self, key: str) -> None:
        self.stop_worker()
        bodies, softening, zoom = scenario_bodies(key, self.galaxy_particles)
        self.physics_clock = FixedStepClock(timestep=.5 if key == "galaxy" else .045, budget_seconds=.008)
        self.current_scenario = key
        cuda_enabled = self.sim.cuda_enabled
        self.sim = NBodySimulation(bodies, softening=softening, solver=self.sim.solver,
                                   background=GALAXY_BACKGROUND if key == "galaxy" else None)
        self.sim.cuda_enabled = cuda_enabled
        self.diagnostic_points = []
        self.diagnostic_text = "H: spatial tree   A: pause + force error   U: CUDA exact"
        self.trail_maxlen = 120 if key == "galaxy" else 600
        self.trails = []
        self.ensure_trails()
        self.show_trails = key != "galaxy"
        self.time_scale = 1.0
        self.effective_time_scale = 1.0
        self.zoom = min(self.screen.get_width(), self.screen.get_height())*.40/850 if key == "galaxy" else zoom
        self.camera_x = self.camera_y = 0.0
        self.reference_body = None
        self.reference_pick_armed = False
        self.paused = False
        self.render_positions = None
        self.worker_stats = None
        if self.renderer:
            self.renderer.reset_history()
        self.mode = "simulation"

    def reset(self) -> None:
        self.load_scenario(self.current_scenario)

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.screen.get_size()
        ref_x, ref_y = self.reference_origin()
        sx = (x - ref_x - self.camera_x) * self.zoom + w / 2
        sy = (y - ref_y - self.camera_y) * self.zoom + h / 2
        return int(sx), int(sy)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        w, h = self.screen.get_size()
        ref_x, ref_y = self.reference_origin()
        x = (sx - w / 2) / self.zoom + self.camera_x + ref_x
        y = (sy - h / 2) / self.zoom + self.camera_y + ref_y
        return x, y

    def nearest_body_on_screen(
        self, pos: tuple[int, int], max_distance_px: float = 28.0
    ) -> Body | None:
        best_body = None
        best_d2 = max_distance_px * max_distance_px
        for body in self.sim.bodies:
            sx, sy = self.world_to_screen(body.x, body.y)
            d2 = (sx - pos[0]) ** 2 + (sy - pos[1]) ** 2
            if d2 <= best_d2:
                best_body = body
                best_d2 = d2
        return best_body

    def choose_reference_body(self, pos: tuple[int, int]) -> None:
        body = self.nearest_body_on_screen(pos)
        if body is not None:
            if self.renderer:
                self.renderer.reset_history()
            self.reference_body = body
            self.reference_pick_armed = False
            self.camera_x = 0.0
            self.camera_y = 0.0

    @property
    def spawn_color(self) -> tuple[int, int, int]:
        return PALETTE[self.spawn_color_index]

    def add_body(self, pos: tuple[int, int]) -> None:
        self.stop_worker()
        self.render_positions = None
        x, y = self.screen_to_world(*pos)
        angle = random.random() * math.tau
        speed = random.uniform(0.15, 1.0)

        ref_vx = self.reference_body.vx if self.reference_index() is not None else 0.0
        ref_vy = self.reference_body.vy if self.reference_index() is not None else 0.0

        self.sim.bodies.append(
            Body(
                x,
                y,
                ref_vx + math.cos(angle) * speed,
                ref_vy + math.sin(angle) * speed,
                mass=self.spawn_mass,
                radius=self.spawn_radius,
                color=self.spawn_color,
                name=f"Custom {len(self.sim.bodies) + 1}",
            )
        )
        self.ensure_trails()
        self.diagnostic_points = []

    def remove_nearest(self, pos: tuple[int, int]) -> None:
        self.stop_worker()
        self.render_positions = None
        if not self.sim.bodies:
            return
        x, y = self.screen_to_world(*pos)
        index = min(
            range(len(self.sim.bodies)),
            key=lambda i: (self.sim.bodies[i].x - x) ** 2 + (self.sim.bodies[i].y - y) ** 2,
        )
        removed = self.sim.bodies[index]
        del self.sim.bodies[index]
        del self.trails[index]
        self.diagnostic_points = []
        if removed is self.reference_body:
            self.clear_reference()

    def handle_menu_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_1:
                self.load_scenario("solar")
            elif event.key == pygame.K_2:
                self.load_scenario("binary")
            elif event.key == pygame.K_3:
                self.load_scenario("random")
            elif event.key == pygame.K_4:
                self.load_scenario("galaxy")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in self.menu_rects.items():
                if rect.collidepoint(event.pos):
                    self.load_scenario(key)
                    break
        return True

    def handle_simulation_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key in (pygame.K_SPACE, pygame.K_a, pygame.K_s, pygame.K_b, pygame.K_u, pygame.K_m):
                self.stop_worker()
            if event.key == pygame.K_TAB:
                self.show_help = not self.show_help
            elif event.key == pygame.K_F1:
                self.show_hud = not self.show_hud
            elif event.key == pygame.K_v and self.renderer:
                self.renderer.mode = "density" if self.renderer.mode == "cinematic" else "cinematic"
            elif event.key in (pygame.K_COMMA, pygame.K_PERIOD) and self.renderer:
                factor = 1.2 if event.key == pygame.K_PERIOD else 1/1.2
                self.renderer.exposure = min(8., max(.15, self.renderer.exposure*factor))
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_h:
                self.show_tree = not self.show_tree
            elif event.key == pygame.K_a:
                from diagnostics import force_sample
                self.paused = True
                self.diagnostic_points, error = force_sample(self.sim)
                self.diagnostic_text = f"Paused force sample: RMS error {error:.3%}; green <1%, amber <5%, red >=5%"
            elif event.key == pygame.K_u:
                try:
                    from cuda_gravity import cuda_available, verify_device
                    if not cuda_available():
                        raise RuntimeError("No CUDA device detected")
                    # Compile and exercise the real kernel before changing the selection.
                    verify_device(use_graphs=self.use_graphs)
                    self.sim.cuda_enabled = True
                    self.sim.solver_mode = "cuda-exact"
                    self.diagnostic_text = "CUDA exact active (FP32). S/B cycles all enabled solvers."
                except (ImportError, RuntimeError) as exc:
                    self.diagnostic_text = "CUDA unavailable; see console. CPU solvers remain active."
                    print(f"CUDA setup: {exc}. Run run_windows_cuda.bat to install Warp.", file=sys.stderr)
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_m:
                self.mode = "menu"
            elif event.key in (pygame.K_s, pygame.K_b):
                self.sim.cycle_solver()
                self.diagnostic_points = []
                self.diagnostic_text = "H: spatial tree   A: pause + force error   U: CUDA exact"
            elif event.key == pygame.K_PAGEUP:
                self.physics_clock.adjust_timestep(1)
            elif event.key == pygame.K_PAGEDOWN:
                self.physics_clock.adjust_timestep(-1)
            elif event.key in (pygame.K_0, pygame.K_KP0):
                self.physics_clock.adjust_timestep()
            elif event.key == pygame.K_g:
                self.show_glow = not self.show_glow
            elif event.key == pygame.K_t:
                self.show_trails = not self.show_trails
            elif event.key == pygame.K_f:
                if self.reference_body is not None:
                    self.clear_reference()
                else:
                    self.reference_pick_armed = not self.reference_pick_armed
            elif event.key == pygame.K_c:
                self.spawn_color_index = (self.spawn_color_index + 1) % len(PALETTE)
            elif event.key in (pygame.K_RIGHTBRACKET,):
                self.spawn_radius = min(30.0, self.spawn_radius + 1.0)
                self.spawn_mass = min(200.0, self.spawn_mass * 1.35)
            elif event.key in (pygame.K_LEFTBRACKET,):
                self.spawn_radius = max(2.0, self.spawn_radius - 1.0)
                self.spawn_mass = max(0.1, self.spawn_mass / 1.35)
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.time_scale = min(4096.0, self.time_scale * 2.0)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.time_scale = max(0.0625, self.time_scale / 2.0)
        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.12 ** event.y
            self.zoom = max(0.03, min(12.0, self.zoom * factor))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.reference_pick_armed:
                    self.choose_reference_body(event.pos)
                else:
                    self.add_body(event.pos)
            elif event.button == 3:
                self.remove_nearest(event.pos)
            elif event.button == 2 and self.reference_body is None:
                self.drag_start = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self.drag_start = None
        elif event.type == pygame.MOUSEMOTION and self.drag_start is not None:
            dx = event.pos[0] - self.drag_start[0]
            dy = event.pos[1] - self.drag_start[1]
            if self.renderer:
                self.renderer.reset_history()
            self.camera_x -= dx / self.zoom
            self.camera_y -= dy / self.zoom
            self.drag_start = event.pos
        return True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.VIDEORESIZE and self.renderer:
            self.screen = pygame.Surface((max(1,event.w), max(1,event.h)), pygame.SRCALPHA)
            self.renderer.reset_history()
            return True
        if self.mode == "menu":
            return self.handle_menu_event(event)
        return self.handle_simulation_event(event)

    def consume_snapshot(self, snapshot):
        if snapshot is None or snapshot.sequence == self._snapshot_seq:
            return False
        if len(snapshot.positions) != len(self.sim.bodies):
            raise RuntimeError("Stale physics snapshot: particle counts differ")
        self._snapshot_seq = snapshot.sequence
        self.render_positions = snapshot.positions
        for body, p, v in zip(self.sim.bodies, snapshot.positions, snapshot.velocities):
            body.x, body.y = map(float, p)
            body.vx, body.vy = map(float, v)
        self.effective_time_scale = snapshot.effective_speed
        self.worker_stats = snapshot
        self._render_token += 1
        return True

    def stop_worker(self):
        # Coordinated state is already synchronized and reflected in Body objects.
        self._frame_controller = None
        if self._worker is not None:
            worker = self._worker
            snapshot = worker.close()
            self.consume_snapshot(snapshot)
            self._worker = None
            self._snapshot_seq = -1
            if worker.error:
                self.paused = True
                self.diagnostic_text = "Physics stopped: " + worker.error
                print(self.diagnostic_text, file=sys.stderr)

    def update(self, elapsed: float = 1/60) -> None:
        if self.mode != "simulation":
            return
        changed = False
        if self.sim.active_solver == "cuda-exact" and self.threaded:
            if not self.paused and self._worker is None:
                from physics_worker import PhysicsWorker
                self._snapshot_seq = -1
                self._worker = PhysicsWorker(self.sim.bodies, self.sim.G, self.sim.softening,
                    self.sim.background, dt=self.physics_clock.timestep, speed=self.time_scale,
                    use_graphs=self.use_graphs)
            if self._worker:
                self._worker.controls(self.physics_clock.timestep, self.time_scale, self.paused)
                changed = self.consume_snapshot(self._worker.latest())
                if self._worker.error:
                    self.stop_worker()
            if self.paused:
                self.effective_time_scale = 0.
        elif self.sim.active_solver == "cuda-exact":
            if self._worker:
                self.stop_worker()
            if self._frame_controller is None and not self.paused:
                from cuda_controller import CudaFrameController
                self._frame_controller = CudaFrameController(self.sim.bodies,self.sim.G,
                    self.sim.softening,self.sim.background,use_graphs=self.use_graphs)
            count = self.physics_clock.advance_batches(
                self._frame_controller.batch if self._frame_controller else lambda n,dt: None,
                elapsed,self.time_scale,paused=self.paused)
            if count:
                pos,vel=self._frame_controller.snapshot()
                self.render_positions=pos
                for body,p,v in zip(self.sim.bodies,pos,vel):
                    body.x,body.y=map(float,p)
                    body.vx,body.vy=map(float,v)
                self._render_token+=1
                changed=True
                self.worker_stats=self._frame_controller
            self.effective_time_scale=self.physics_clock.effective_speed
        else:
            self.stop_worker()
            if not self.paused:
                self.sim.begin_device_frame()
            try:
                count = self.physics_clock.advance(self.sim.step, elapsed, self.time_scale, paused=self.paused)
            finally:
                self.sim.end_device_frame()
            self.effective_time_scale = self.physics_clock.effective_speed
            changed = count > 0
            self.render_positions = None
            if changed:
                self._render_token += 1
        if changed:
            self.diagnostic_points = []
            # OpenGL trails use GPU image history; avoid 5000 Python trail queues.
            if self.renderer is None:
                self.ensure_trails()
                for trail, body in zip(self.trails, self.sim.bodies):
                    trail.append((body.x, body.y))

    def draw_grid(self) -> None:
        w, h = self.screen.get_size()
        spacing_world = 100.0
        spacing_px = spacing_world * self.zoom
        if spacing_px < 35:
            spacing_world *= math.ceil(35 / max(spacing_px, 0.001))

        left, top = self.screen_to_world(0, 0)
        right, bottom = self.screen_to_world(w, h)
        start_x = math.floor(left / spacing_world) * spacing_world
        start_y = math.floor(top / spacing_world) * spacing_world

        x = start_x
        while x <= right:
            sx, _ = self.world_to_screen(x, 0)
            pygame.draw.line(self.screen, GRID, (sx, 0), (sx, h), 1)
            x += spacing_world

        y = start_y
        while y <= bottom:
            _, sy = self.world_to_screen(0, y)
            pygame.draw.line(self.screen, GRID, (0, sy), (w, sy), 1)
            y += spacing_world

    def transformed_trail_points(self, body_index: int) -> list[tuple[int, int]]:
        trail = self.trails[body_index]
        ref_index = self.reference_index()

        if ref_index is None:
            return [self.world_to_screen(x, y) for x, y in trail]

        ref_trail = self.trails[ref_index]
        n = min(len(trail), len(ref_trail))
        if n < 2:
            return []

        body_points = list(trail)[-n:]
        ref_points = list(ref_trail)[-n:]
        w, h = self.screen.get_size()
        points: list[tuple[int, int]] = []
        for (x, y), (rx, ry) in zip(body_points, ref_points):
            sx = (x - rx) * self.zoom + w / 2
            sy = (y - ry) * self.zoom + h / 2
            points.append((int(sx), int(sy)))
        return points

    def draw_menu(self) -> None:
        self.screen.fill((0,0,0,0) if self.renderer else BACKGROUND)
        w,h=self.screen.get_size()
        margin=max(24,min(70,w//20))
        card_w=min(475,w-2*margin)
        # A translucent scrim preserves contrast while leaving the galaxy visible.
        if self.renderer:
            for x in range(0,min(w,760),8):
                alpha=int(235*(1-min(1,x/760))**.5)
                pygame.draw.rect(self.screen,(7,11,22,alpha),(x,0,8,h))
        def label(value,xy,font,color):
            self.screen.blit(font.render(value,True,color),xy)
        label("G R A V I T Y   /   L A B",(margin,40),self.small_font,(137,172,196))
        label("Observatory",(margin,78),self.big_font,TEXT)
        label("Explore the patterns gravity leaves behind.",(margin,126),self.small_font,MUTED)
        top=184
        card_h=min(92,max(60,(h-top-90)//4-12))
        mouse=pygame.mouse.get_pos()
        self.menu_rects={}
        descriptions={"solar":"Nine bodies. A familiar celestial clock.",
                      "binary":"Two suns and a shared gravitational dance.",
                      "random":"A new rotating system with every visit.",
                      "galaxy":f"{self.galaxy_particles:,} stars. One evolving stellar disk."}
        for i,scenario in enumerate(SCENARIOS):
            rect=pygame.Rect(margin,top+i*(card_h+12),card_w,card_h)
            self.menu_rects[scenario.key]=rect
            hover=rect.collidepoint(mouse)
            pygame.draw.rect(self.screen,(23,33,52,245) if hover else (13,22,37,225),rect,border_radius=12)
            pygame.draw.rect(self.screen,(75,128,159) if hover else (37,53,76),rect,1,border_radius=12)
            label(f"0{i+1}",(rect.x+17,rect.y+18),self.font,(147,177,201))
            label(scenario.title,(rect.x+62,rect.y+13),self.title_font,TEXT)
            if card_h>=72:
                label(descriptions[scenario.key],(rect.x+62,rect.y+49),self.small_font,MUTED)
        label("SELECT A SYSTEM   /   CLICK OR PRESS 1—4",(margin,h-42),self.small_font,(117,143,169))
        if w>1050:
            if self.gpu_profile:
                p=self.gpu_profile
                label(f"AUTO  /  {p.particles:,} STARS  /  {p.total_vram_gib:.0f} GiB VRAM",(w-430,h-42),self.small_font,(113,139,165))
            else:
                label("A QUIET GALAXY, WAITING TO MOVE",(w-405,h-42),self.small_font,(113,109,145))
        self.present()

    def draw_simulation(self) -> None:
        self.screen.fill((0, 0, 0, 0) if self.renderer else BACKGROUND)
        if self.current_scenario != "galaxy":
            self.draw_grid()
        elif self.show_glow and self.renderer is None:
            self.galaxy_lighting.draw_bulge(self.screen,self.world_to_screen(0,0),self.zoom)

        if self.show_tree:
            from diagnostics import tree_boxes
            for x0, y0, x1, y1, depth in tree_boxes(self.sim):
                a, b = self.world_to_screen(x0, y0), self.world_to_screen(x1, y1)
                pygame.draw.rect(self.screen, (35+depth*20, 75, 120),
                                 pygame.Rect(a[0], a[1], max(1, b[0]-a[0]), max(1, b[1]-a[1])), 1)
        for x, y, ax, ay, error in self.diagnostic_points:
            start = self.world_to_screen(x, y)
            norm = math.hypot(ax, ay)
            if norm:
                end = (start[0]+24*ax/norm, start[1]+24*ay/norm)
                color = (90, 230, 150) if error < .01 else ((255, 190, 80) if error < .05 else (255, 80, 80))
                pygame.draw.line(self.screen, color, start, end, 2)
                angle = math.atan2(ay, ax)
                for delta in (-.5, .5):
                    tip = (end[0]-6*math.cos(angle+delta), end[1]-6*math.sin(angle+delta))
                    pygame.draw.line(self.screen, color, end, tip, 2)

        if self.show_trails and self.renderer is None:
            for i, body in enumerate(self.sim.bodies):
                points = self.transformed_trail_points(i)
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, body.color, False, points, 1)

        if self.renderer is None:
            for body in self.sim.bodies:
                sx, sy = self.world_to_screen(body.x, body.y)
                if self.current_scenario == "galaxy" and self.show_glow:
                    self.galaxy_lighting.draw_star(self.screen,body,(sx,sy),self.zoom)
                radius = max(1, round(body.radius * min(self.zoom, 2.2)))
                pygame.draw.circle(self.screen, body.color, (sx, sy), radius)
                if body is self.reference_body:
                    pygame.draw.circle(self.screen, LOCK_COLOR, (sx, sy), radius + 5, 2)
        elif self.reference_body is not None:
            b = self.reference_body
            pygame.draw.circle(self.screen, LOCK_COLOR, self.world_to_screen(b.x,b.y), 10, 1)

        if self.show_hud:
            self.draw_hud()
        self.present()

    def draw_hud(self):
        w, h = self.screen.get_size()
        def text(value, xy, color=MUTED, font=None):
            self.screen.blit((font or self.small_font).render(value, True, color), xy)
        pygame.draw.rect(self.screen, (10, 15, 29, 225), (16,16,min(440,w-32),96), border_radius=12)
        text("O B S E R V A T O R Y", (30,26), (220,225,248), self.font)
        mode = self.renderer.mode.upper() if self.renderer else "SOFTWARE"
        text(f"{len(self.sim.bodies):,} particles   {mode}   {self.clock.get_fps():.0f} FPS", (30,54))
        text(f"{self.sim.active_solver}   dt {self.physics_clock.timestep:g}", (30,78), (142,180,227))
        state = "PAUSED" if self.paused else f"{self.effective_time_scale:.1f}x / {self.time_scale:g}x"
        state_surface = self.font.render(state, True, (225,198,144))
        status_x = max(24,w-state_surface.get_width()-28)
        status_y = 28 if w>=900 else 122
        self.screen.blit(state_surface, (status_x,status_y))
        steps_s=0. if self.paused else self.effective_time_scale*2.7/self.physics_clock.timestep
        if w>=900:
            text(f"{steps_s:,.0f} steps/s",(w-180,54),(151,170,192))
        if self.worker_stats:
            a=self.worker_stats
            text(f"batch {a.batch_size} / {a.batch_ms:.2f} ms   copy {a.transfer_ms:.2f} ms", (max(24,w-345),78 if w>=900 else 151))
        if self.renderer and self.renderer.mode=="density":
            text("MASS DENSITY  /  LOG SCALE  /  BLUE → GOLD", (30,124 if w>=900 else 177), (217,173,235))
        if self.reference_body:
            text(f"FRAME / {self.reference_body.name}", (30,146), LOCK_COLOR)
        elif self.reference_pick_armed:
            text("Click a particle to follow it", (30,146), LOCK_COLOR)
        text("TAB help   F1 clean view   V density   T trails   SPACE pause", (24,h-52))
        text(self.diagnostic_text, (24,h-29), (113,135,165))
        if self.show_help:
            lines=[
                "CONTROLS",
                "+/- speed | PgUp/PgDn timestep | 0 reset timestep",
                "S/B solver | U enable CUDA | G glow | T trails",
                "V cinematic/density | comma/period exposure",
                "H spatial tree | A pause + sample force errors",
                "Wheel zoom | middle-drag pan | F follow particle",
                "Left add | right remove | C color | [ ] body size",
                f"New body: mass {self.spawn_mass:.2f}, radius {self.spawn_radius:g}",
                "R reset | M scenario menu | Esc quit",
                "OpenGL trails: fading screen-space exposure",
            ]
            y=180 if w>=900 else 210
            pygame.draw.rect(self.screen, (11,17,32,240), (16,y,min(620,w-32),len(lines)*25+24),border_radius=12)
            for i,line in enumerate(lines):
                if y+12+i*25<h-65:
                    text(line,(30,y+12+i*25), TEXT if i==0 else MUTED)

    def present(self):
        if self.renderer:
            if self.mode=="menu":
                if self._menu_preview is None:
                    bs=spiral_galaxy(star_count=1800)
                    self._menu_preview=(bs,np.array([(b.x,b.y) for b in bs],dtype='f4'))
                w,h=self.screen.get_size()
                zoom=min(w,h)*.54/850
                view=self.renderer.mode
                self.renderer.mode="cinematic"
                self.renderer.render(self.screen,*self._menu_preview,
                    (-w*.22/zoom,0.),zoom,galaxy=True,paused=True)
                self.renderer.mode=view
                pygame.display.flip()
                return
            positions=self.render_positions
            if positions is None:
                positions=np.asarray([(b.x,b.y) for b in self.sim.bodies],dtype='f4').reshape(-1,2)
            rx,ry=self.reference_origin()
            self.renderer.render(self.screen,self.sim.bodies,positions,
                (rx+self.camera_x,ry+self.camera_y),self.zoom,
                trails=self.show_trails,glow=self.show_glow,galaxy=self.current_scenario=="galaxy",
                token=self._render_token,paused=self.paused,menu=self.mode=="menu")
        pygame.display.flip()

    def draw(self) -> None:
        if self.mode == "menu":
            self.draw_menu()
        else:
            self.draw_simulation()

    def run(self) -> None:
        running = True
        last_tick = time.perf_counter()
        try:
            while running:
                now = time.perf_counter()
                elapsed = now-last_tick
                last_tick = now
                for event in pygame.event.get():
                    running = self.handle_event(event)
                    if not running:
                        break
                if not running:
                    break
                self.update(elapsed)
                render_start = time.perf_counter()
                self.draw()
                if self.renderer:
                    self.renderer.ctx.finish()
                render_cost = time.perf_counter()-render_start
                previous_render = getattr(self,"_render_cost",render_cost)
                self._render_cost = .85*previous_render+.15*render_cost
                self.physics_clock.budget_seconds = max(.001,min(.012,1/60-self._render_cost-.002))
                self.clock.tick(60)
        finally:
            self.stop_worker()
            if self.renderer:
                self.renderer.release()
            pygame.quit()


def main() -> int:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--solver", choices=("fmm", "exact", "barnes-hut", "auto", "cuda-exact"), default="fmm")
        parser.add_argument("--galaxy-particles", default="auto", help="auto benchmarks the GPU; or supply an integer >=2")
        parser.add_argument("--renderer", choices=("auto", "opengl", "software"), default="auto")
        parser.add_argument("--threaded", action="store_true", help="Experimental concurrent CUDA worker; default is coordinated frame pacing")
        parser.add_argument("--no-threading", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--no-graphs", action="store_true")
        args = parser.parse_args()
        if args.galaxy_particles!="auto":
            try:
                args.galaxy_particles=int(args.galaxy_particles)
            except ValueError:
                parser.error("--galaxy-particles must be auto or an integer")
            if args.galaxy_particles<2:
                parser.error("--galaxy-particles must be at least 2")
        profile=None
        if args.solver == "cuda-exact":
            from cuda_gravity import cuda_available, verify_device
            if not cuda_available():
                raise RuntimeError("CUDA requested but no NVIDIA CUDA device is available")
            print(f"CUDA startup checks passed: {verify_device(use_graphs=not args.no_graphs)}")
        if args.galaxy_particles=="auto":
            if args.solver=="cuda-exact":
                from gpu_profile import calibrate
                print("Calibrating a starting population on your GPU...")
                profile=calibrate(use_graphs=not args.no_graphs)
                args.galaxy_particles=profile.particles
                print(f"{profile.device}: {profile.total_vram_gib:.1f} GiB VRAM, "
                      f"{profile.free_vram_gib:.1f} GiB free; selected {profile.particles:,} particles "
                      f"at {profile.measured_step_ms:.3f} ms/step (physics only).")
            else:
                args.galaxy_particles=1200
        app=GravityApp(args.solver, args.galaxy_particles, renderer=args.renderer,
                      threaded=args.threaded and not args.no_threading, use_graphs=not args.no_graphs)
        app.gpu_profile=profile
        app.run()
        return 0
    except (pygame.error, ImportError, RuntimeError) as exc:
        print(f"Simulator could not start: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
