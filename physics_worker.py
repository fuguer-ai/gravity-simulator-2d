"""Single-owner CUDA worker with a bounded latest-snapshot mailbox.

Pygame/OpenGL always stay on the main thread. Only this thread touches its
DeviceState. Commands are coalesced controls; structural edits stop/join, consume
the final snapshot, then construct a new worker. No unbounded task/frame queue.
"""
from dataclasses import dataclass
import copy
import math
import threading
import time


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    positions: object
    velocities: object
    simulated_time: float
    effective_speed: float
    batch_ms: float
    batch_size: int
    transfer_ms: float


class PhysicsWorker:
    def __init__(self, bodies, G, softening, background=None, *, dt=.5, speed=1.,
                 device='cuda:0', use_graphs=True, state_factory=None):
        self._bodies = copy.deepcopy(bodies)
        self._args = (G, softening, background)
        self.device, self.use_graphs = device, use_graphs
        self._factory = state_factory
        self._condition = threading.Condition()
        self._controls = (dt, speed, False)
        self._revision = 0
        self._stopping = False
        self._snapshot = None
        self.error = None
        self._thread = threading.Thread(target=self._run, name='gravity-cuda', daemon=True)
        self._thread.start()

    def controls(self, dt, speed, paused=False):
        if not math.isfinite(dt+speed) or dt <= 0 or speed < 0:
            raise ValueError('Positive finite dt and nonnegative finite speed required')
        with self._condition:
            value = (dt, speed, bool(paused))
            if value != self._controls:
                self._controls = value
                self._revision += 1
                self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._snapshot

    def close(self, timeout=10.):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError('Physics worker did not stop; state must not be edited yet')
        return self.latest()

    def _run(self):
        try:
            if self._factory is None:
                from cuda_gravity import DeviceState
                factory = DeviceState
            else:
                factory = self._factory
            state = factory(self._bodies, *self._args, device=self.device)
            # Compile/initialize before starting the pacing clock; no startup catch-up.
            state.force()
            state.synchronize()
            sequence = 0
            simulated = speed_estimate = last_batch_ms = transfer_ms = 0.
            batch_size = 1
            per_step = .001

            def publish():
                nonlocal sequence, transfer_ms
                start = time.perf_counter()
                pos, vel = state.snapshot()
                transfer_ms = (time.perf_counter()-start)*1000
                pos.setflags(write=False)
                vel.setflags(write=False)
                sequence += 1
                snapshot = Snapshot(sequence, pos, vel, simulated, speed_estimate,
                                    last_batch_ms, batch_size, transfer_ms)
                with self._condition:
                    self._snapshot = snapshot

            publish()
            previous = sample_start = last_publish = time.perf_counter()
            advanced = pending = 0.
            revision = -1
            while True:
                with self._condition:
                    if self._stopping:
                        break
                    dt, speed, paused = self._controls
                    new_revision = self._revision
                now = time.perf_counter()
                elapsed = min(now-previous, .1)
                previous = now
                if new_revision != revision:
                    revision = new_revision
                    pending = advanced = 0.
                    sample_start = now
                    elapsed = 0.
                if paused or speed == 0 or not state.n:
                    speed_estimate = 0.
                    publish()
                    with self._condition:
                        # Predicate prevents a lost notification between publish and wait.
                        if not self._stopping and self._revision == revision:
                            self._condition.wait(.1)
                    previous = time.perf_counter()
                    continue
                pending += elapsed*2.7*speed
                due = int((pending+1e-12)/dt)
                if due == 0:
                    with self._condition:
                        if not self._stopping and self._revision == revision:
                            self._condition.wait(min(.01, max(.0002, (dt-pending)/(2.7*speed))))
                    continue
                # Bound a batch to ~4 ms based on measured prior cost, max 32 steps.
                # Powers of two keep CUDA graph cache small. A single step may exceed
                # that budget at large N, so it is not a hard real-time guarantee.
                limit = max(1, min(32, int(.004/max(per_step, 1e-6))))
                batch_size = 2**int(math.log2(min(due, limit)))
                start = time.perf_counter()
                state.step_batch(dt, batch_size, use_graphs=self.use_graphs)
                state.synchronize()
                end = time.perf_counter()
                last_batch_ms = (end-start)*1000
                per_step = .8*per_step + .2*(end-start)/batch_size
                increment = dt*batch_size
                simulated += increment
                advanced += increment
                # Discard excess backlog under overload instead of catching up later.
                pending = min(max(0., pending-increment), dt*limit)
                if end-sample_start >= .25:
                    speed_estimate = advanced/(2.7*(end-sample_start))
                    advanced = 0.
                    sample_start = end
                if end-last_publish >= 1/60:
                    publish()
                    last_publish = time.perf_counter()
            publish()  # Final completed batch must not be lost during edits/reset.
        except Exception as exc:
            with self._condition:
                self.error = f'{type(exc).__name__}: {exc}'
                self._condition.notify_all()
