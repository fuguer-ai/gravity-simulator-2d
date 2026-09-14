"""Persistent CUDA state for the coordinated physics-then-display loop."""
import time


class CudaFrameController:
    def __init__(self, bodies, G, softening, background=None, *, use_graphs=True,
                 device='cuda:0', state_factory=None):
        if state_factory is None:
            from cuda_gravity import DeviceState
            state_factory=DeviceState
        self.state=state_factory(bodies,G,softening,background,device=device)
        self.use_graphs=use_graphs
        self.batch_ms=0.
        self.batch_size=0
        self.transfer_ms=0.
        self.steps=0

    def batch(self,count,dt):
        before=time.perf_counter()
        self.state.step_batch(dt,count,use_graphs=self.use_graphs)
        self.state.synchronize()
        self.batch_ms=(time.perf_counter()-before)*1000
        self.batch_size=count
        self.steps+=count

    def snapshot(self):
        before=time.perf_counter()
        pos,vel=self.state.snapshot()
        self.transfer_ms=(time.perf_counter()-before)*1000
        return pos,vel
