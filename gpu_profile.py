"""Hardware-measured starting population for the exact CUDA solver.

VRAM is a safety ceiling, never a throughput estimate. A small warm calibration
fits t(N)=a+b*N², then checks the chosen N on the actual device. This calibrates
physics, not end-to-end FPS; OpenGL work and desktop load still matter.
"""
from dataclasses import dataclass, asdict
import math
import statistics
import time


@dataclass(frozen=True)
class GPUProfile:
    device: str
    total_vram_gib: float
    free_vram_gib: float
    particles: int
    measured_step_ms: float
    target_step_ms: float
    samples: list


def choose_count(samples, free_bytes, *, target_ms=.8, maximum=50000):
    if not samples or not math.isfinite(target_ms) or target_ms<=0:
        raise ValueError('Finite positive target and measured samples required')
    # 256 MiB for renderer/driver headroom plus a deliberately generous 128 B
    # per body. Actual persistent solver arrays are 28 B/body plus tile padding.
    memory_limit=max(0,int((free_bytes*.35-256*1024**2)/128))
    cap=min(maximum,memory_limit)
    if cap<512:
        raise RuntimeError('Insufficient free GPU memory for automatic configuration')
    points=sorted((int(n),float(t)) for n,t in samples)
    if any(n<1 or not math.isfinite(t) or t<=0 for n,t in points):
        raise ValueError('Invalid calibration measurements')
    n0,t0=points[0]
    n1,t1=points[-1]
    b=max(0.,(t1-t0)/max(1,n1*n1-n0*n0))
    a=max(0.,t0-b*n0*n0)
    if b==0 or target_ms<=a:
        # No reliable quadratic slope: avoid optimistic extrapolation.
        estimate=n1*math.sqrt(target_ms/t1)
    else:
        estimate=math.sqrt((target_ms-a)/b)
    # Never extrapolate more than 4x the largest measured population.
    return max(512,min(cap,n1*4,int(estimate/256)*256))


def calibrate(device='cuda:0', *, target_ms=.8, use_graphs=True):
    import numpy as np
    import warp as wp
    from cuda_gravity import DeviceState
    from gravity_sim import Body, PlummerBackground
    wp.init()
    gpu=wp.get_device(device)
    if not gpu.is_cuda:
        raise RuntimeError('Automatic GPU population selection requires CUDA')
    free,total=gpu.free_memory,gpu.total_memory
    rng=np.random.default_rng(901)
    def measure(n):
        positions=rng.normal(size=(n,2))*200
        bodies=[Body(float(p[0]),float(p[1]),0,0,1200/n) for p in positions]
        state=DeviceState(bodies,1.,12.,PlummerBackground(((1800.,65.),(18000.,450.))),device=device)
        state.step_batch(.5,1,use_graphs=use_graphs)
        state.synchronize()
        timings=[]
        for _ in range(3):
            before=time.perf_counter()
            state.step_batch(.5,1,use_graphs=use_graphs)
            state.synchronize()
            timings.append((time.perf_counter()-before)*1000)
        return statistics.median(timings)
    # Free-memory cap is checked before allocating pilot arrays.
    choose_count([(4096,1.)],free,target_ms=1.)  # validate the memory floor
    limit=min(8192,max(512,int((free*.35-256*1024**2)/128)))
    pilots=sorted(set((min(2048,limit),min(8192,limit))))
    samples=[(n,measure(n)) for n in pilots]
    count=choose_count(samples,free,target_ms=target_ms)
    measured=measure(count)
    if measured>target_ms*1.2 and count>512:
        count=max(512,int(count*math.sqrt(target_ms/measured)/256)*256)
        measured=measure(count)
    return GPUProfile(gpu.name,total/1024**3,free/1024**3,count,measured,target_ms,samples)


if __name__=='__main__':
    import json
    print(json.dumps(asdict(calibrate()),indent=2))
