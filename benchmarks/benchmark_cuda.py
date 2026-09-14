"""Synchronized CUDA timings and sampled force error; no invented GPU estimates."""
import argparse
import json
import sys
import time
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cuda_gravity import DeviceState, verify_device
from gravity_sim import NBodySimulation
from galaxy import spiral_galaxy, BACKGROUND, SOFTENING


def timed(call, sync, repeats):
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter()-start)*1000)
    return statistics.median(samples)


def sampled_exact(bs, indices):
    pos = np.asarray([(b.x,b.y) for b in bs], dtype=np.float64)
    masses = np.asarray([b.mass for b in bs], dtype=np.float64)
    d = pos[None]-pos[indices, None]
    r2 = (d*d).sum(-1)+SOFTENING**2
    r2[np.arange(len(indices)), indices] = np.inf
    return (d*(masses[None]*r2**-1.5)[..., None]).sum(1) + np.asarray([
        BACKGROUND.acceleration(*pos[i]) for i in indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[1200, 5000, 10000])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', default='cuda:0', help='cpu is validation only, not GPU performance')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare-cpu', action='store_true')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1:
        parser.error('Sizes >= 2 and positive repeats required')
    name = verify_device(args.device)
    rows = []
    for n in args.sizes:
        bs = spiral_galaxy(star_count=n)
        state = DeviceState(bs, 1, SOFTENING, BACKGROUND, device=args.device)
        indices = np.linspace(0, n-1, min(n, 64), dtype=int)
        expected = sampled_exact(bs, indices)
        actual = state.force().numpy()[indices]
        rms = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-30))
        if not np.isfinite(rms) or rms > 1e-4:
            raise RuntimeError(f'Force accuracy gate failed at N={n}: {rms}')
        force_ms = timed(state.force, state.synchronize, args.repeats)
        # Warm up integration separately; then measure resident steps.
        state.step(.5)
        step_ms = timed(lambda: state.step(.5), state.synchronize, args.repeats)
        transfer_ms = timed(lambda: state.download(bs), state.synchronize, args.repeats)
        # Compare the old frame pattern (upload + per-step barriers + download)
        # with resident graph batches at identical N, dt and eight steps/frame.
        def old_frame():
            state.upload(bs)
            for _ in range(8):
                state.step(.5)
                state.synchronize()
            state.download(bs)
        old_frame()
        old_frame_ms=timed(old_frame,state.synchronize,args.repeats)
        state.step_batch(.5,8)
        state.synchronize()
        def graph_frame():
            state.step_batch(.5,8)
            state.synchronize()
            state.download(bs)
        graph_frame_ms=timed(graph_frame,state.synchronize,args.repeats)
        batch_ms=timed(lambda:state.step_batch(.5,8),state.synchronize,args.repeats)
        row = dict(n=n, force_ms=force_ms, resident_step_ms=step_ms,
                   download_ms=transfer_ms, sampled_force_relative_rms=rms,
                   state_bytes=state.size*7*4, batch8_ms=batch_ms,
                   batch_step_ms=batch_ms/8, old_frame8_ms=old_frame_ms,
                   coordinated_frame8_ms=graph_frame_ms,
                   measured_frame_speedup=old_frame_ms/graph_frame_ms)
        if args.compare_cpu:
            # Use original seeded state for an apples-to-apples force comparison.
            initial = spiral_galaxy(star_count=n)
            for mode in ('fmm', 'barnes-hut'):
                sim = NBodySimulation(initial, solver=mode, softening=SOFTENING, background=BACKGROUND)
                approx = np.asarray(sim.accelerations())[indices]
                row[mode+'_relative_rms'] = float(np.linalg.norm(approx-expected)/np.linalg.norm(expected))
                row[mode+'_force_ms'] = timed(sim.accelerations, lambda: None, args.repeats)
        rows.append(row)
        print(json.dumps(row))
    result = dict(device=name, backend=args.device, precision='float32',
                  rendering_included=False, warmup_excluded=True, rows=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    main()
