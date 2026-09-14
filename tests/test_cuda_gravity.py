import copy
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from gravity_sim import Body, NBodySimulation, PlummerBackground

HAS_WARP = importlib.util.find_spec('warp') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


def bodies(n):
    rng = np.random.default_rng(32)
    return [Body(*p, *v, m) for p, v, m in zip(rng.normal(size=(n, 2))*4,
            rng.normal(size=(n, 2))*.01, rng.uniform(0, 10, n))]


@unittest.skipUnless(HAS_WARP, 'Optional Warp not installed')
class DeviceTests(unittest.TestCase):
    device = 'cpu'

    def test_forces_and_tail_sizes(self):
        from cuda_gravity import DeviceState
        for n in (0, 1, 2, 63, 64, 65, 129):
            with self.subTest(n=n):
                bs = bodies(n)
                if n > 1:
                    bs[0].mass = 0  # Test particle still accelerates.
                    bs[1].x, bs[1].y = bs[0].x, bs[0].y
                bg = PlummerBackground(((4, 2), (3, 7)))
                ref = NBodySimulation(bs, softening=.3, background=bg, solver='exact').accelerations()
                state = DeviceState(bs, 1, .3, bg, device=self.device)
                actual = state.force().numpy()[:n]
                np.testing.assert_allclose(actual, np.asarray(ref).reshape(-1, 2), rtol=3e-5, atol=2e-5)

    def test_verlet_background_and_momentum(self):
        from cuda_gravity import DeviceState
        for bg in (None, PlummerBackground(((20, 3),))):
            bs = bodies(20)
            ref = NBodySimulation(copy.deepcopy(bs), softening=.5, background=bg, solver='exact')
            initial_momentum = np.array(ref.total_momentum())
            state = DeviceState(bs, 1, .5, bg, device=self.device)
            for _ in range(30):
                ref.step(.001)
                state.step(.001)
            state.download(bs)
            actual = np.array([(b.x, b.y, b.vx, b.vy) for b in bs])
            expected = np.array([(b.x, b.y, b.vx, b.vy) for b in ref.bodies])
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
            if bg is None:
                p = NBodySimulation(bs).total_momentum()
                np.testing.assert_allclose(p, initial_momentum, atol=2e-5)

    def test_host_edits_and_frames(self):
        from cuda_gravity import DeviceState
        def make(*a, **k):
            return DeviceState(*a, **k, device=self.device)
        bs = bodies(8)
        sim = NBodySimulation(bs, solver='cuda-exact')
        with patch('cuda_gravity.DeviceState', side_effect=make):
            sim.step(.01)
            bs[0].mass = 33
            bs.append(Body(9, 9, 0, 0, 1))
            ref = NBodySimulation(copy.deepcopy(bs), solver='exact')
            sim.begin_device_frame()
            for _ in range(4):
                sim.step(.001)
                ref.step(.001)
            sim.end_device_frame()
            np.testing.assert_allclose([(b.x,b.y) for b in bs], [(b.x,b.y) for b in ref.bodies], atol=2e-5)
            sim.solver_mode = 'fmm'
            sim.step(.001)
            sim.solver_mode = 'cuda-exact'
            sim.step(.001)

    def test_invalid_inputs(self):
        from cuda_gravity import DeviceState
        for eps in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                DeviceState(bodies(1), 1, eps, device=self.device)
        bs = bodies(1)
        bs[0].mass = -1
        with self.assertRaises(ValueError):
            DeviceState(bs, 1, 1, device=self.device)

    @unittest.skipUnless(HAS_TORCH, 'Optional Torch not installed')
    def test_torch_views_share_memory(self):
        from cuda_gravity import DeviceState
        s = DeviceState(bodies(3), 1, .2, device=self.device)
        views = s.torch_views()
        self.assertEqual(tuple(views['positions'].shape), (3, 2))
        self.assertEqual(views['positions'].data_ptr(), s.pos.ptr)


@unittest.skipUnless(HAS_TORCH, 'Optional Torch not installed')
class TorchTests(unittest.TestCase):
    def test_tiled_exact(self):
        import torch
        from torch_gravity import accelerations
        bs = bodies(35)
        ref = NBodySimulation(bs, solver='exact', softening=.2).accelerations()
        for dtype in (torch.float32, torch.float64):
            p = torch.tensor([(b.x,b.y) for b in bs], dtype=dtype)
            m = torch.tensor([b.mass for b in bs], dtype=dtype)
            for block in (1, 16, 64):
                actual = accelerations(p, m, softening=.2, block_size=block)
                np.testing.assert_allclose(actual.numpy(), ref, rtol=2e-5, atol=1e-5)


if HAS_WARP:
    from cuda_gravity import cuda_available
    if cuda_available():
        class CudaTests(DeviceTests):
            device = 'cuda:0'
    else:
        @unittest.skip('NVIDIA driver/device unavailable; actual CUDA execution NOT tested')
        class CudaTests(unittest.TestCase):
            def test_cuda_execution(self):
                pass
