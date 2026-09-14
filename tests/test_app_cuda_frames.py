import importlib.util
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import unittest
from unittest.mock import patch
from gravity_sim import Body

HAS_WARP=importlib.util.find_spec('warp') is not None


@unittest.skipUnless(HAS_WARP,'Optional Warp not installed')
class AppFrameTests(unittest.TestCase):
    def test_default_is_coordinated_and_edits_invalidate_state(self):
        import pygame
        from gravity_app import GravityApp
        from cuda_controller import CudaFrameController
        from cuda_gravity import DeviceState
        def factory(*args,**kwargs):
            return CudaFrameController(*args,**kwargs,device='cpu')
        app=GravityApp(solver='cuda-exact',renderer='software',galaxy_particles=16)
        self.assertFalse(app.threaded)
        app.draw_menu()
        app.handle_menu_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=app.menu_rects['galaxy'].center))
        self.assertEqual(app.current_scenario,'galaxy')
        app.time_scale=64
        try:
            with patch('cuda_controller.CudaFrameController',side_effect=factory):
                app.update(.02)
                state=app._frame_controller
                self.assertIsNotNone(state)
                app.update(.02)
                self.assertIs(app._frame_controller,state)
                self.assertIsNone(app._worker)
                app.add_body((400,300))
                self.assertIsNone(app._frame_controller)
                app.update(.02)
                self.assertEqual(app._frame_controller.state.n,17)
                app.handle_simulation_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_SPACE))
                before=[(b.x,b.y) for b in app.sim.bodies]
                app.update(.2)
                self.assertEqual(before,[(b.x,b.y) for b in app.sim.bodies])
                app.handle_simulation_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_s))
                self.assertEqual(app.sim.solver,'auto')
                self.assertEqual(app.sim.active_solver,'exact')
                app.draw_simulation()
        finally:
            app.stop_worker();pygame.quit()
