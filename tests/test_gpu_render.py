import importlib.util
import unittest
import numpy as np
from gravity_sim import Body

HAS_GL=importlib.util.find_spec('moderngl') is not None and importlib.util.find_spec('pygame') is not None


@unittest.skipUnless(HAS_GL,'Optional OpenGL dependencies not installed')
class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import moderngl
        try:cls.ctx=moderngl.create_standalone_context(backend='egl')
        except Exception as exc:raise unittest.SkipTest(f'Headless EGL unavailable: {exc}')
    @classmethod
    def tearDownClass(cls):cls.ctx.release()
    def setUp(self):
        import pygame
        from gpu_render import GalaxyRenderer
        self.renderer=GalaxyRenderer(self.ctx)
        self.surface=pygame.Surface((256,192),pygame.SRCALPHA)
        self.fbo=self.ctx.simple_framebuffer((256,192))
        self.bs=[Body(0,0,0,0,1,color=(255,160,70)),Body(35,10,0,0,2,color=(90,140,255))]
        self.pos=np.array([(b.x,b.y) for b in self.bs],dtype='f4')
    def tearDown(self):self.renderer.release();self.fbo.release()
    def draw(self,**kw):
        self.renderer.render(self.surface,self.bs,self.pos,(0.,0.),1.,target=self.fbo,**kw)
        return np.frombuffer(self.fbo.read(components=3),dtype='u1').reshape(192,256,3)[::-1].copy()
    def test_modes_and_overlay_orientation(self):
        self.surface.fill((0,0,0,0))
        self.surface.fill((0,255,0,255),(2,3,9,8))
        a=self.draw()
        np.testing.assert_array_equal(a[5,5],[0,255,0])
        self.assertGreater(int(a[96,128].sum()),100)
        self.renderer.mode='density'
        b=self.draw()
        self.assertGreater(np.abs(a.astype(float)-b).sum(),1000)
    def test_paused_history_is_stable(self):
        self.draw(trails=True)
        a=self.draw(trails=True,paused=True)
        b=self.draw(trails=True,paused=True)
        np.testing.assert_array_equal(a,b)
        self.renderer.reset_history()
        self.pos[:,0]+=20
        c=self.draw(trails=True,paused=True)
        self.assertGreater(np.abs(c.astype(float)-b).sum(),1000)
    def test_empty_and_resize(self):
        self.bs=[];self.pos=np.empty((0,2),dtype='f4')
        self.draw()
        self.renderer.resize((300,200))
        self.assertEqual(self.renderer._size,(300,200))
        self.assertFalse(self.renderer._history)
