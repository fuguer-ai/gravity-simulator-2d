import time
import unittest
import numpy as np
from gravity_sim import Body
from physics_worker import PhysicsWorker
from simulation_clock import FixedStepClock
from gpu_profile import choose_count


class LinearState:
    instances=[]
    def __init__(self,bodies,*args,**kw):
        self.n=len(bodies)
        self.pos=np.array([(b.x,b.y) for b in bodies],dtype='f4').reshape(-1,2)
        self.vel=np.ones_like(self.pos)
        self.batches=[]
        LinearState.instances.append(self)
    def force(self):pass
    def synchronize(self):pass
    def snapshot(self):return self.pos.copy(),self.vel.copy()
    def step_batch(self,dt,count,**kw):
        self.batches.append((dt,count))
        self.pos+=dt*count


def until(predicate):
    deadline=time.perf_counter()+2
    while time.perf_counter()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.002)
    raise AssertionError('Worker did not reach expected state')


class WorkerTests(unittest.TestCase):
    def test_final_state_immutable_mailbox_and_controls(self):
        worker=PhysicsWorker([Body(0,0,0,0,1)],1,1,dt=.01,speed=32,state_factory=LinearState)
        try:
            first=until(lambda: (s:=worker.latest()) and s.simulated_time>.02 and s)
            self.assertFalse(first.positions.flags.writeable)
            old=first.positions.copy()
            worker.controls(.02,8,True)
            until(lambda: (s:=worker.latest()) and s.effective_speed==0 and s.sequence>first.sequence)
            frozen=worker.latest().simulated_time
            time.sleep(.02)
            self.assertEqual(worker.latest().simulated_time,frozen)
            worker.controls(.005,32,False)
            until(lambda: worker.latest().simulated_time>frozen)
            final=worker.close()
            self.assertIsNone(worker.error)
            self.assertAlmostEqual(float(final.positions[0,0]),final.simulated_time,places=4)
            np.testing.assert_array_equal(first.positions,old)
            self.assertIs(worker.latest(),final)
            self.assertEqual(len(LinearState.instances[-1].batches)>0,True)
        finally:worker.close()

    def test_initialization_error_is_reported(self):
        def fail(*a,**kw):raise RuntimeError('expected failure')
        worker=PhysicsWorker([],1,1,state_factory=fail)
        try:
            until(lambda: worker.error)
            self.assertIn('expected failure',worker.error)
        finally:worker.close()

    def test_empty_state_stops(self):
        worker=PhysicsWorker([],1,1,state_factory=LinearState)
        until(lambda: worker.latest())
        self.assertEqual(worker.close().simulated_time,0)


class BatchClockTests(unittest.TestCase):
    def test_fixed_time_and_budget(self):
        clock=FixedStepClock(timestep=.5,budget_seconds=.004)
        wall=[0.]
        calls=[]
        def batch(count,dt):
            calls.append((count,dt))
            wall[0]+=count*.001
        steps=clock.advance_batches(batch,.1,100,clock=lambda:wall[0])
        self.assertGreater(steps,0)
        self.assertLessEqual(steps,5)
        self.assertEqual(steps,sum(n for n,_ in calls))
        self.assertTrue(all(dt==.5 for _,dt in calls))
        self.assertLess(clock.pending,clock.timestep)
        clock.advance_batches(batch,.1,100,paused=True,clock=lambda:wall[0])
        self.assertEqual(clock.pending,0.)

    def test_no_steps_when_not_due(self):
        clock=FixedStepClock(timestep=.5)
        self.assertEqual(clock.advance_batches(lambda *a:self.fail('unexpected batch'),.001,1),0)


class ProfileTests(unittest.TestCase):
    def test_quadratic_selection_and_memory_guard(self):
        samples=[(2048,.1),(8192,1.6)]
        count=choose_count(samples,16*1024**3,target_ms=.8)
        self.assertGreater(count,5000)
        self.assertLess(count,8192)
        # VRAM beyond the safety cap cannot buy extra arithmetic throughput.
        self.assertEqual(count,choose_count(samples,24*1024**3,target_ms=.8))
        with self.assertRaises(RuntimeError):choose_count(samples,128*1024**2)

    def test_no_optimistic_extrapolation_on_noisy_samples(self):
        count=choose_count([(2048,.4),(8192,.3)],16*1024**3)
        self.assertLessEqual(count,8192*4)
        self.assertGreaterEqual(count,512)
