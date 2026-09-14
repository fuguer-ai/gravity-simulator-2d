"""Equal-duration timestep/accuracy comparison for the existing galaxy model.

No automatic timestep changes. Energy uses FP64 direct summation of the same
softened model including its static background. CPU mode is for small validation.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from cuda_gravity import DeviceState
from galaxy import spiral_galaxy, BACKGROUND, SOFTENING


def energy(pos,vel,mass):
    pos=np.asarray(pos,dtype='f8');vel=np.asarray(vel,dtype='f8');mass=np.asarray(mass,dtype='f8')
    total=float((.5*mass*(vel*vel).sum(-1)).sum())
    for i in range(0,len(pos),256):
        delta=pos[None]-pos[i:i+256,None]
        r2=(delta*delta).sum(-1)+SOFTENING**2
        r2[np.arange(len(delta)),np.arange(i,i+len(delta))]=np.inf
        total-=.5*float((mass[i:i+256,None]*mass[None]/np.sqrt(r2)).sum())
    for m,a in BACKGROUND.components:
        total-=float((mass*m/np.sqrt((pos*pos).sum(-1)+a*a)).sum())
    return total


def integrate(bodies,dt,duration,device):
    steps=round(duration/dt)
    if not math.isclose(steps*dt,duration,abs_tol=1e-10):
        raise ValueError('Duration must be divisible by every timestep')
    s=DeviceState(bodies,1,SOFTENING,BACKGROUND,device=device)
    start=time.perf_counter()
    while steps:
        count=min(16,steps)
        s.step_batch(dt,count)
        steps-=count
    s.synchronize()
    elapsed=time.perf_counter()-start
    return (*s.snapshot(),elapsed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--particles',type=int,default=512)
    p.add_argument('--duration',type=float,default=64.)
    p.add_argument('--timesteps',nargs='+',type=float,default=[.25,.5,1.,2.])
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.particles<2 or args.duration<=0 or min(args.timesteps)<=0 or not all(map(math.isfinite,[args.duration,*args.timesteps])):
        p.error('Finite positive duration/timesteps and >=2 particles required')
    bodies=spiral_galaxy(star_count=args.particles)
    masses=np.array([b.mass for b in bodies])
    e0=energy([(b.x,b.y) for b in bodies],[(b.vx,b.vy) for b in bodies],masses)
    ref_dt=min(args.timesteps)/2
    ref_pos,ref_vel,_=integrate(bodies,ref_dt,args.duration,args.device)
    rows=[]
    for dt in args.timesteps:
        pos,vel,seconds=integrate(bodies,dt,args.duration,args.device)
        rows.append(dict(dt=dt,seconds_including_capture=seconds,
            relative_energy_change=(energy(pos,vel,masses)-e0)/abs(e0),
            position_relative_rms=float(np.linalg.norm(pos-ref_pos)/np.linalg.norm(ref_pos)),
            velocity_relative_rms=float(np.linalg.norm(vel-ref_vel)/np.linalg.norm(ref_vel)),
            radial_quantiles=np.quantile(np.linalg.norm(pos,axis=1),[.1,.5,.9]).tolist()))
    result=dict(particles=args.particles,duration=args.duration,reference_dt=ref_dt,device=args.device,rows=rows)
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
