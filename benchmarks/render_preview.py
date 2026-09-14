"""Headless visual QA of the actual application HUD and OpenGL renderer."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pygame
import numpy as np
import moderngl
from PIL import Image
from gravity_app import GravityApp
from gpu_render import GalaxyRenderer
from galaxy import spiral_galaxy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--particles',type=int,default=5000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    app=GravityApp(renderer='software',galaxy_particles=args.particles)
    app.load_scenario('galaxy')
    app.paused=True
    ctx=moderngl.create_standalone_context(backend='egl')
    renderer=GalaxyRenderer(ctx)
    app.renderer=renderer
    fbo=ctx.simple_framebuffer(app.screen.get_size())
    def present():
        bs=app.sim.bodies
        pos=np.array([(b.x,b.y) for b in bs],dtype='f4')
        origin=app.reference_origin()
        if app.mode=='menu':
            bs=spiral_galaxy(star_count=1800)
            pos=np.array([(b.x,b.y) for b in bs],dtype='f4')
            w,h=app.screen.get_size()
            zoom=min(w,h)*.54/850
            renderer.render(app.screen,bs,pos,(-w*.22/zoom,0.),zoom,galaxy=True,paused=True,target=fbo)
        else:
            renderer.render(app.screen,bs,pos,origin,app.zoom,galaxy=True,
                            trails=app.show_trails,paused=app.paused,target=fbo)
    app.present=present
    for name in ('cinematic','density','menu','help'):
        app.mode='menu' if name=='menu' else 'simulation'
        app.show_help=name=='help'
        renderer.mode='density' if name=='density' else 'cinematic'
        app.draw()
        Image.frombytes('RGB',fbo.size,fbo.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(args.output/(name+'.png'))
    renderer.release()
    fbo.release()
    ctx.release()
    pygame.quit()
    print(f'Visual previews: {args.output}')


if __name__=='__main__':main()
