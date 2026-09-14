"""OpenGL 3.3 renderer. All methods must run on the context-owning main thread.

Particles are instanced quads. Density is additive Gaussian mass splatting;
cinematic trails are screen-space persistence, not physical trajectories.
No CUDA/OpenGL shared resource lifetime or stream assumptions are required.
"""
import math
import time
import numpy as np
import moderngl as gl

QUAD_VERTEX = '''#version 330
in vec2 in_vertex;
out vec2 uv;
void main(){ uv=in_vertex*.5+.5; gl_Position=vec4(in_vertex,0,1); }
'''
STAR_VERTEX = '''#version 330
in vec2 corner;
in vec2 position;
in vec3 color;
in float radius;
in float mass;
uniform vec2 viewport;
uniform vec2 origin;
uniform float zoom;
uniform float splat_radius;
uniform int density;
out vec2 local;
out vec3 tint;
out float weight;
void main(){
    float r = density==1 ? splat_radius : clamp((5.+radius*3.)*sqrt(zoom),3.,45.);
    vec2 pixel=(position-origin)*zoom+viewport*.5+corner*r;
    gl_Position=vec4(pixel.x/viewport.x*2.-1., 1.-pixel.y/viewport.y*2.,0,1);
    local=corner; tint=color; weight=mass;
}
'''
STAR_FRAGMENT = '''#version 330
in vec2 local;
in vec3 tint;
in float weight;
uniform int density;
uniform int glow;
out vec4 frag;
void main(){
    float r2=dot(local,local);
    if(r2>1.) discard;
    if(density==1){
        frag=vec4(vec3(weight*exp(-4.*r2)),1.);
    } else {
        float core=exp(-150.*r2);
        float halo=glow==1 ? .17*exp(-7.*r2) : 0.;
        frag=vec4(tint*(core*1.6+halo),1.);
    }
}
'''
ACCUM_FRAGMENT = '''#version 330
in vec2 uv;
uniform sampler2D current_tex;
uniform sampler2D previous_tex;
uniform float persistence;
uniform float injection;
out vec4 frag;
void main(){frag=vec4(texture(previous_tex,uv).rgb*persistence+texture(current_tex,uv).rgb*injection,1.);}
'''
BLUR_FRAGMENT = '''#version 330
in vec2 uv;
uniform sampler2D source;
uniform vec2 direction;
out vec4 frag;
void main(){
 vec3 c=texture(source,uv).rgb*.227027;
 c+=(texture(source,uv+direction*1.384615).rgb+texture(source,uv-direction*1.384615).rgb)*.316216;
 c+=(texture(source,uv+direction*3.230769).rgb+texture(source,uv-direction*3.230769).rgb)*.070270;
 frag=vec4(c,1.);
}
'''
COMPOSITE_FRAGMENT = '''#version 330
in vec2 uv;
uniform sampler2D scene;
uniform sampler2D bloom;
uniform sampler2D overlay;
uniform float exposure;
uniform float bloom_strength;
uniform int density;
uniform int galaxy;
uniform vec2 viewport;
uniform vec2 origin;
uniform float zoom;
out vec4 frag;
vec3 palette(float x){
 vec3 a=vec3(.02,.06,.19),b=vec3(.15,.28,.85),c=vec3(.7,.18,.58),d=vec3(1.,.81,.44);
 if(x<.33)return mix(a,b,x/.33);
 if(x<.7)return mix(b,c,(x-.33)/.37);
 return mix(c,d,clamp((x-.7)/.3,0.,1.));
}
void main(){
 vec3 radiance=texture(scene,uv).rgb;
 vec3 color;
 if(density==1){
    float value=1.-exp(-radiance.r*exposure*.3);
    color=mix(vec3(.009,.013,.033),palette(value),smoothstep(0.,.15,value));
 } else {
    vec2 world=(vec2(uv.x,1.-uv.y)*viewport-viewport*.5)/zoom+origin;
    float r=length(world);
    vec3 bulge=galaxy==1 ? vec3(.28,.16,.20)*pow(1.+r*r/(65.*65.),-2.) : vec3(0.);
    radiance+=bloom_strength*texture(bloom,uv).rgb+bulge;
    color=vec3(.008,.011,.025)+1.-exp(-radiance*exposure);
 }
 float vignette=1.-.28*dot(uv-.5,uv-.5);
 color*=vignette;
 vec4 hud=texture(overlay,vec2(uv.x,1.-uv.y));
 frag=vec4(mix(color,hud.rgb,hud.a),1.);
}
'''
OVERLAY_FRAGMENT = '''#version 330
in vec2 uv;
uniform sampler2D overlay;
out vec4 frag;
void main(){ frag=texture(overlay,vec2(uv.x,1.-uv.y)); }
'''


class GalaxyRenderer:
    def __init__(self, context=None):
        self.ctx=context or gl.create_context(require=330)
        self.mode='cinematic'
        self.exposure=1.5
        self._resources=[]
        self._size=None
        self._camera=None
        self._last_time=time.perf_counter()
        self._history=False
        self._last_token=None
        self.quad=self.ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[-1,1],[1,-1],[1,1]],dtype='f4').tobytes())
        self.stars=self.ctx.program(vertex_shader=STAR_VERTEX,fragment_shader=STAR_FRAGMENT)
        self.accum=self.ctx.program(vertex_shader=QUAD_VERTEX,fragment_shader=ACCUM_FRAGMENT)
        self.blur=self.ctx.program(vertex_shader=QUAD_VERTEX,fragment_shader=BLUR_FRAGMENT)
        self.composite=self.ctx.program(vertex_shader=QUAD_VERTEX,fragment_shader=COMPOSITE_FRAGMENT)
        self.menu=self.ctx.program(vertex_shader=QUAD_VERTEX,fragment_shader=OVERLAY_FRAGMENT)
        self.full={p:self.ctx.vertex_array(p,[(self.quad,'2f','in_vertex')]) for p in (self.accum,self.blur,self.composite,self.menu)}
        self.vbo=self.ctx.buffer(reserve=28)
        self.vao=self.ctx.vertex_array(self.stars,[(self.quad,'2f','corner'),(self.vbo,'2f 3f 1f 1f /i','position','color','radius','mass')])
        self._capacity=1
        self._metadata_key=None
        self._data=np.zeros((0,7),dtype='f4')

    def resize(self,size):
        if size==self._size:return
        for obj in reversed(self._resources):obj.release()
        self._resources=[]
        self._size=size
        w,h=size
        def target(size):
            texture=self.ctx.texture(size,4,dtype='f2')
            texture.filter=(gl.LINEAR,gl.LINEAR)
            texture.repeat_x=texture.repeat_y=False
            fbo=self.ctx.framebuffer([texture])
            self._resources.extend([texture,fbo])
            fbo.clear()
            return texture,fbo
        self.scene=target(size)
        self.histories=[target(size),target(size)]
        self.blurs=[target((max(1,w//2),max(1,h//2))) for _ in range(2)]
        self.hud=self.ctx.texture(size,4,dtype='f1')
        self.hud.filter=(gl.LINEAR,gl.LINEAR)
        self._resources.append(self.hud)
        self.reset_history()

    def reset_history(self):
        self._history=False
        self._last_token=None

    def metadata(self,bodies):
        # Body objects survive snapshots. Structural edits trigger a metadata rebuild.
        key=tuple(id(b) for b in bodies)
        if key!=self._metadata_key:
            self._metadata_key=key
            self._data=np.zeros((len(bodies),7),dtype='f4')
            if bodies:
                self._data[:,2:5]=np.array([b.color for b in bodies],dtype='f4')/255
                self._data[:,5]=[b.radius for b in bodies]
                m=np.array([b.mass for b in bodies],dtype='f4')
                self._data[:,6]=m/max(float(m.mean()),1e-20)
            self.reset_history()

    def render(self, surface, bodies, positions, origin, zoom, *, trails=False,
               glow=True, galaxy=False, token=None, paused=False, menu=False, target=None):
        import pygame
        size=surface.get_size()
        self.resize(size)
        target=target or self.ctx.screen
        self.ctx.disable(gl.DEPTH_TEST)
        self.ctx.disable(gl.BLEND)
        self.hud.write(pygame.image.tobytes(surface,'RGBA',False))
        if menu:
            target.use()
            self.ctx.viewport=(0,0,*size)
            self.hud.use(0)
            self.menu['overlay']=0
            self.full[self.menu].render()
            self.reset_history()
            return
        self.metadata(bodies)
        camera=(tuple(origin),zoom,self.mode,trails,glow)
        if camera!=self._camera:
            self.reset_history()
            self._camera=camera
        now=time.perf_counter()
        elapsed=min(.1,max(0.,now-self._last_time))
        self._last_time=now
        self.scene[1].use()
        self.ctx.viewport=(0,0,*size)
        self.scene[1].clear()
        self.ctx.enable(gl.BLEND)
        self.ctx.blend_func=(gl.ONE,gl.ONE)
        if len(bodies):
            self._data[:,:2]=positions
            if len(bodies)>self._capacity:
                self._capacity=2**math.ceil(math.log2(len(bodies)))
                self.vbo.orphan(self._capacity*28)
            self.vbo.write(self._data.tobytes())
            self.stars['viewport']=size
            self.stars['origin']=origin
            self.stars['zoom']=zoom
            self.stars['density']=int(self.mode=='density')
            self.stars['glow']=int(glow)
            self.stars['splat_radius']=min(45.,max(10.,math.sqrt(size[0]*size[1]/len(bodies))*1.7))
            self.vao.render(vertices=6,instances=len(bodies))
        self.ctx.disable(gl.BLEND)
        active=self.scene
        if trails and self.mode=='cinematic':
            previous,next_=self.histories
            next_[1].use()
            self.scene[0].use(0)
            previous[0].use(1)
            self.accum['current_tex']=0
            self.accum['previous_tex']=1
            decay=math.exp(-elapsed/0.30) if self._history else 0.
            # EMA gives stable brightness, independent of snapshot/render frequency.
            # Frozen frames retain exactly the previous image.
            if paused and self._history:
                decay=1.
            self.accum['persistence']=decay
            self.accum['injection']=1.-decay
            self.full[self.accum].render()
            self.histories.reverse()
            active=next_
            self._history=True
        self._last_token=token
        for i,direction in enumerate(((1./self.blurs[0][0].width,0.),(0.,1./self.blurs[1][0].height))):
            self.blurs[i][1].use()
            self.ctx.viewport=(0,0,*self.blurs[i][0].size)
            (active[0] if i==0 else self.blurs[0][0]).use(0)
            self.blur['source']=0
            self.blur['direction']=direction
            self.full[self.blur].render()
        target.use()
        self.ctx.viewport=(0,0,*size)
        active[0].use(0)
        self.blurs[1][0].use(1)
        self.hud.use(2)
        for name,value in dict(scene=0,bloom=1,overlay=2,exposure=self.exposure,
                               bloom_strength=.8 if glow else 0.,density=int(self.mode=='density'),
                               galaxy=int(galaxy and glow),viewport=size,origin=origin,zoom=zoom).items():
            self.composite[name]=value
        self.full[self.composite].render()

    def release(self):
        for obj in reversed(self._resources):obj.release()
        self.vao.release()
        self.vbo.release()
        for vao in self.full.values():vao.release()
        for p in (self.stars,self.accum,self.blur,self.composite,self.menu):p.release()
        self.quad.release()
