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
uniform float star_scale;
uniform int jelly;
uniform int density;
out vec2 local;
out vec3 tint;
out float weight;
void main(){
    float r = density==1 ? splat_radius : clamp((5.+radius*3.)*sqrt(zoom)*star_scale*(jelly==1 ? 1.8 : 1.),3.,90.);
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
uniform int jelly;
out vec4 frag;
void main(){
    float r2=dot(local,local);
    if(r2>1.) discard;
    if(density==1){
        frag=vec4(vec3(weight*exp(-4.*r2)),1.);
    } else {
        float core=exp(-150.*r2);
        float halo=glow==1 ? .17*exp(-7.*r2) : 0.;
        if(jelly==1){
            // Additive translucent sphere impostor: soft interior, iridescent rim,
            // and an off-centre specular highlight. No sorting or extra geometry.
            float r=sqrt(r2);
            float edge=1.-smoothstep(.84,1.,r);
            float rim=exp(-pow((r-.68)/.13,2.));
            float highlight=exp(-90.*dot(local-vec2(-.23,-.27),local-vec2(-.23,-.27)));
            vec3 pearl=mix(tint,vec3(.48,.72,1.),.35+.25*local.x);
            frag=vec4((pearl*(.075*sqrt(1.-r2)+.18*rim)+vec3(.8,.9,1.)*highlight*.5+ tint*(core*.35+halo*.4))*edge,1.);
        } else frag=vec4(tint*(core*1.6+halo),1.);
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
uniform sampler2D density_tex;
uniform float exposure;
uniform float bloom_strength;
uniform int density;
uniform int galaxy;
uniform int hybrid;
uniform int mist;
uniform int flares;
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
    float value=clamp(log(1.+radiance.r*exposure*.35)/3.6,0.,1.);
    color=mix(vec3(.009,.013,.033),palette(value),smoothstep(0.,.08,value));
    float iso=abs(fract(value*14.+.5)-.5);
    float line=1.-smoothstep(.015,.045,iso);
    color+=line*.035*smoothstep(.1,.3,value);
 } else {
    vec2 world=(vec2(uv.x,1.-uv.y)*viewport-viewport*.5)/zoom+origin;
    float r=length(world);
    vec3 bulge=galaxy==1 ? vec3(.28,.16,.20)*pow(1.+r*r/(65.*65.),-2.) : vec3(0.);
    float d=texture(density_tex,uv).r;
    vec3 unresolved=vec3(.07,.027,.105)*log(1.+d*.20);
    radiance+=bloom_strength*texture(bloom,uv).rgb+bulge+unresolved*bloom_strength;
    if(hybrid==1){
        float heat=clamp(log(1.+d*exposure*.35)/3.6,0.,1.);
        // Same log palette as density mode, combined in HDR before tone mapping.
        radiance+=palette(heat)*smoothstep(0.,.12,heat)*(.16+heat*1.1);
    }
    if(mist==1){
        vec2 offset=vec2(16.)/viewport;
        float haze=(texture(density_tex,uv+offset).r+texture(density_tex,uv-offset).r
                   +texture(density_tex,uv+vec2(offset.x,-offset.y)).r
                   +texture(density_tex,uv+vec2(-offset.x,offset.y)).r)*.25;
        float structure=.75+.25*sin(world.x*.013+sin(world.y*.009));
        radiance+=vec3(.10,.055,.19)*log(1.+haze*.25)*structure;
    }
    if(flares==1){
        // Screen-space anamorphic streaks from bright bloom; six extra taps.
        vec3 streak=vec3(0.);
        for(int i=1;i<=3;i++){
            vec2 offset=vec2(float(i)*12./viewport.x,0.);
            streak+=(max(texture(bloom,uv+offset).rgb-vec3(.22),vec3(0.))
                    +max(texture(bloom,uv-offset).rgb-vec3(.22),vec3(0.)))/float(i);
        }
        radiance+=streak*vec3(.20,.32,.50)*bloom_strength;
    }
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
        self.mode='hybrid'
        self.jelly=False
        self.star_scale=1.0
        self.mist=False
        self.flares=False
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
        self.density_field=target(size)
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
                self._data[:,6]=m/max(float(m.sum()),1e-20)*5000.
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
        camera=(zoom,self.mode,trails,glow,self.jelly,self.star_scale)
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
            self.stars['jelly']=int(self.jelly)
            self.stars['star_scale']=self.star_scale
            self.stars['splat_radius']=min(45.,max(10.,math.sqrt(size[0]*size[1]/len(bodies))*1.7))
            self.vao.render(vertices=6,instances=len(bodies))
            if self.mode!='density':
                self.density_field[1].use()
                self.density_field[1].clear()
                self.stars['density']=1
                self.stars['splat_radius']=min(55.,max(20.,math.sqrt(size[0]*size[1]/len(bodies))*2.5))
                self.vao.render(vertices=6,instances=len(bodies))
        if not len(bodies):
            self.density_field[1].clear()
        self.ctx.disable(gl.BLEND)
        active=self.scene
        if trails and self.mode!='density' and paused and self._history:
            active=self.histories[0]
        elif trails and self.mode!='density':
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
        self.density_field[0].use(3)
        for name,value in dict(scene=0,bloom=1,overlay=2,density_tex=3,exposure=self.exposure,
                               bloom_strength=.8 if glow else 0.,density=int(self.mode=='density'),
                               galaxy=int(galaxy and glow),hybrid=int(self.mode=='hybrid'),
                               mist=int(self.mist),flares=int(self.flares),viewport=size,origin=origin,zoom=zoom).items():
            self.composite[name]=value
        self.full[self.composite].render()

    def release(self):
        for obj in reversed(self._resources):obj.release()
        self.vao.release()
        self.vbo.release()
        for vao in self.full.values():vao.release()
        for p in (self.stars,self.accum,self.blur,self.composite,self.menu):p.release()
        self.quad.release()
