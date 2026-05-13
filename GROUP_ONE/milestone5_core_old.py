import math,sys,os,time
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0,os.path.dirname(__file__))
from terrain_core import Vector3,Terrain

# Particle type constants
PTYPE_LEAF = 0
PTYPE_DUST = 1
PTYPE_EMBER = 2

class KeyframeAnimator:
    def __init__(self,width=128,depth=128,scale=6.0,height_scale=2.8,octaves=7,persistence=0.5,lacunarity=2.0):
        self.W=width;self.D=depth;self.scale=scale
        self.height_scale=height_scale;self.octaves=octaves
        self.persistence=persistence;self.lacunarity=lacunarity
        self.keyframes=[];self._built=False
    def add_keyframe(self,t,seed):
        terrain=Terrain(width=self.W,depth=self.D,scale=self.scale,
                        height_scale=self.height_scale,octaves=self.octaves,
                        persistence=self.persistence,lacunarity=self.lacunarity,seed=seed)
        hg=np.array(terrain.height_grid())
        self.keyframes.append((float(t),hg));self._built=False
        print(f"    KF t={t} seed={seed} h=[{hg.min():.3f},{hg.max():.3f}]")
    def build_splines(self):
        times=np.array([k[0] for k in self.keyframes])
        grids=np.array([k[1] for k in self.keyframes])
        flat=grids.reshape(len(self.keyframes),self.D*self.W)
        self.splines=CubicSpline(times,flat,axis=0,bc_type='not-a-knot')
        self._built=True;self.t_min=times[0];self.t_max=times[-1]
    def get_frame(self,t):
        if not self._built:self.build_splines()
        return self.splines(np.clip(t,self.t_min,self.t_max)).reshape(self.D,self.W)
    def get_velocity(self,t):
        if not self._built:self.build_splines()
        return self.splines(np.clip(t,self.t_min,self.t_max),1).reshape(self.D,self.W)
    def duration(self): return self.t_max-self.t_min if self._built else 0.0
    def motion_analysis(self,n_steps=50):
        ts=np.linspace(self.t_min,self.t_max,n_steps);results=[]
        prev=self.get_frame(ts[0])
        for t in ts[1:]:
            curr=self.get_frame(t);vel=self.get_velocity(t);delta=curr-prev
            results.append({'t':round(float(t),3),'mean_height':round(float(curr.mean()),4),
                'max_velocity':round(float(np.abs(vel).max()),4),
                'rms_change':round(float(np.sqrt((delta**2).mean())),4)})
            prev=curr
        return results

class ParticleSystem:
    TYPES={'leaves':{'gravity':-1.5,'drag':0.92,'wind_scale':1.0,'life_range':(4.0,8.0),
                     'size_range':(0.015,0.035),'colour':lambda r:np.array([r.uniform(0.55,0.75),r.uniform(0.35,0.55),r.uniform(0.05,0.20)])},
           'dust':  {'gravity':-0.3,'drag':0.97,'wind_scale':1.5,'life_range':(2.0,5.0),
                     'size_range':(0.008,0.020),'colour':lambda r:np.array([r.uniform(0.72,0.88),r.uniform(0.62,0.78),r.uniform(0.40,0.55)])},
           'embers':{'gravity':-0.8,'drag':0.94,'wind_scale':0.8,'life_range':(1.5,3.5),
                     'size_range':(0.005,0.015),'colour':lambda r:np.array([r.uniform(0.9,1.0),r.uniform(0.3,0.6),0.0])}}
    def __init__(self,max_particles=2000,seed=42):
        self.max_n=max_particles;self.rng=np.random.default_rng(seed);self.n=0;self.ptype='leaves'
        self.pos=np.zeros((max_particles,3));self.vel=np.zeros((max_particles,3))
        self.life=np.zeros(max_particles);self.age=np.zeros(max_particles)
        self.size=np.zeros(max_particles);self.colour=np.zeros((max_particles,3))
        self.wind_dir=np.array([1.0,0.0,0.3]);self.wind_speed=1.5;self.time=0.0
    def set_type(self,t):
        if t in self.TYPES:self.ptype=t
    def emit(self,n_new,emitter_pos,emitter_radius=0.5):
        props=self.TYPES[self.ptype]
        for _ in range(min(n_new,self.max_n-self.n)):
            i=self.n;angle=self.rng.uniform(0,2*math.pi);r=self.rng.uniform(0,emitter_radius)
            self.pos[i]=[emitter_pos[0]+r*math.cos(angle),emitter_pos[1]+self.rng.uniform(0.05,0.2),emitter_pos[2]+r*math.sin(angle)]
            self.vel[i]=[self.rng.uniform(-0.3,0.3)+self.wind_dir[0]*self.wind_speed*0.3,self.rng.uniform(0.2,0.8),self.rng.uniform(-0.3,0.3)+self.wind_dir[2]*self.wind_speed*0.3]
            lmin,lmax=props['life_range'];smin,smax=props['size_range']
            self.life[i]=self.rng.uniform(lmin,lmax);self.age[i]=0.0
            self.size[i]=self.rng.uniform(smin,smax);self.colour[i]=props['colour'](self.rng)
            self.n+=1
    def update(self,dt,terrain_hgrid,terrain_bounds):
        if self.n==0:return
        self.time+=dt;props=self.TYPES[self.ptype]
        gust=math.sin(self.time*0.7)*0.5+math.sin(self.time*1.3)*0.3
        wind_now=self.wind_dir*(self.wind_speed+gust)
        accel=np.array([0,props['gravity'],0])+wind_now*props['wind_scale']*0.8
        self.vel[:self.n]=self.vel[:self.n]*props['drag']+accel*dt
        self.pos[:self.n]+=self.vel[:self.n]*dt
        D2,W2=terrain_hgrid.shape;xmin,xmax,zmin,zmax=terrain_bounds
        for i in range(self.n):
            x,y,z=self.pos[i]
            gx=int((x-xmin)/(xmax-xmin)*(W2-1));gz=int((z-zmin)/(zmax-zmin)*(D2-1))
            if 0<=gx<W2 and 0<=gz<D2:
                gh=terrain_hgrid[gz,gx]
                if y<gh+0.01:
                    self.pos[i,1]=gh+0.01;self.vel[i,1]=abs(self.vel[i,1])*0.15
                    self.vel[i,0]*=0.6;self.vel[i,2]*=0.6
        self.age[:self.n]+=dt;mask=self.age[:self.n]<self.life[:self.n];na=int(mask.sum())
        if na<self.n:
            for arr in[self.pos,self.vel,self.life,self.age,self.size,self.colour]:arr[:na]=arr[:self.n][mask]
            self.n=na
    def get_points(self):return self.pos[:self.n].copy()
    def get_colours(self):
        if self.n==0:return np.zeros((0,3))
        lf=self.age[:self.n]/self.life[:self.n];fade=(1.0-lf[:,None])**1.5
        return np.clip(self.colour[:self.n]*fade,0,1)
    def stats(self):return{'active':self.n,'wind_speed':round(self.wind_speed,3),'time':round(self.time,2)}

class WaveSimulator:
    def __init__(self,grid_size=64,wave_speed=1.8,damping=0.995,dx=0.1):
        self.N=grid_size;self.c=wave_speed;self.damp=damping;self.dx=dx
        self.dt=0.9*dx/(wave_speed*math.sqrt(2))
        self.h_cur=np.zeros((grid_size,grid_size));self.h_prev=np.zeros((grid_size,grid_size))
        self.r2=(wave_speed*self.dt/dx)**2;self.time=0.0;self.step_count=0;self._wind_phase=0.0
    def stability_check(self):
        r=self.c*self.dt/self.dx
        return{'r':round(r,4),'stable':r<=1/math.sqrt(2),'dt':round(self.dt,5),'dx':self.dx,'c':self.c}
    def add_drop(self,x,z,amplitude=0.08,radius=3):
        i0=int(np.clip(z*self.N,0,self.N-1));j0=int(np.clip(x*self.N,0,self.N-1))
        for di in range(-radius,radius+1):
            for dj in range(-radius,radius+1):
                i=(i0+di)%self.N;j=(j0+dj)%self.N
                d=math.sqrt(di**2+dj**2);self.h_cur[i,j]+=amplitude*math.exp(-d**2/4.0)
    def add_wind_ripples(self,wx=1.0,wz=0.3):
        self._wind_phase+=0.05
        XI,ZI=np.meshgrid(np.arange(self.N),np.arange(self.N))
        self.h_cur+=(np.sin((XI*wx+ZI*wz)*0.3+self._wind_phase))*0.004
    def step(self,n_steps=1):
        for _ in range(n_steps):
            lap=(np.roll(self.h_cur,1,0)+np.roll(self.h_cur,-1,0)+np.roll(self.h_cur,1,1)+np.roll(self.h_cur,-1,1)-4*self.h_cur)
            h_next=2*self.h_cur-self.h_prev+self.r2*lap
            h_next*=self.damp;self.h_prev=self.h_cur.copy();self.h_cur=h_next
            self.time+=self.dt;self.step_count+=1
    def get_surface(self,base=-0.12):return self.h_cur+base
    def get_energy(self):
        k=np.sum((self.h_cur-self.h_prev)**2)/(2*self.dt**2);p=np.sum(self.h_cur**2)/2
        return float(k+p)

class TerrainDeformer:
    def __init__(self,base_hgrid):
        self.base=base_hgrid.copy();self.deform=np.zeros_like(base_hgrid)
        self.D,self.W=base_hgrid.shape;self.active_ripples=[];self.time=0.0
    def apply_deformation(self,gx,gz,strength=0.3,radius=8):
        for j in range(max(0,gz-radius*2),min(self.D,gz+radius*2)):
            for i in range(max(0,gx-radius*2),min(self.W,gx+radius*2)):
                d=math.sqrt((i-gx)**2+(j-gz)**2)
                if d<radius*2:self.deform[j,i]+=strength*math.exp(-d**2/(2*(radius*0.6)**2))
        self.active_ripples.append({'cx':gx,'cz':gz,'time':0.0,'speed':8.0,
                                     'amp':abs(strength)*0.3,'decay':0.85,'wave_k':0.8})
    def update(self,dt):
        self.time+=dt;ripple_grid=np.zeros((self.D,self.W));still=[]
        for rip in self.active_ripples:
            rip['time']+=dt;t=rip['time'];cx,cz=rip['cx'],rip['cz']
            radius=rip['speed']*t;amp=rip['amp']*(rip['decay']**t)
            if amp<0.001:continue
            band=int(radius)+5
            for j in range(max(0,cz-band),min(self.D,cz+band)):
                for i in range(max(0,cx-band),min(self.W,cx+band)):
                    d=math.sqrt((i-cx)**2+(j-cz)**2);df=d-radius
                    if abs(df)<8:ripple_grid[j,i]+=amp*math.cos(rip['wave_k']*df)*math.exp(-abs(df)*0.3)
            still.append(rip)
        self.active_ripples=still;self.current_grid=self.base+self.deform+ripple_grid
    def get_height_grid(self):return getattr(self,'current_grid',self.base+self.deform)
    def reset_deformations(self):self.deform[:]=0.0;self.active_ripples.clear()
    def deformation_stats(self):
        d=self.deform
        return{'max_raise':round(float(d.max()),4),'max_lower':round(float(d.min()),4),
               'rms_deform':round(float(np.sqrt((d**2).mean())),4),
               'active_ripples':len(self.active_ripples),'affected_area':int((np.abs(d)>0.001).sum())}

if __name__=="__main__":
    print("Milestone 5 core OK")
    t=Terrain(width=16,depth=16,scale=5.0,height_scale=2.5,octaves=4,seed=42)
    hg=np.array(t.height_grid())
    a=KeyframeAnimator(width=16,depth=16,octaves=4)
    a.add_keyframe(0,42);a.add_keyframe(5,123);a.build_splines()
    print("Keyframe:",a.get_frame(2.5).shape)
    ps=ParticleSystem(100);ps.emit(20,[0,1,0]);ps.update(0.1,hg,(-2,2,-2,2))
    print("Particles:",ps.n)
    ws=WaveSimulator(16);ws.add_drop(0.5,0.5,0.1);ws.step(5)
    print("Waves:",ws.get_surface().shape)
    d=TerrainDeformer(hg);d.apply_deformation(8,8,0.5,3);d.update(0.1)
    print("Deform:",d.deformation_stats())
    print("All OK")
