import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math, sys, os, time

sys.path.insert(0, '/home/claude/sprint3')
from terrain_core    import Terrain
from milestone5_core import KeyframeAnimator, ParticleSystem, WaveSimulator, TerrainDeformer
from milestone6_core import AdaptiveFractalNoise, EcosystemSimulator

OUT5 = "/home/claude/sprint3/figs_m5"; os.makedirs(OUT5, exist_ok=True)
OUT6 = "/home/claude/sprint3/figs_m6"; os.makedirs(OUT6, exist_ok=True)

def save(fig, path, name):
    fig.savefig(f"{path}/{name}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.png")

terrain = Terrain(width=32, depth=32, scale=6.0, height_scale=2.8, octaves=6, seed=77)
hgrid   = np.array(terrain.height_grid())

# ── M5 FIG 1: Keyframe Animation ──────────────────────────────
print("M5 Fig 1 — Keyframe...")
anim = KeyframeAnimator(width=32, depth=32, scale=6.0, height_scale=2.8, octaves=5)
seeds=[77,42,123,777,77]; times=[0,4,8,12,16]
for t,s in zip(times,seeds): anim.add_keyframe(t,s)
anim.build_splines()

fig, axes = plt.subplots(2,4,figsize=(14,7))
fig.suptitle("Figure M5.1 — Keyframe Animation: Cubic Spline Terrain Morphing\n"
             "Person A — H(x,z,t) = CubicSpline([t₀…tₙ],[H₀…Hₙ])(t) — C² continuity",
             fontsize=11, fontweight='bold')
for idx,t in enumerate([0.0,4.0,8.0,12.0]):
    hf=anim.get_frame(t)
    im=axes[0][idx].imshow(hf,cmap='terrain',origin='lower',interpolation='bilinear')
    axes[0][idx].set_title(f"t={t:.0f}s  seed={seeds[idx]}",fontsize=9.5,fontweight='bold')
    axes[0][idx].axis('off'); plt.colorbar(im,ax=axes[0][idx],shrink=0.8)

ts_fine=np.linspace(0,16,200)
vh=[anim.get_frame(t)[16,16] for t in ts_fine]
vv=[anim.get_velocity(t)[16,16] for t in ts_fine]
axes[1][0].plot(ts_fine,vh,'b-',lw=2,label='Height h(t)')
ax2=axes[1][0].twinx(); ax2.plot(ts_fine,vv,'r--',lw=1.5,label='Velocity')
for t in times: axes[1][0].axvline(t,color='grey',lw=0.8,ls='--',alpha=0.5)
axes[1][0].set_title("Single vertex over time",fontsize=9,fontweight='bold')
axes[1][0].set_xlabel("Time (s)"); axes[1][0].legend(fontsize=7.5,loc='upper left')
ax2.legend(fontsize=7.5,loc='upper right'); axes[1][0].grid(True,alpha=0.3)

analysis=anim.motion_analysis(40)
ts_a=[r['t'] for r in analysis]; vels=[r['max_velocity'] for r in analysis]
rms=[r['rms_change'] for r in analysis]
axes[1][1].plot(ts_a,vels,'r-',lw=2,label='Max velocity')
axes[1][1].plot(ts_a,rms,'b-',lw=2,label='RMS change')
axes[1][1].set_title("Motion Analysis",fontsize=9,fontweight='bold')
axes[1][1].set_xlabel("Time (s)"); axes[1][1].legend(fontsize=8); axes[1][1].grid(True,alpha=0.3)

axes[1][2].plot(ts_fine,vh,'b-',lw=2,label='C² spline')
axes[1][2].scatter(times[:-1],[anim.get_frame(t)[16,16] for t in times[:-1]],s=80,color='red',zorder=5,label='Keyframes')
axes[1][2].set_title("Cubic Spline Continuity",fontsize=9,fontweight='bold')
axes[1][2].set_xlabel("Time (s)"); axes[1][2].legend(fontsize=8); axes[1][2].grid(True,alpha=0.3)

axes[1][3].axis('off')
axes[1][3].text(0.1,0.9,f"Keyframe System\n\nKeyframes : {len(seeds)}\nDuration  : {anim.duration():.0f}s\nResolution: 32×32\nContinuity: C²\n\nFormula:\nH(x,z,t)=\nCubicSpline\n([t₀..tₙ],[H₀..Hₙ])(t)",
    transform=axes[1][3].transAxes,fontsize=9,va='top',fontfamily='monospace',
    bbox=dict(boxstyle='round',fc='#f0f0f0'))
plt.tight_layout(); save(fig,OUT5,"m5_fig1_keyframe")

# ── M5 FIG 2: Particle System ──────────────────────────────────
print("M5 Fig 2 — Particles...")
fig,axes=plt.subplots(1,3,figsize=(13,5))
fig.suptitle("Figure M5.2 — Particle System: Wind + Gravity + Terrain Collision (Person B)\n"
             "Euler integration: dv/dt = gravity + wind·drag",fontsize=11,fontweight='bold')
bounds=(-2,2,-2,2)
for ci,ptype in enumerate(['leaves','dust','embers']):
    ps=ParticleSystem(400,seed=ci*7); ps.set_type(ptype)
    ps.emit(100,[0,1.5,0],0.8)
    for step in range(60):
        if step%8==0: ps.emit(10,[np.random.uniform(-0.5,0.5),1.5,np.random.uniform(-0.5,0.5)])
        ps.update(0.033,hgrid,bounds)
    ax=axes[ci]; ax.set_facecolor('#1a1a2e')
    pts=ps.get_points(); cols=ps.get_colours()
    if len(pts)>0: ax.scatter(pts[:,0],pts[:,2],c=cols,s=12,alpha=0.8)
    ax.set_title(f"{ptype.title()} ({ps.n} active)",fontsize=10,fontweight='bold',color='white')
    ax.set_xlabel("X",color='white'); ax.set_ylabel("Z",color='white')
    ax.tick_params(colors='white'); ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
    ax.grid(True,alpha=0.2,color='white')
    props=ps.TYPES[ptype]
    ax.text(0.05,0.95,f"g={props['gravity']}\ndrag={props['drag']}\nwind={props['wind_scale']}×",
            transform=ax.transAxes,fontsize=8,color='white',va='top',fontfamily='monospace')
plt.tight_layout(); save(fig,OUT5,"m5_fig2_particles")

# ── M5 FIG 3: Wave Simulation ─────────────────────────────────
print("M5 Fig 3 — Waves...")
wave=WaveSimulator(64,1.8,0.994,0.07)
wave.add_drop(0.5,0.5,0.12); wave.add_drop(0.3,0.7,0.08); wave.add_drop(0.7,0.3,0.09)
fig,axes=plt.subplots(2,4,figsize=(14,7))
fig.suptitle("Figure M5.3 — Wave Simulation: 2D Finite Difference (Person C)\n"
             "∂²h/∂t² = c²∇²h  |  CFL: r = c·dt/dx ≤ 1/√2",fontsize=11,fontweight='bold')
snaps=[]; energies=[]
for si in range(60):
    wave.add_wind_ripples(1.0,0.3); wave.step(2); energies.append(wave.get_energy())
    if si in [0,10,25,50]: snaps.append(wave.get_surface().copy())
for idx,(sn,title) in enumerate(zip(snaps,["Step 0","Step 20","Step 50","Step 100"])):
    im=axes[0][idx].imshow(sn,cmap='RdBu_r',vmin=-0.15,vmax=0.15,origin='lower',interpolation='bilinear')
    axes[0][idx].set_title(title,fontsize=9.5,fontweight='bold'); axes[0][idx].axis('off')
    plt.colorbar(im,ax=axes[0][idx],shrink=0.8)
axes[1][0].plot(range(len(energies)),energies,color='steelblue',lw=2)
axes[1][0].set_title("Energy Decay",fontsize=9,fontweight='bold')
axes[1][0].set_xlabel("Step"); axes[1][0].set_ylabel("Energy"); axes[1][0].grid(True,alpha=0.3)
stab=wave.stability_check()
r_vals=np.linspace(0,1.5,100); lim=1/math.sqrt(2)
axes[1][1].fill_between(r_vals[r_vals<=lim],0,1,alpha=0.3,color='green',label='Stable')
axes[1][1].fill_between(r_vals[r_vals>lim],0,1,alpha=0.3,color='red',label='Unstable')
axes[1][1].axvline(stab['r'],color='blue',lw=2.5,label=f"r={stab['r']} ✓")
axes[1][1].axvline(lim,color='black',lw=1.5,ls='--',label='CFL limit')
axes[1][1].set_title("CFL Stability",fontsize=9,fontweight='bold')
axes[1][1].set_xlabel("r = c·dt/dx"); axes[1][1].legend(fontsize=7.5); axes[1][1].grid(True,alpha=0.3)
surf=wave.get_surface()
axes[1][2].plot(surf[32,:],'b-',lw=2,label='Row 32')
axes[1][2].plot(surf[:,32],'r--',lw=2,label='Col 32')
axes[1][2].set_title("Wave Profile",fontsize=9,fontweight='bold')
axes[1][2].set_xlabel("Grid pos"); axes[1][2].legend(fontsize=8); axes[1][2].grid(True,alpha=0.3)
axes[1][3].axis('off')
axes[1][3].text(0.05,0.95,f"Wave Parameters\n\nc={stab['c']}\ndt={stab['dt']:.5f}s\ndx={stab['dx']}\nr={stab['r']} (stable)\ndamping=0.994\ngrid=64×64\n\nLaplacian:\n∇²h≈h[i+1]+h[i-1]\n   +h[j+1]+h[j-1]\n   -4·h[i,j]",
    transform=axes[1][3].transAxes,fontsize=8.5,va='top',fontfamily='monospace',
    bbox=dict(boxstyle='round',fc='#e8f4f8'))
plt.tight_layout(); save(fig,OUT5,"m5_fig3_waves")

# ── M5 FIG 4: Terrain Deformation ─────────────────────────────
print("M5 Fig 4 — Deformation...")
hg32=np.array(Terrain(width=32,depth=32,scale=5.0,height_scale=2.5,octaves=5,seed=42).height_grid())
deformer=TerrainDeformer(hg32)
deformer.apply_deformation(16,16,0.8,5); deformer.apply_deformation(8,24,-0.5,4)
snaps_d=[deformer.get_height_grid().copy()]
for _ in range(15): deformer.update(0.05)
snaps_d.append(deformer.get_height_grid().copy())
for _ in range(25): deformer.update(0.05)
snaps_d.append(deformer.get_height_grid().copy())
deformer.apply_deformation(24,8,0.6,3)
for _ in range(10): deformer.update(0.05)
snaps_d.append(deformer.get_height_grid().copy())
fig,axes=plt.subplots(2,4,figsize=(14,7))
fig.suptitle("Figure M5.4 — Terrain Deformation: Gaussian Bumps + Ripple Propagation (Person D)",fontsize=11,fontweight='bold')
for idx,(state,title) in enumerate(zip(snaps_d,["Initial","After 15 steps","After 40 steps","New deform"])):
    im=axes[0][idx].imshow(state,cmap='terrain',origin='lower',interpolation='bilinear')
    axes[0][idx].set_title(title,fontsize=9.5,fontweight='bold'); axes[0][idx].axis('off')
    plt.colorbar(im,ax=axes[0][idx],shrink=0.8)
diff=deformer.get_height_grid()-hg32
axes[1][0].imshow(diff,cmap='RdBu_r',origin='lower',vmin=-1,vmax=1); axes[1][0].set_title("Deformation field",fontsize=9,fontweight='bold'); axes[1][0].axis('off')
axes[1][1].plot(hg32[16,:],'b-',lw=2,label='Original'); axes[1][1].plot(deformer.get_height_grid()[16,:],'r-',lw=2,label='Deformed')
axes[1][1].fill_between(range(32),hg32[16,:],deformer.get_height_grid()[16,:],alpha=0.3,color='orange',label='Diff')
axes[1][1].set_title("Cross-section row 16",fontsize=9,fontweight='bold'); axes[1][1].legend(fontsize=8); axes[1][1].grid(True,alpha=0.3)
xv=np.linspace(-8,8,200)
for r,col in [(3,'steelblue'),(5,'firebrick'),(7,'green')]:
    axes[1][2].plot(xv,np.exp(-xv**2/(2*(r*0.6)**2)),color=col,lw=2,label=f'r={r}')
axes[1][2].set_title("Gaussian Profile\nstrength·exp(-d²/2σ²)",fontsize=9,fontweight='bold'); axes[1][2].legend(fontsize=8); axes[1][2].grid(True,alpha=0.3)
s=deformer.deformation_stats(); axes[1][3].axis('off')
axes[1][3].text(0.05,0.95,f"Stats\n\nMax raise: {s['max_raise']}\nMax lower: {s['max_lower']}\nRMS: {s['rms_deform']}\nRipples: {s['active_ripples']}\nAffected: {s['affected_area']}v\n\nRipple:\nh+=A·cos(k·d)\n   ·e^(-|d|·0.3)\ndecay=0.85/step",
    transform=axes[1][3].transAxes,fontsize=9,va='top',fontfamily='monospace',bbox=dict(boxstyle='round',fc='#f5f5f5'))
plt.tight_layout(); save(fig,OUT5,"m5_fig4_deformation")

# ── M6 FIG 1: AFN ─────────────────────────────────────────────
print("M6 Fig 1 — AFN...")
afn=AdaptiveFractalNoise(min_octaves=2,max_octaves=8,seed=42)
hg_afn,oct_grid,stats=afn.generate_adaptive(32,32,5.0,(4,4.5,7))
hg_unif=np.zeros((32,32))
for j in range(32):
    for i in range(32): hg_unif[j,i]=afn.evaluate_adaptive(i/32*5,j/32*5,8)*2.8
fig,axes=plt.subplots(2,4,figsize=(14,7))
fig.suptitle("Figure M6.1 — Novel 1: Adaptive Fractal Noise (AFN)\n"
             "Adjusts octave count per-vertex by curvature+distance → ~27-55% cost saving",fontsize=11,fontweight='bold')
axes[0][0].imshow(hg_afn,cmap='terrain',origin='lower',interpolation='bilinear'); axes[0][0].set_title("AFN Output",fontsize=9.5,fontweight='bold'); axes[0][0].axis('off')
im2=axes[0][1].imshow(oct_grid,cmap='hot',origin='lower',vmin=2,vmax=8); axes[0][1].set_title("Octave Map\n(bright=more)",fontsize=9.5,fontweight='bold'); axes[0][1].axis('off'); plt.colorbar(im2,ax=axes[0][1],shrink=0.8,label='Octaves')
axes[0][2].imshow(hg_unif,cmap='terrain',origin='lower',interpolation='bilinear'); axes[0][2].set_title("Uniform 8 Oct\n(reference)",fontsize=9.5,fontweight='bold'); axes[0][2].axis('off')
diff_afn=np.abs(hg_afn-hg_unif); im3=axes[0][3].imshow(diff_afn,cmap='Reds',origin='lower')
axes[0][3].set_title(f"Difference\nRMSE={np.sqrt((diff_afn**2).mean()):.4f}",fontsize=9.5,fontweight='bold'); axes[0][3].axis('off'); plt.colorbar(im3,ax=axes[0][3],shrink=0.8)
cam_configs=[(4,4.5,7,"Overview"),(1,1,2,"Close-up"),(0,10,0.1,"Aerial")]
cols_afn=['steelblue','firebrick','green']
for idx,(cx,cy,cz,name) in enumerate(cam_configs):
    _,og2,st2=afn.generate_adaptive(32,32,5.0,(cx,cy,cz))
    counts=[np.sum(og2==n) for n in range(2,9)]
    axes[1][idx].bar(range(2,9),counts,color=cols_afn[idx],alpha=0.85,edgecolor='white')
    axes[1][idx].set_title(f"{name}\nSaving: {st2['cost_saving']}%",fontsize=8.5,fontweight='bold')
    axes[1][idx].set_xlabel("Octaves"); axes[1][idx].set_ylabel("Vertices"); axes[1][idx].grid(axis='y',alpha=0.3)
methods=['Min\noct','AFN\nover','AFN\nclose','AFN\naerial','Max\noct']
st_over=afn.generate_adaptive(32,32,5.0,(4,4.5,7))[2]['total_evals']
st_close=afn.generate_adaptive(32,32,5.0,(1,1,2))[2]['total_evals']
st_air=afn.generate_adaptive(32,32,5.0,(0,10,0.1))[2]['total_evals']
evals_=[32*32*2,st_over,st_close,st_air,32*32*8]
bars=axes[1][3].bar(methods,evals_,color=['#2ecc71','#3498db','#e74c3c','#f39c12','#c0392b'],alpha=0.85,edgecolor='white')
for bar,v in zip(bars,evals_): axes[1][3].text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f'{v:,}',ha='center',fontsize=7.5,fontweight='bold')
axes[1][3].set_ylabel("Total evals"); axes[1][3].set_title("Cost Comparison",fontsize=9.5,fontweight='bold'); axes[1][3].grid(axis='y',alpha=0.3)
plt.tight_layout(); save(fig,OUT6,"m6_fig1_afn")

# ── M6 FIG 2: Ecosystem ───────────────────────────────────────
print("M6 Fig 2 — Ecosystem...")
eco=EcosystemSimulator(W=48,D=48,seed=42)
fig,axes=plt.subplots(2,4,figsize=(14,7))
fig.suptitle("Figure M6.2 — Novel 2: Biome-Driven Ecosystem Simulation (BPES)\n"
             "Terrain+biome co-evolve: erosion, deposition, logistic vegetation growth",fontsize=11,fontweight='bold')
axes[0][0].imshow(eco.temperature,cmap='RdYlBu_r',origin='lower',interpolation='bilinear'); axes[0][0].set_title("Temperature T(x,z)",fontsize=9.5,fontweight='bold'); axes[0][0].axis('off')
axes[0][1].imshow(eco.moisture_map,cmap='Blues',origin='lower',interpolation='bilinear'); axes[0][1].set_title("Moisture M(x,z)",fontsize=9.5,fontweight='bold'); axes[0][1].axis('off')
bm0=eco.get_biome_map(); axes[0][2].imshow(bm0,origin='lower',interpolation='nearest'); axes[0][2].set_title("Initial Biome Map",fontsize=9.5,fontweight='bold'); axes[0][2].axis('off')
h0=eco.get_height_map().copy(); axes[0][3].imshow(h0,cmap='terrain',origin='lower',interpolation='bilinear'); axes[0][3].set_title("Initial Height (t=0)",fontsize=9.5,fontweight='bold'); axes[0][3].axis('off')
print("  Running 300 steps...")
eco.step(dt=0.1,n_steps=200); h200=eco.get_height_map().copy(); veg200=eco.vegetation.copy()
eco.step(dt=0.1,n_steps=100); h300=eco.get_height_map().copy()
axes[1][0].imshow(h200,cmap='terrain',origin='lower',interpolation='bilinear'); axes[1][0].set_title("Height t=20\n(erosion forming)",fontsize=9.5,fontweight='bold'); axes[1][0].axis('off')
im_v=axes[1][1].imshow(veg200,cmap='Greens',origin='lower',vmin=0,vmax=1); axes[1][1].set_title("Vegetation t=20\n(logistic growth)",fontsize=9.5,fontweight='bold'); axes[1][1].axis('off'); plt.colorbar(im_v,ax=axes[1][1],shrink=0.8)
hdiff=h300-h0; imd=axes[1][2].imshow(hdiff,cmap='RdBu_r',origin='lower',vmin=-0.5,vmax=0.5); axes[1][2].set_title("Δheight (t=0→30)\nred=erode blue=deposit",fontsize=9.5,fontweight='bold'); axes[1][2].axis('off'); plt.colorbar(imd,ax=axes[1][2],shrink=0.8)
hist=eco.history; ts_h=[r['time'] for r in hist]; veg_h=[r['mean_veg'] for r in hist]; ero_h=[r['mean_erosion'] for r in hist]
axes[1][3].plot(ts_h,veg_h,'g-',lw=2,label='Vegetation'); axr=axes[1][3].twinx(); axr.plot(ts_h,ero_h,'r--',lw=1.5,label='Erosion')
axes[1][3].set_title("Ecosystem Evolution",fontsize=9.5,fontweight='bold'); axes[1][3].set_xlabel("Time"); axes[1][3].set_ylabel("Vegetation",color='g'); axr.set_ylabel("Erosion",color='r')
axes[1][3].legend(loc='upper left',fontsize=7.5); axr.legend(loc='upper right',fontsize=7.5); axes[1][3].grid(True,alpha=0.3)
plt.tight_layout(); save(fig,OUT6,"m6_fig2_ecosystem")
print("\nAll M5+M6 figures done!")
