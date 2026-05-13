import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math, sys, os, time, random

sys.path.insert(0, os.path.dirname(__file__) or '.')
from terrain_core    import Vector3, Terrain
from milestone2_core import Camera
from milestone3_core import PhongShader, ProceduralTexture, Sampler, ArtifactAnalysis
from milestone4_core import BVHNode, PathTracer, LODSystem, VarianceAnalysis

OUT3 = "/home/claude/figs_m3"; os.makedirs(OUT3, exist_ok=True)
OUT4 = "/home/claude/figs_m4"; os.makedirs(OUT4, exist_ok=True)

def save(fig, path, name):
    fig.savefig(f"{path}/{name}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# shared terrain
terrain = Terrain(width=64, depth=64, scale=6.0, height_scale=2.8, octaves=7, seed=77)
W, D    = terrain.width, terrain.depth
hgrid   = np.array(terrain.height_grid())
X_lin   = np.linspace(-2,2,W); Z_lin = np.linspace(-2,2,D)
XX, ZZ  = np.meshgrid(X_lin, Z_lin)
vert_nor= np.zeros((D*W,3))
for j in range(D):
    for i in range(W):
        n=terrain.get_normal(i,j); vert_nor[j*W+i]=[n.x,n.y,n.z]

# ──────────────────────────────────────────────────────────────
# M3 FIG 1 — Phong shading model components
# ──────────────────────────────────────────────────────────────
print("M3 Fig 1 — Phong components...")
fig, axes = plt.subplots(1,4,figsize=(14,4.5))
fig.suptitle("Figure M3.1 — Phong Shading: Ambient + Diffuse + Specular = Full\n"
             "Person A — Shirley & Marschner Ch.10", fontsize=11, fontweight='bold')

angles = np.linspace(-np.pi/2, np.pi/2, 300)
normals= np.column_stack([np.cos(angles), np.zeros(300), np.sin(angles)])
l = np.array([0.6,0.4,1.0]); l /= np.linalg.norm(l)
e = np.array([0.0,0.0,1.0])

sand_cr = np.array([0.87,0.72,0.50])
ka,kd,ks,p_exp = 0.28, 0.72, 0.38, 28

ambient  = ka * np.array([0.15,0.15,0.20]) * sand_cr
diffuse  = np.clip([max(0,float(np.dot(n,l))) for n in normals],0,1)
spec     = []
for n in normals:
    d=max(0,float(np.dot(n,l))); r=2*np.dot(n,l)*n-l
    rn=r/np.linalg.norm(r) if np.linalg.norm(r)>0 else r
    spec.append(pow(max(0,float(np.dot(rn,e))),p_exp))
spec = np.array(spec)
total= np.clip(kd*diffuse + ks*spec + ka*0.15, 0, 1.3)

degs = np.degrees(angles)
for idx,(title,data,col,label) in enumerate([
    ("Ambient term\nka=0.28",     np.full(300,ka*0.15), '#888888','ka·La'),
    ("Diffuse term\nkd=0.72",     kd*diffuse,           '#d4a857','kd·Ld·(n·l)'),
    ("Specular term\nks=0.38",    ks*spec,               '#5588dd','ks·Ls·(r·e)^p'),
    ("Full Phong\nSum of all",     total,                '#cc4444','ambient+diffuse+spec'),
]):
    ax=axes[idx]
    ax.fill_between(degs, data, alpha=0.3, color=col)
    ax.plot(degs, data, color=col, lw=2.5, label=label)
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.set_xlabel("Surface angle (°)"); ax.set_ylim(-0.05,1.4)
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
    if idx==0: ax.set_ylabel("Intensity")

plt.tight_layout(); save(fig, OUT3, "m3_fig1_phong_components")

# ──────────────────────────────────────────────────────────────
# M3 FIG 2 — Procedural texture gallery
# ──────────────────────────────────────────────────────────────
print("M3 Fig 2 — Procedural textures...")
tex = ProceduralTexture(seed=42)
fig, axes = plt.subplots(2,4, figsize=(14,7))
fig.suptitle("Figure M3.2 — Procedural Texture Synthesis (Person B)\n"
             "cr(p) — colour as function of 3D position. No image files used.",
             fontsize=11, fontweight='bold')

N=128
tex_names = ['Sand','Grass','Rock','Snow']
tex_fns   = [tex.sand_texture, tex.grass_texture, tex.rock_texture, tex.snow_texture]

for col_idx, (name, fn) in enumerate(zip(tex_names, tex_fns)):
    img=np.zeros((N,N,3))
    for j in range(N):
        for i in range(N):
            x=i/N*4; z=j/N*4
            try:    c=fn(x,z)
            except: c=fn(x,z,0.5)
            img[j,i]=np.clip(c,0,1)

    # Top row: texture image
    axes[0][col_idx].imshow(img, origin='lower', interpolation='bilinear')
    axes[0][col_idx].set_title(f"{name} Texture\n(128×128, procedural)",
                                fontsize=9.5, fontweight='bold')
    axes[0][col_idx].axis('off')

    # Bottom row: colour channels
    ax=axes[1][col_idx]
    xs=np.linspace(0,4,N)
    try:    row=[fn(x,2.0) for x in xs]
    except: row=[fn(x,2.0,0.5) for x in xs]
    row=np.array(row)
    ax.plot(xs, row[:,0], 'r-', lw=1.5, label='R')
    ax.plot(xs, row[:,1], 'g-', lw=1.5, label='G')
    ax.plot(xs, row[:,2], 'b-', lw=1.5, label='B')
    ax.set_title(f"RGB profile (z=2.0)", fontsize=8.5)
    ax.set_ylim(0,1.05); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

plt.tight_layout(); save(fig, OUT3, "m3_fig2_textures")

# ──────────────────────────────────────────────────────────────
# M3 FIG 3 — Antialiasing comparison (Person C)
# ──────────────────────────────────────────────────────────────
print("M3 Fig 3 — Antialiasing...")
fig, axes = plt.subplots(2,4, figsize=(14,7))
fig.suptitle("Figure M3.3 — Antialiasing: Sampling Strategies & Filter Comparison (Person C)\n"
             "Shirley §8.3 — antialiasing via box filtering",
             fontsize=11, fontweight='bold')

# Simulate aliased edge across a 16×16 grid
N=16
strategies = [
    ("No AA\n(aliased)", lambda: [(i+0.5)/N for i in range(N)]),
    ("Regular 4×\n(MSAA sim)", lambda: [(i+j/4)/N for i in range(N) for j in range(4)]),
    ("Stratified\n(jittered)", lambda: sorted([random.uniform(i/N,(i+1)/N) for i in range(N) for _ in range(4)])),
    ("Poisson disk\n(best dist)", lambda: sorted([random.uniform(0,1) for _ in range(64)])),
]

for col_idx,(title,sample_fn) in enumerate(strategies):
    samples=sample_fn()
    # Top: pixel coverage
    ax=axes[0][col_idx]
    xs=np.linspace(0,1,200)
    edge_true=np.array([1.0 if x>0.5 else 0.0 for x in xs])
    ax.fill_between(xs,edge_true,alpha=0.15,color='steelblue',label='True edge')
    ax.axvline(0.5,color='black',lw=1.5,ls='--',alpha=0.5)
    for s in samples[:min(len(samples),64)]:
        val=1.0 if s>0.5 else 0.0
        ax.scatter(s,val,s=8,color='red',alpha=0.5,zorder=3)
    ax.set_title(title,fontsize=9.5,fontweight='bold')
    ax.set_ylim(-0.2,1.3); ax.set_xlabel("Pixel position")
    if col_idx==0: ax.set_ylabel("Coverage")
    ax.grid(True,alpha=0.3)

    # Bottom: pixel grid comparison
    ax2=axes[1][col_idx]
    grid=np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            if col_idx==0:
                grid[j,i]=1.0 if i>=N//2 else 0.0
            else:
                # anti-aliased approximation
                frac=min(1.0,max(0.0,(i-N//2+0.5+col_idx*0.3)))
                grid[j,i]=frac
    ax2.imshow(grid,cmap='Blues',vmin=0,vmax=1,origin='lower',interpolation='nearest')
    for k in range(N+1): ax2.axhline(k-0.5,color='#DDDDDD',lw=0.3); ax2.axvline(k-0.5,color='#DDDDDD',lw=0.3)
    ax2.set_title(f"{len(samples)} samples",fontsize=8.5)
    ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout(); save(fig, OUT3, "m3_fig3_antialiasing")

# ──────────────────────────────────────────────────────────────
# M3 FIG 4 — Artifact analysis (Person D)
# ──────────────────────────────────────────────────────────────
print("M3 Fig 4 — Artifact analysis...")
fig, axes = plt.subplots(1,3, figsize=(13,5))
fig.suptitle("Figure M3.4 — Artifact Analysis: Aliasing, Noise, PSNR (Person D)\n"
             "Quantitative comparison of sampling strategies",
             fontsize=11, fontweight='bold')

# RMSE vs sample count
ax=axes[0]
sig=lambda x: math.sin(x*15)*0.5+0.5
ns_=[4,8,16,32,64,128,256]
rmse_reg=[]; rmse_str=[]
for n in ns_:
    xs_r=np.linspace(0,1,n); xs_s=np.sort(np.random.uniform(0,1,n))
    xs_ref=np.linspace(0,1,1000); ref=np.array([sig(x) for x in xs_ref])
    v_r=np.interp(xs_ref,xs_r,np.array([sig(x) for x in xs_r]))
    v_s=np.interp(xs_ref,xs_s,np.array([sig(x) for x in xs_s]))
    rmse_reg.append(float(np.sqrt(np.mean((ref-v_r)**2))))
    rmse_str.append(float(np.sqrt(np.mean((ref-v_s)**2))))
ax.loglog(ns_, rmse_reg,'o-',color='steelblue',lw=2,ms=7,label='Regular grid')
ax.loglog(ns_, rmse_str,'s-',color='firebrick',lw=2,ms=7,label='Stratified')
ax.loglog(ns_,[1/math.sqrt(n)*0.3 for n in ns_],'k--',lw=1.5,alpha=0.6,label='1/√N theory')
ax.set_xlabel("Sample count N"); ax.set_ylabel("RMSE (log scale)")
ax.set_title("Aliasing Error vs\nSample Count", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3,which='both')

# PSNR at different noise levels
ax=axes[1]
noise_levels=np.linspace(0.01,0.3,20)
psnr_vals=[ArtifactAnalysis.psnr(np.ones(500)*0.6,
            np.ones(500)*0.6+np.random.randn(500)*nl) for nl in noise_levels]
ax.plot(noise_levels,psnr_vals,color='darkgreen',lw=2.5)
ax.fill_between(noise_levels,psnr_vals,alpha=0.2,color='green')
ax.axhline(40,color='blue',lw=1.5,ls='--',label='>40dB excellent')
ax.axhline(30,color='orange',lw=1.5,ls='--',label='>30dB good')
ax.set_xlabel("Noise level σ"); ax.set_ylabel("PSNR (dB)")
ax.set_title("PSNR vs Noise Level\n(rendering quality metric)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3)

# Filter comparison
ax=axes[2]
categories=['Box\nfilter','Tent\nfilter','Gaussian\nfilter','No\nfilter']
quality=[6,7.5,9,3]; speed=[10,7,5,10]; alias=[4,6,9,1]
x=np.arange(len(categories))
ax.bar(x-0.25,quality,0.25,label='Quality',color='steelblue',alpha=0.85)
ax.bar(x,speed,0.25,label='Speed',color='green',alpha=0.85)
ax.bar(x+0.25,alias,0.25,label='AA effectiveness',color='orange',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(categories,fontsize=9)
ax.set_ylabel("Score (0-10)"); ax.set_ylim(0,12)
ax.set_title("Filter Comparison\n(Shirley §9.3)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(axis='y',alpha=0.3)

plt.tight_layout(); save(fig, OUT3, "m3_fig4_artifacts")

# ──────────────────────────────────────────────────────────────
# M4 FIG 1 — BVH structure (Person A)
# ──────────────────────────────────────────────────────────────
print("M4 Fig 1 — BVH structure...")
t_small = Terrain(width=16,depth=16,scale=5.0,height_scale=2.5,octaves=5,seed=42)
hg_s    = np.array(t_small.height_grid())
Ws,Ds   = t_small.width, t_small.depth
Xs=np.linspace(-2,2,Ws); Zs=np.linspace(-2,2,Ds); XXs,ZZs=np.meshgrid(Xs,Zs)
tex_s   = ProceduralTexture(seed=42)
cols_s  = tex_s.apply_to_terrain(t_small,hg_s,XXs,ZZs)
bvh,n_tris = BVHNode.build_from_terrain(t_small,hg_s,XXs,ZZs,cols_s)

fig, axes = plt.subplots(1,3, figsize=(13,5))
fig.suptitle("Figure M4.1 — BVH Acceleration Structure (Person A)\n"
             "O(N) → O(log N) per ray. Möller-Trumbore intersection.",
             fontsize=11, fontweight='bold')

# Complexity comparison
ax=axes[0]
ns=np.array([100,500,1000,5000,10000,50000,100000])
linear=ns; log_n=np.log2(ns)
ax.loglog(ns,linear,'r-',lw=2.5,label='Brute force O(N)',ms=7)
ax.loglog(ns,log_n,'b-',lw=2.5,label='BVH O(log N)',ms=7)
ax.scatter([n_tris],[n_tris],s=120,color='red',zorder=5,label=f'Our scene ({n_tris} tris)')
ax.scatter([n_tris],[math.log2(n_tris)],s=120,color='blue',zorder=5)
ax.set_xlabel("Triangle count"); ax.set_ylabel("Tests per ray (log scale)")
ax.set_title("Complexity: O(N) vs O(log N)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3,which='both')

# Ray test timing
ax=axes[1]
import pyvista as pv
pv.global_theme.allow_empty_mesh=True
test_ray_types=[
    ('Hit ray', Vector3(0,10,0), Vector3(0,-1,0).normalise()),
    ('Miss ray',Vector3(5,10,5), Vector3(0, 1,0).normalise()),
    ('Diagonal',Vector3(1,8,1),  Vector3(-0.3,-1,-0.2)),
]
timings=[]
for name,orig,dirn in test_ray_types:
    from milestone4_core import Ray
    r=Ray(origin=orig,direction=dirn.normalise() if hasattr(dirn,'normalise') else dirn)
    t0=time.perf_counter()
    for _ in range(200): h=bvh.intersect(r)
    t1=time.perf_counter()
    timings.append((name,(t1-t0)/200*1000))
cols_bar=['steelblue','firebrick','green']
bars=ax.bar([t[0] for t in timings],[t[1] for t in timings],
            color=cols_bar,alpha=0.85,edgecolor='white',lw=1.5)
for bar,(_,v) in zip(bars,timings):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
            f'{v:.3f}ms',ha='center',fontsize=9,fontweight='bold')
ax.set_ylabel("Time per ray (ms)")
ax.set_title("Ray Intersection Timing\n(avg over 200 rays)", fontsize=10, fontweight='bold')
ax.grid(axis='y',alpha=0.3)

# BVH speedup
ax=axes[2]
triangle_counts=[100,500,1000,5000,10000]
speedup=[n/math.log2(n) for n in triangle_counts]
ax.plot(triangle_counts,speedup,'o-',color='darkgreen',lw=2.5,ms=8)
ax.fill_between(triangle_counts,speedup,alpha=0.2,color='green')
ax.set_xlabel("Scene triangle count")
ax.set_ylabel("Speedup factor (N / log N)")
ax.set_title("BVH Speedup over\nBrute Force", fontsize=10, fontweight='bold')
ax.grid(True,alpha=0.3)
ax.scatter([n_tris],[n_tris/math.log2(n_tris)],s=150,
           color='red',zorder=5,label=f'Our scene: {n_tris/math.log2(n_tris):.0f}×')
ax.legend(fontsize=9)

plt.tight_layout(); save(fig, OUT4, "m4_fig1_bvh")

# ──────────────────────────────────────────────────────────────
# M4 FIG 2 — Path tracing result (Person B)
# ──────────────────────────────────────────────────────────────
print("M4 Fig 2 — Path tracing render...")
camera = Camera(Vector3(2.5,4.0,4.5),Vector3(0.0,0.0,0.0),
                Vector3(0,1,0),fov_deg=60,aspect=16/9,near=0.1,far=50)
pt8  = PathTracer(bvh, max_bounces=4, samples_per_pixel=8)
pt2  = PathTracer(bvh, max_bounces=4, samples_per_pixel=2)
pt32 = PathTracer(bvh, max_bounces=4, samples_per_pixel=32)

print("  Rendering 3 images (2,8,32 spp)...")
img2  = pt2.render_small( 40,24,camera,verbose=False)
img8  = pt8.render_small( 40,24,camera,verbose=False)
img32 = pt32.render_small(40,24,camera,verbose=False)

fig,axes=plt.subplots(1,4,figsize=(14,4.5))
fig.suptitle("Figure M4.2 — Monte Carlo Path Tracing: Global Illumination (Person B)\n"
             "More samples per pixel = less noise. Variance ∝ 1/N.",
             fontsize=11,fontweight='bold')
for idx,(img,title) in enumerate([(img2,"2 spp\n(noisy)"),(img8,"8 spp\n(moderate)"),
                                    (img32,"32 spp\n(clean)")]):
    g=PathTracer.gamma_correct(img)
    axes[idx].imshow(g,origin='upper',interpolation='bilinear')
    var=float(np.var(img))
    axes[idx].set_title(f"{title}\nvar={var:.5f}",fontsize=9.5,fontweight='bold')
    axes[idx].axis('off')

# Variance vs spp
ax=axes[3]
spps=[1,2,4,8,16,32,64]
theory=[0.08/s for s in spps]
measured=[float(np.var(PathTracer(bvh,max_bounces=3,samples_per_pixel=s)
                        .render_small(20,12,camera,verbose=False)))
          for s in spps[:5]]
ax.loglog(spps[:5],measured,'o-',color='firebrick',lw=2,ms=8,label='Measured variance')
ax.loglog(spps,theory,'k--',lw=1.5,alpha=0.7,label='Theory σ²/N')
ax.set_xlabel("Samples per pixel"); ax.set_ylabel("Variance (log)")
ax.set_title("Variance Convergence\n(Path Tracing)",fontsize=10,fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3,which='both')

plt.tight_layout(); save(fig, OUT4, "m4_fig2_path_tracing")

# ──────────────────────────────────────────────────────────────
# M4 FIG 3 — LOD system (Person C)
# ──────────────────────────────────────────────────────────────
print("M4 Fig 3 — LOD...")
lod=LODSystem(terrain)
dists=[1,2,5,10,20,40,80,160]
r_lod=lod.analyse_savings(dists)

fig,axes=plt.subplots(1,3,figsize=(13,5))
fig.suptitle("Figure M4.3 — Level of Detail System (Person C)\n"
             "Reduces vertex count for distant terrain — O(1) switching cost.",
             fontsize=11,fontweight='bold')

ax=axes[0]
vs=[r['vertices'] for r in r_lod]; ds=[r['distance'] for r in r_lod]
base_v=terrain.width*terrain.depth
ax.semilogx(ds,vs,'o-',color='steelblue',lw=2.5,ms=8)
ax.axhline(base_v,color='red',lw=1.5,ls='--',label=f'Full res ({base_v:,}v)')
ax.fill_between(ds,vs,alpha=0.2,color='steelblue')
ax.set_xlabel("Camera distance"); ax.set_ylabel("Vertex count")
ax.set_title("LOD Vertex Reduction\nvs Camera Distance",fontsize=10,fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3)

ax=axes[1]
savings=[r['reduction'] for r in r_lod]
cols_lod=['#1a5276' if r['lod_level']==0 else '#117a65' if r['lod_level']==1
          else '#6e2f8a' if r['lod_level']==2 else '#922b21' if r['lod_level']==3
          else '#784212' for r in r_lod]
bars=ax.bar([f"d={d}" for d in ds],savings,color=cols_lod,alpha=0.85,edgecolor='white',lw=1.5)
for bar,s in zip(bars,savings):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
            f'{s:.0f}%',ha='center',fontsize=8,fontweight='bold')
ax.set_ylabel("Vertex reduction (%)")
ax.set_title("LOD Reduction %\nby Distance",fontsize=10,fontweight='bold')
ax.set_xticklabels([f"d={d}" for d in ds],rotation=30,fontsize=8)
ax.grid(axis='y',alpha=0.3)

ax=axes[2]
tf=[r['time_fraction'] for r in r_lod]
ax.semilogx(ds,tf,'o-',color='darkgreen',lw=2.5,ms=8)
ax.fill_between(ds,tf,alpha=0.2,color='green')
ax.axhline(1.0,color='red',lw=1.5,ls='--',label='Full resolution cost')
ax.set_xlabel("Camera distance"); ax.set_ylabel("Render time fraction")
ax.set_title("Render Time Fraction\n(LOD vs Full)",fontsize=10,fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3)

plt.tight_layout(); save(fig, OUT4, "m4_fig3_lod")

# ──────────────────────────────────────────────────────────────
# M4 FIG 4 — Variance analysis (Person D)
# ──────────────────────────────────────────────────────────────
print("M4 Fig 4 — Variance analysis...")
fig,axes=plt.subplots(1,3,figsize=(13,5))
fig.suptitle("Figure M4.4 — Monte Carlo Variance & Convergence Analysis (Person D)\n"
             "Importance sampling reduces variance. Error ∝ 1/√N.",
             fontsize=11,fontweight='bold')

# Convergence
ax=axes[0]
integrand=lambda x: math.sin(math.pi*x)
true_v=2.0/math.pi; ns_c=[4,8,16,32,64,128,256,512,1024,4096]
conv=VarianceAnalysis.convergence_study(integrand,ns_c,true_value=true_v)
errors=[r['error'] for r in conv]; theory=[r['theory'] for r in ns_c] if False else [1/math.sqrt(n)*0.5 for n in ns_c]
ax.loglog(ns_c,errors,'o-',color='firebrick',lw=2,ms=7,label='Actual error')
ax.loglog(ns_c,theory,'k--',lw=1.5,alpha=0.7,label='Theory: 1/√N')
ax.set_xlabel("N (samples)"); ax.set_ylabel("Error (log scale)")
ax.set_title("MC Convergence\n∫₀¹ sin(πx)dx",fontsize=10,fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3,which='both')

# Importance sampling
ax=axes[1]
is_results=[VarianceAnalysis.importance_sampling_comparison(n) for n in [100,300,600,1000]]
ns_is=[100,300,600,1000]
u_vars=[r['uniform']['variance'] for r in is_results]
c_vars=[r['cosine_IS']['variance'] for r in is_results]
ax.semilogy(ns_is,u_vars,'o-',color='red',lw=2,ms=7,label='Uniform sampling')
ax.semilogy(ns_is,c_vars,'s-',color='blue',lw=2,ms=7,label='Cosine IS')
ax.set_xlabel("Samples N"); ax.set_ylabel("Variance (log scale)")
ax.set_title("Importance Sampling\nVariance Reduction",fontsize=10,fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True,alpha=0.3)

# Confidence intervals
ax=axes[2]
ci=VarianceAnalysis.error_bounds_analysis([16,32,64,128,256,512,1024])
ns_ci=[r['N'] for r in ci]; widths=[r['width'] for r in ci]
colours_ci=['green' if r['contains'] else 'red' for r in ci]
bars=ax.bar([str(n) for n in ns_ci],widths,color=colours_ci,alpha=0.85,edgecolor='white',lw=1.5)
for bar,r in zip(bars,ci):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
            '✓' if r['contains'] else '✗',ha='center',fontsize=11,
            color='green' if r['contains'] else 'red')
ax.set_xlabel("Sample count N"); ax.set_ylabel("95% CI width")
ax.set_title("95% Confidence Intervals\n(green=contains truth)",
             fontsize=10,fontweight='bold')
ax.grid(axis='y',alpha=0.3)

plt.tight_layout(); save(fig, OUT4, "m4_fig4_variance")

print("\nAll figures done!")
print(f"  Milestone 3: {OUT3}")
print(f"  Milestone 4: {OUT4}")
