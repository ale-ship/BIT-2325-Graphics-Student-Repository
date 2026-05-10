# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 2 — Visualisation Script (PyVista + Matplotlib)
# Topic: Procedural Content Generation Systems
#
# Authors:  Esther Achieng Otieno (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu (SCT221-C004-0217/2024)
#           Wendy Wachira (SCT221-C004-0194/2024)
#           Alexander Somba (SCT221-C004-0680/2023)
# Date  : May 2026 | JKUAT
#
# WHY PyVista:
#   PyVista wraps VTK (Visualization Toolkit) — a full 3D engine.
#   Unlike Matplotlib's basic 3D, PyVista provides:
#     - Real camera model (position, focal point, view-up, FOV)
#     - Smooth surface shading with normals
#     - Direct 4×4 matrix transformations on meshes
#     - Multiple light sources
#   Our camera and transformation MATHS come from milestone2_core.py
#   PyVista only handles the visual output — not the computation.
#
# Run: python3 milestone2_visualise.py
# Output: ./figures_m2/  (8 PNG figures)
# =============================================================

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyvista as pv
import math

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core  import Vector3, Matrix4, Terrain
from milestone2_core import TransformPipeline, Camera, NumericalAnalysis

pv.global_theme.allow_empty_mesh = True

OUT = "./figures_m2"
os.makedirs(OUT, exist_ok=True)

def save_mpl(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}.png")

def save_pv(plotter, name):
    plotter.screenshot(f"{OUT}/{name}.png", window_size=[1200,800])
    plotter.close()
    print(f"  Saved: {name}.png")

# ── Shared terrain and biome colours ─────────────────────────
BIOME_CMAP = {
    'deep_water':    [0.10,0.25,0.44],
    'shallow_water': [0.18,0.42,0.63],
    'sand':          [0.76,0.70,0.50],
    'grass':         [0.35,0.54,0.24],
    'forest':        [0.18,0.35,0.11],
    'rock':          [0.48,0.44,0.38],
    'snow':          [0.94,0.94,0.94],
}

def make_terrain_mesh(terrain):
    """Convert Terrain to a PyVista StructuredGrid mesh."""
    W, D = terrain.width, terrain.depth
    hgrid = np.array(terrain.height_grid())
    X = np.linspace(-1, 1, W)
    Z = np.linspace(-1, 1, D)
    XX, ZZ = np.meshgrid(X, Z)
    YY = hgrid
    grid = pv.StructuredGrid(XX, YY, ZZ)
    # Attach height as scalar for colouring
    grid['height'] = hgrid.flatten(order='C')
    # Biome colours as RGB
    colours = np.zeros((D*W, 3))
    for j in range(D):
        for i in range(W):
            bm = terrain.classify_biome(hgrid[j,i])
            colours[j*W+i] = BIOME_CMAP[bm]
    grid['biome_rgb'] = colours
    return grid


# =============================================================
# FIGURE 1 — Main 3D terrain with camera (PyVista)
# =============================================================
print("Generating Figure 1 — Main 3D terrain (PyVista)...")

terrain = Terrain(width=128, depth=128, scale=5.5,
                  height_scale=3.0, octaves=7, seed=42)
mesh = make_terrain_mesh(terrain)

camera_m2 = Camera(
    position = Vector3(4.0, 5.0, 8.0),
    target   = Vector3(0.0, 0.0, 0.0),
    up       = Vector3(0.0, 1.0, 0.0),
    fov_deg  = 60.0, aspect=16/9, near=0.1, far=50.0
)

pl = pv.Plotter(off_screen=True, window_size=[1200,800])
pl.set_background('#0d0d1a')
pl.add_mesh(mesh, scalars='biome_rgb', rgb=True,
            smooth_shading=True, show_edges=False,
            lighting=True)

# Apply our camera to PyVista
pl.camera.position    = (4.0, 5.0, 8.0)
pl.camera.focal_point = (0.0, 0.0, 0.0)
pl.camera.up          = (0.0, 1.0, 0.0)
pl.camera.view_angle  = 60.0

# Lighting
pl.add_light(pv.Light(position=(5,10,5),   focal_point=(0,0,0),
                       color='white', intensity=0.9))
pl.add_light(pv.Light(position=(-3,5,-5),  focal_point=(0,0,0),
                       color='#aaddff', intensity=0.4))

pl.add_title("Milestone 2 — Terrain in 3D World Space\n"
             "Camera: pos=(4,5,8)  FOV=60°  lookAt=(0,0,0)",
             font_size=12, color='white')
save_pv(pl, "fig1_m2_terrain_pyvista")


# =============================================================
# FIGURE 2 — Camera orbit: 4 viewing angles (PyVista)
# =============================================================
print("Generating Figure 2 — Camera orbit views (PyVista)...")

orbit_configs = [
    ("Front-Left\nyaw=30° pitch=25°",  30,  25),
    ("Top-Down\nyaw=0° pitch=80°",      0,  80),
    ("Side View\nyaw=90° pitch=15°",   90,  15),
    ("Rear-Right\nyaw=200° pitch=30°",200,  30),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Figure 2 — Camera Orbit: Same Terrain Viewed from 4 Positions\n"
             "Demonstrates lookAt() view matrix with different camera positions",
             fontsize=12, fontweight='bold')
fig.patch.set_facecolor('#0d0d1a')

for idx, (label, yaw, pitch) in enumerate(orbit_configs):
    row, col = idx//2, idx%2
    # orbit camera
    orb_cam = Camera(Vector3(0,5,8), Vector3(0,0,0), Vector3(0,1,0),
                     fov_deg=55, aspect=16/9, near=0.1, far=50)
    orb_cam.orbit(yaw_deg=yaw, pitch_deg=pitch, distance=8)

    pl2 = pv.Plotter(off_screen=True, window_size=[600,400])
    pl2.set_background('#0d0d1a')
    pl2.add_mesh(mesh, scalars='biome_rgb', rgb=True,
                 smooth_shading=True, lighting=True)
    pos = orb_cam.position
    pl2.camera.position    = (pos.x, pos.y, pos.z)
    pl2.camera.focal_point = (0.0, 0.0, 0.0)
    pl2.camera.up          = (0.0, 1.0, 0.0)
    pl2.camera.view_angle  = 55.0
    pl2.add_light(pv.Light(position=(5,10,5), focal_point=(0,0,0),
                            color='white', intensity=0.9))
    img = pl2.screenshot(return_img=True)
    pl2.close()

    ax = axes[row][col]
    ax.imshow(img)
    ax.set_title(label + f"\nPos: ({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})",
                 fontsize=9, fontweight='bold', color='white')
    ax.axis('off')
    ax.set_facecolor('#0d0d1a')

plt.tight_layout()
save_mpl(fig, "fig2_m2_orbit")


# =============================================================
# FIGURE 3 — Transformation pipeline visualisation
# =============================================================
print("Generating Figure 3 — Transformation pipeline...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Figure 3 — MVP Transformation Pipeline Applied to Terrain\n"
             "Each column shows one stage of the pipeline",
             fontsize=12, fontweight='bold')

pipeline = TransformPipeline()
camera3  = Camera(Vector3(3,4,6), Vector3(0,0,0), Vector3(0,1,0),
                  fov_deg=60, aspect=16/9, near=0.1, far=50)
pipeline.set_view(camera3.view_matrix)
pipeline.set_projection(camera3.projection_matrix)

terrain_s = Terrain(width=32, depth=32, scale=5.0,
                    height_scale=3.0, octaves=6, seed=42)

# Collect points at each pipeline stage
obj_pts, world_pts, cam_pts, ndc_pts = [], [], [], []
for j in range(0, terrain_s.depth, 4):
    for i in range(0, terrain_s.width, 4):
        v     = terrain_s.get_vertex(i,j)
        w_pt  = pipeline._model * v
        c_pt  = camera3.view_matrix * w_pt
        ndc   = camera3.project_point(w_pt, camera3)

        obj_pts.append(v)
        world_pts.append(w_pt)
        cam_pts.append(c_pt)
        if ndc: ndc_pts.append(ndc)

def pts_to_arrays(pts):
    return ([p.x for p in pts],[p.y for p in pts],[p.z for p in pts])

def ndc_to_arrays(ndcs):
    return ([n[0] for n in ndcs],[n[1] for n in ndcs],[n[2] for n in ndcs])

BGDARK = '#0d0d1a'
stage_data = [
    ("Object Space\n(local coords)", obj_pts,   True,  'steelblue'),
    ("World Space\n(after Model M)",  world_pts, True,  'limegreen'),
    ("Camera Space\n(after View V)",  cam_pts,   True,  'orange'),
]

for idx, (title, pts, is3d, col) in enumerate(stage_data):
    ax = axes[0][idx]
    xs,ys,zs = pts_to_arrays(pts)
    sc = ax.scatter(xs, zs, c=ys, cmap='terrain', s=20, alpha=0.8)
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_facecolor(BGDARK)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    plt.colorbar(sc, ax=ax, label='Height')
    ax.grid(True, alpha=0.2, color='white')

# NDC (after Projection P)
ax = axes[1][0]
nx,ny,nz = ndc_to_arrays(ndc_pts)
sc = ax.scatter(nx, ny, c=nz, cmap='plasma', s=20, alpha=0.8)
ax.set_title("NDC Space\n(after Projection P)", fontsize=9.5, fontweight='bold')
ax.set_xlabel("X NDC"); ax.set_ylabel("Y NDC")
ax.add_patch(plt.Rectangle((-1,-1),2,2,fc='none',ec='red',lw=2,ls='--'))
ax.text(0,1.1,"screen boundary",ha='center',fontsize=8,color='red')
ax.set_facecolor(BGDARK); ax.tick_params(colors='white')
ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
ax.title.set_color('white')
plt.colorbar(sc, ax=ax, label='Depth')
ax.grid(True, alpha=0.2, color='white')

# Matrix composition diagram
ax = axes[1][1]
ax.set_facecolor(BGDARK); ax.axis('off')
ax.text(0.5,0.95,"MVP = Projection × View × Model",
        ha='center',va='top',fontsize=11,fontweight='bold',color='white',
        transform=ax.transAxes)
mvp = pipeline.mvp
lines = []
for r in range(4):
    row = [f"{mvp.get(r,c):7.3f}" for c in range(4)]
    lines.append("│ " + "  ".join(row) + " │")
for k,l in enumerate(lines):
    ax.text(0.5,0.68-k*0.12,l,ha='center',va='top',fontsize=9,
            color='limegreen',fontfamily='monospace',transform=ax.transAxes)
ax.text(0.5,0.15,"One matrix — all transforms combined",
        ha='center',fontsize=9,color='grey',style='italic',transform=ax.transAxes)
ax.title.set_color('white')
ax.set_title("Combined MVP Matrix", fontsize=9.5, fontweight='bold', color='white')

# Depth buffer precision
ax = axes[1][2]
ax.set_facecolor(BGDARK)
z_vals = np.linspace(-0.1, -50, 500)
prec   = [2*0.1*50/((50-0.1)*z**2) for z in z_vals]
ax.semilogy(np.abs(z_vals), prec, color='orange', lw=2)
ax.fill_between(np.abs(z_vals), prec, alpha=0.2, color='orange')
ax.set_xlabel("World depth (-z)"); ax.set_ylabel("Depth buffer precision (log)")
ax.set_title("Depth Buffer Precision\n(near plane has most precision)",
             fontsize=9.5, fontweight='bold', color='white')
ax.axvline(5, color='red', lw=1.5, ls='--', label='z=5 (low)')
ax.axvline(0.5, color='lime', lw=1.5, ls='--', label='z=0.5 (high)')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.2, color='white')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
ax.title.set_color('white')

plt.tight_layout()
save_mpl(fig, "fig3_m2_pipeline")


# =============================================================
# FIGURE 4 — FOV comparison (PyVista)
# =============================================================
print("Generating Figure 4 — FOV comparison...")

fov_configs = [(30,"Narrow FOV=30°\n(telephoto)"),(60,"Standard FOV=60°"),
               (90,"Wide FOV=90°"),(110,"Ultra-wide FOV=110°")]

fig, axes = plt.subplots(1,4,figsize=(16,5))
fig.suptitle("Figure 4 — Field of View Effect on Terrain Perception\n"
             "Same camera position (4,4,8), only FOV changes",
             fontsize=12, fontweight='bold')
fig.patch.set_facecolor(BGDARK)

for idx,(fov,label) in enumerate(fov_configs):
    pl_fov = pv.Plotter(off_screen=True, window_size=[500,400])
    pl_fov.set_background(BGDARK)
    pl_fov.add_mesh(mesh, scalars='biome_rgb', rgb=True,
                    smooth_shading=True, lighting=True)
    pl_fov.camera.position    = (4.0, 4.0, 8.0)
    pl_fov.camera.focal_point = (0.0, 0.0, 0.0)
    pl_fov.camera.view_angle  = float(fov)
    pl_fov.add_light(pv.Light(position=(5,10,5),focal_point=(0,0,0),
                               color='white',intensity=0.9))
    img = pl_fov.screenshot(return_img=True)
    pl_fov.close()

    f_val = 1/math.tan(math.radians(fov/2))
    ax = axes[idx]
    ax.imshow(img)
    ax.set_title(label+f"\nf = {f_val:.3f}", fontsize=9.5,
                 fontweight='bold', color='white')
    ax.axis('off'); ax.set_facecolor(BGDARK)

plt.tight_layout()
save_mpl(fig, "fig4_m2_fov")


# =============================================================
# FIGURE 5 — Transformation effects (scale, rotate, translate)
# =============================================================
print("Generating Figure 5 — Transformation effects...")

transforms = [
    ("Identity\n(original)",        Matrix4.identity()),
    ("Scale(1, 2, 1)\n(stretch Y)", Matrix4.scale(1,2,1)),
    ("RotateY(45°)\n(tilted)",      Matrix4.rotation_y(45)),
    ("Translate(0.5,0,0)\n(shift)", Matrix4.translation(0.5,0,0)),
]

fig, axes = plt.subplots(1,4,figsize=(16,5))
fig.suptitle("Figure 5 — Model Matrix Transformations Applied to Terrain\n"
             "Camera fixed at (3,4,7), only Model matrix changes",
             fontsize=12, fontweight='bold')
fig.patch.set_facecolor(BGDARK)

terrain_t = Terrain(width=64,depth=64,scale=5.0,height_scale=3.0,octaves=6,seed=42)
mesh_t    = make_terrain_mesh(terrain_t)

for idx,(label,M) in enumerate(transforms):
    # Apply matrix to mesh via numpy
    pts = mesh_t.points.copy()
    pts_h = np.hstack([pts, np.ones((len(pts),1))])   # homogeneous
    M_np  = np.array(M.data).reshape(4,4)
    transformed = (M_np @ pts_h.T).T[:,:3]

    mesh_copy = mesh_t.copy()
    mesh_copy.points = transformed

    pl_t = pv.Plotter(off_screen=True, window_size=[500,400])
    pl_t.set_background(BGDARK)
    pl_t.add_mesh(mesh_copy, scalars='biome_rgb', rgb=True,
                  smooth_shading=True, lighting=True)
    pl_t.camera.position    = (3.0,4.0,7.0)
    pl_t.camera.focal_point = (0.0,0.0,0.0)
    pl_t.camera.view_angle  = 60.0
    pl_t.add_light(pv.Light(position=(5,10,5),focal_point=(0,0,0),
                             color='white',intensity=0.9))
    img = pl_t.screenshot(return_img=True)
    pl_t.close()

    ax = axes[idx]
    ax.imshow(img)
    ax.set_title(label, fontsize=9.5, fontweight='bold', color='white')
    ax.axis('off'); ax.set_facecolor(BGDARK)

plt.tight_layout()
save_mpl(fig, "fig5_m2_transforms")


# =============================================================
# FIGURE 6 — Numerical stability analysis
# =============================================================
print("Generating Figure 6 — Numerical stability...")

fig, axes = plt.subplots(1,3,figsize=(14,5))
fig.suptitle("Figure 6 — Numerical Stability Analysis of the Transformation Pipeline",
             fontsize=12, fontweight='bold')

# Panel 1: depth precision curve
ax = axes[0]
for near,far,col,lbl in [(0.1,100,'steelblue','near=0.1 far=100'),
                           (1.0,100,'orange',  'near=1.0 far=100'),
                           (0.1,500,'red',     'near=0.1 far=500')]:
    z_vals = np.linspace(-(near+0.01),-far+0.1,300)
    prec   = [2*near*far/((far-near)*z**2) for z in z_vals]
    ax.semilogy(np.abs(z_vals), prec, color=col, lw=2, label=lbl)
ax.set_xlabel("World depth (-z)"); ax.set_ylabel("Depth precision (log scale)")
ax.set_title("Z-Buffer Precision\nby near/far clip settings", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.text(50,0.0001,"Lost precision\n(far objects)",fontsize=8,color='red',style='italic')

# Panel 2: rotation error accumulation
ax = axes[1]
R = Matrix4.rotation_y(1.0)   # 1° per step
errors = []
M_acc  = Matrix4.identity()
for step in range(360):
    M_acc = R * M_acc
    err   = NumericalAnalysis.orthogonality_error(M_acc)
    errors.append(err)
ax.plot(range(360), errors, color='firebrick', lw=2)
ax.set_xlabel("Accumulated rotations (steps × 1°)")
ax.set_ylabel("Orthogonality error |R^TR - I|")
ax.set_title("Float Error Accumulation\nin Repeated Rotation", fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.fill_between(range(360), errors, alpha=0.2, color='firebrick')
ax.text(180,max(errors)*0.7,"Error grows with\nmore rotations",
        fontsize=8.5,color='firebrick',style='italic',ha='center')

# Panel 3: condition numbers per matrix
ax = axes[2]
pipe_test = TransformPipeline()
cam_test  = Camera(Vector3(3,4,6),Vector3(0,0,0),Vector3(0,1,0),
                   fov_deg=60,near=0.1,far=100)
pipe_test.set_view(cam_test.view_matrix)
pipe_test.set_projection(cam_test.projection_matrix)
analysis  = NumericalAnalysis.analyse_pipeline(pipe_test)
names_  = list(analysis.keys())
kappas  = [analysis[n]['condition'] for n in names_]
colors_ = ['steelblue','orange','green','red']
bars    = ax.bar(names_, kappas, color=colors_, alpha=0.85, edgecolor='white', lw=1.5)
ax.set_title("Condition Numbers κ(M)\n(lower = more stable)", fontsize=10, fontweight='bold')
ax.set_ylabel("Condition number κ"); ax.grid(axis='y', alpha=0.3)
for bar,k in zip(bars,kappas):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{k:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.text(0.5,-0.12,"κ ≈ 1.0 for all matrices → well conditioned pipeline",
        ha='center',transform=ax.transAxes,fontsize=8.5,style='italic',color='grey')

plt.tight_layout()
save_mpl(fig, "fig6_m2_stability")


# =============================================================
# FIGURE 7 — System architecture (Milestone 1 + 2)
# =============================================================
print("Generating Figure 7 — System architecture...")

fig, ax = plt.subplots(figsize=(14,8))
ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis('off')
fig.suptitle("Figure 7 — Milestone 1 + 2 Combined System Architecture",
             fontsize=13, fontweight='bold')

from matplotlib.patches import FancyBboxPatch

def box(ax,x,y,w,h,label,sub,fc,ec='#333333'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                                fc=fc,ec=ec,lw=1.8,zorder=3))
    ax.text(x+w/2,y+h*0.65,label,ha='center',va='center',
            fontsize=9,fontweight='bold',color='white',zorder=4)
    ax.text(x+w/2,y+h*0.28,sub,ha='center',va='center',
            fontsize=7.5,color='#dddddd',zorder=4,style='italic')

def arr(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color='#555555',lw=1.8),zorder=2)

# M1 layer
box(ax,0.2,0.3,3.0,1.2,"Vector3","dot|cross|normalise|lerp",'#1a5276')
box(ax,3.7,0.3,3.0,1.2,"Matrix4","translate|scale|rotate",'#1a5276')
box(ax,7.2,0.3,3.0,1.2,"PerlinNoise","gradient|octave fBm",'#117a65')
box(ax,10.7,0.3,3.0,1.2,"Terrain M1","height|normals|biomes",'#6e2f8a')

# M2 layer — new
box(ax,0.2,2.3,4.5,1.3,"TransformPipeline","Model×View×Proj|MVP cache",'#1f618d')
box(ax,5.2,2.3,4.5,1.3,"Camera","lookAt()|perspective()|orbit()|rays()",'#1f618d')
box(ax,10.2,2.3,3.5,1.3,"NumericalAnalysis","κ(M)|ortho_err|depth precision",'#1f618d')

# Output layer
box(ax,0.2,4.4,4.5,1.3,"PyVista Renderer","3D terrain|lighting|camera export",'#922b21')
box(ax,5.2,4.4,4.5,1.3,"Matplotlib Analysis","pipeline|FOV|stability charts",'#922b21')
box(ax,10.2,4.4,3.5,1.3,"Report","derivations|analysis|figures",'#922b21')

# Arrows M1 → M2
arr(ax,1.7,1.5,2.4,2.3); arr(ax,5.2,1.5,7.4,2.3)
arr(ax,8.7,1.5,7.4,2.3); arr(ax,12.2,1.5,11.0,2.3)
# Arrows M2 → output
arr(ax,2.4,3.6,2.4,4.4); arr(ax,7.4,3.6,7.4,4.4)
arr(ax,11.9,3.6,11.9,4.4)

# Layer labels
ax.text(0.05,0.85,"MILESTONE 1\nFoundations",fontsize=8.5,color='#1a5276',fontweight='bold')
ax.text(0.05,2.85,"MILESTONE 2\nSpace & Camera",fontsize=8.5,color='#1f618d',fontweight='bold')
ax.text(0.05,4.95,"OUTPUT",fontsize=8.5,color='#922b21',fontweight='bold')

# NEW badge
ax.text(7.0,1.85,"NEW ↑",fontsize=9,color='gold',fontweight='bold',ha='center')
ax.text(3.5,1.85,"NEW ↑",fontsize=9,color='gold',fontweight='bold',ha='center')
ax.text(11.9,1.85,"NEW ↑",fontsize=9,color='gold',fontweight='bold',ha='center')

plt.tight_layout()
save_mpl(fig, "fig7_m2_architecture")


# =============================================================
# FIGURE 8 — Multi-camera render grid (PyVista)
# =============================================================
print("Generating Figure 8 — Multi-camera showcase...")

cam_configs = [
    ("Aerial Overview\nFOV=50°",  (0,8,0.1), 50),
    ("Ground Level\nFOV=70°",     (5,0.5,5), 70),
    ("Dramatic Angle\nFOV=60°",   (6,3,6),   60),
    ("Isometric Style\nFOV=25°",  (6,6,6),   25),
]

fig, axes = plt.subplots(2,2,figsize=(14,9))
fig.suptitle("Figure 8 — PyVista Multi-Camera Showcase\n"
             "Same terrain, different camera positions and FOV values",
             fontsize=12, fontweight='bold')
fig.patch.set_facecolor(BGDARK)

for idx,((label,pos,fov)) in enumerate(cam_configs):
    pl_c = pv.Plotter(off_screen=True, window_size=[700,500])
    pl_c.set_background(BGDARK)
    pl_c.add_mesh(mesh, scalars='biome_rgb', rgb=True,
                  smooth_shading=True, lighting=True)
    pl_c.camera.position    = pos
    pl_c.camera.focal_point = (0.0,0.0,0.0)
    pl_c.camera.up          = (0.0,1.0,0.0)
    pl_c.camera.view_angle  = float(fov)
    pl_c.add_light(pv.Light(position=(5,10,5),focal_point=(0,0,0),
                             color='white',intensity=0.9))
    pl_c.add_light(pv.Light(position=(-3,3,-5),focal_point=(0,0,0),
                             color='#aaddff',intensity=0.35))
    img = pl_c.screenshot(return_img=True)
    pl_c.close()

    row,col = idx//2,idx%2
    f_val = 1/math.tan(math.radians(fov/2))
    ax = axes[row][col]
    ax.imshow(img)
    ax.set_title(f"{label}\nPos={pos}  f={f_val:.2f}",
                 fontsize=9.5, fontweight='bold', color='white')
    ax.axis('off'); ax.set_facecolor(BGDARK)

plt.tight_layout()
save_mpl(fig, "fig8_m2_cameras")

print(f"\nAll Milestone 2 figures generated in {OUT}/")
print("Total: 8 figures")
