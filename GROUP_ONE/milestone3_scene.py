# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 3 — Enhanced Interactive Scene
# Adds: Phong shading, procedural textures, AA comparison
#
# Person A — Phong shading live in scene
# Person B — Procedural texture synthesis visible
# Person C — Lighting modes show AA effect
# Person D — Artifact analysis overlaid on HUD
#
# Run: python3 milestone3_scene.py
# Controls: same as milestone2_scene.py + new keys:
#   X    — cycle shading mode (Flat / Phong / Texture+Phong)
#   A    — toggle antialiasing comparison split-screen
#   I    — print artifact analysis to terminal
# =============================================================

import sys, os, math, time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core    import Vector3, Matrix4, Terrain
from milestone2_core import Camera, TransformPipeline, NumericalAnalysis
from milestone3_core import PhongShader, ProceduralTexture, Sampler, ArtifactAnalysis

pv.global_theme.allow_empty_mesh = True

print("=" * 58)
print("  BIT 2325 — Milestone 3: Rendering & Signal Processing")
print("  Interactive Scene")
print("=" * 58)

# =============================================================
# TERRAIN GENERATION
# =============================================================
print("\n[Person A+B] Generating terrain and textures...")

terrain = Terrain(width=200, depth=200, scale=6.0,
                  height_scale=2.8, octaves=8,
                  persistence=0.52, lacunarity=2.1, seed=77)
W, D    = terrain.width, terrain.depth
hgrid   = np.array(terrain.height_grid())
X_lin   = np.linspace(-2, 2, W)
Z_lin   = np.linspace(-2, 2, D)
XX, ZZ  = np.meshgrid(X_lin, Z_lin)
stats   = terrain.stats()

# Vertex positions and normals as numpy arrays
vert_pos = np.zeros((D*W, 3))
vert_nor = np.zeros((D*W, 3))
for j in range(D):
    for i in range(W):
        v = terrain.get_vertex(i, j)
        n = terrain.get_normal(i, j)
        vert_pos[j*W+i] = [v.x*2, v.y, v.z*2]
        vert_nor[j*W+i] = [n.x, n.y, n.z]

# =============================================================
# COLOUR MODES
# =============================================================

# Mode 1: Biome flat colours (Milestone 2 style)
def biome_colours():
    BIOMES = {
        'deep_water':    [0.08,0.18,0.38], 'shallow_water':[0.14,0.38,0.60],
        'sand':          [0.85,0.72,0.50], 'grass':        [0.32,0.50,0.18],
        'forest':        [0.15,0.32,0.10], 'rock':         [0.50,0.46,0.40],
        'snow':          [0.94,0.94,0.96],
    }
    cols = np.zeros((D*W, 3))
    for j in range(D):
        for i in range(W):
            bm = terrain.classify_biome(hgrid[j,i])
            cols[j*W+i] = BIOMES[bm]
    return cols

# Mode 2: Procedural textures (Milestone 3 — Person B)
def procedural_colours():
    print("  [Person B] Computing procedural textures...")
    tex = ProceduralTexture(seed=42)
    return tex.apply_to_terrain(terrain, hgrid, XX, ZZ)

# Mode 3: Phong-shaded colours (Person A)
def phong_colours(base_cols, eye_pos):
    print("  [Person A] Computing Phong shading...")
    shader = PhongShader()
    shader.add_light(Vector3(8, 12, 6),  np.array([1.0,0.95,0.85]), 1.2)
    shader.add_light(Vector3(-5, 8, -4), np.array([0.55,0.68,1.0]),  0.4)
    shader.add_light(Vector3(0, -3, 0),  np.array([0.9,0.7,0.5]),   0.15)
    shaded = shader.shade_array(
        vert_pos, vert_nor, base_cols, eye_pos,
        material_props={'ka':0.28,'kd':0.72,'ks':0.38,'shininess':28}
    )
    return shaded

print("  Computing biome colours...")
cols_biome = biome_colours()
print("  Computing procedural textures...")
cols_proc  = procedural_colours()

# Phong on procedural (eye at default position)
default_eye = Vector3(5.0, 4.5, 7.0)
print("  Computing Phong shading on procedural textures...")
cols_phong  = phong_colours(cols_proc, default_eye)

print(f"  Vertex count: {D*W:,}")

# =============================================================
# MESH BUILDER
# =============================================================

def build_mesh(colours):
    grid = pv.StructuredGrid(XX*2, hgrid, ZZ*2)
    grid['colours'] = np.clip(colours, 0, 1)
    grid['height']  = hgrid.flatten(order='C')
    grid.point_data['normals'] = vert_nor
    grid.point_data.active_normals_name = 'normals'
    return grid

mesh_biome = build_mesh(cols_biome)
mesh_proc  = build_mesh(cols_proc)
mesh_phong = build_mesh(cols_phong)

# Shading mode state
shade_modes = ['Flat Biome (M2)', 'Procedural Texture (M3-B)', 'Phong+Texture (M3-A)']
shade_meshes= [mesh_biome, mesh_proc, mesh_phong]
shade_idx   = [2]   # start on Phong+Texture

# =============================================================
# VEGETATION (same as Milestone 2 scene)
# =============================================================
print("\n[Person B] Placing vegetation...")

def place_trees(count=280, seed=42):
    rng=np.random.default_rng(seed); meshes=[]
    placed=0; att=0
    while placed<count and att<count*12:
        att+=1
        gi=rng.integers(2,W-2); gj=rng.integers(2,D-2)
        h=hgrid[gj,gi]; hn=h/terrain.height_scale
        ny=float(vert_nor[gj*W+gi][1])
        if not(0.18<hn<0.52) or ny<0.75: continue
        wx,wz=float(XX[gj,gi]),float(ZZ[gj,gi])
        th=rng.uniform(0.10,0.22); tr=rng.uniform(0.008,0.018)
        cr=rng.uniform(0.06,0.14)
        g_v=rng.uniform(0.28,0.42); b_v=rng.uniform(0.05,0.14)
        can_col=(rng.uniform(0.05,0.16),g_v,b_v)
        trk_col=(rng.uniform(0.28,0.38),rng.uniform(0.20,0.28),0.10)
        by=h
        trunk=pv.Cylinder(center=(wx,by+th/2,wz),direction=(0,1,0),radius=tr,height=th,resolution=6)
        c1=pv.Sphere(radius=cr,center=(wx,by+th+cr*0.6,wz),theta_resolution=8,phi_resolution=8)
        c2=pv.Sphere(radius=cr*0.7,center=(wx,by+th+cr*1.1,wz),theta_resolution=8,phi_resolution=8)
        meshes.append(('trunk',trunk,trk_col)); meshes.append(('can',c1,can_col))
        meshes.append(('can2',c2,can_col)); placed+=1
    print(f"    Trees: {placed}"); return meshes

def place_cacti(count=180, seed=99):
    rng=np.random.default_rng(seed); meshes=[]
    placed=0; att=0
    while placed<count and att<count*10:
        att+=1
        gi=rng.integers(2,W-2); gj=rng.integers(2,D-2)
        h=hgrid[gj,gi]; hn=h/terrain.height_scale
        if not(-0.05<hn<0.17) or float(vert_nor[gj*W+gi][1])<0.80: continue
        wx,wz=float(XX[gj,gi]),float(ZZ[gj,gi])
        col=(rng.uniform(0.55,0.72),rng.uniform(0.65,0.80),rng.uniform(0.08,0.18))
        th=rng.uniform(0.06,0.14); tr=rng.uniform(0.006,0.012); by=h
        trunk=pv.Cylinder(center=(wx,by+th/2,wz),direction=(0,1,0),radius=tr,height=th,resolution=5)
        meshes.append(('cactus',trunk,col))
        if th>0.09:
            for side in [-1,1]:
                al=rng.uniform(0.025,0.045); as_=by+th*rng.uniform(0.5,0.75)
                ha=pv.Cylinder(center=(wx+side*al/2,as_,wz),direction=(1,0,0),radius=tr*0.75,height=al,resolution=5)
                vt=pv.Cylinder(center=(wx+side*al,as_+al/2,wz),direction=(0,1,0),radius=tr*0.65,height=al,resolution=5)
                meshes.append(('arm',ha,col)); meshes.append(('tip',vt,col))
        placed+=1
    print(f"    Cacti: {placed}"); return meshes

def place_rocks(count=120, seed=55):
    rng=np.random.default_rng(seed); meshes=[]
    placed=0; att=0
    while placed<count and att<count*10:
        att+=1
        gi=rng.integers(2,W-2); gj=rng.integers(2,D-2)
        h=hgrid[gj,gi]; hn=h/terrain.height_scale
        if not(0.05<hn<0.70): continue
        wx,wz=float(XX[gj,gi]),float(ZZ[gj,gi])
        r=rng.uniform(0.012,0.055)
        rock=pv.Sphere(radius=r,center=(wx,h+r*0.45,wz),theta_resolution=8,phi_resolution=6)
        pts=rock.points.copy(); pts[:,1]=(pts[:,1]-(h+r*0.45))*0.55+(h+r*0.45)
        ang=rng.uniform(0,360); ca,sa=math.cos(math.radians(ang)),math.sin(math.radians(ang))
        px_r=pts[:,0]*ca-pts[:,2]*sa; pz_r=pts[:,0]*sa+pts[:,2]*ca
        pts[:,0]=px_r; pts[:,2]=pz_r; rock.points=pts
        gr=rng.uniform(0.38,0.56); w=rng.uniform(0.00,0.06)
        col=tuple(np.clip([gr+w,gr,gr-w*0.5],0,1))
        meshes.append(('rock',rock,col)); placed+=1
    print(f"    Rocks: {placed}"); return meshes

tree_meshes  = place_trees()
cacti_meshes = place_cacti()
rock_meshes  = place_rocks()

# =============================================================
# WATER + SKY
# =============================================================
SEA_LEVEL = -0.12
water_plane = pv.Plane(center=(0,SEA_LEVEL,0),direction=(0,1,0),
                        i_size=4,j_size=4,i_resolution=80,j_resolution=80)
wpts = water_plane.points.copy()
for k in range(len(wpts)):
    x,z=wpts[k,0],wpts[k,2]
    wpts[k,1]+=(math.sin(x*4.1+0.3)*math.cos(z*3.7+0.8))*0.008
water_plane.points = wpts

sky=pv.Sphere(radius=18,theta_resolution=24,phi_resolution=24)
sky_cols=np.zeros((len(sky.points),3))
for k,pt in enumerate(sky.points):
    yn=(pt[1]+18)/36
    if yn>0.5:
        t=(yn-0.5)/0.5; sky_cols[k]=[0.28+t*0.37,0.48+t*0.30,0.80+t*0.12]
    else:
        t=yn/0.5; sky_cols[k]=[0.65+t*0.23,0.78-t*0.30,0.62+t*0.30]
sky['sky_rgb']=sky_cols

# =============================================================
# LIGHTING MODES (enhanced for Milestone 3)
# =============================================================
LIGHTING_MODES = [
    {
        'name':'Golden Hour',
        'bg_top':'#3a5a8a','bg_bottom':'#e8c090',
        'lights':[
            {'pos':(12,4,-8),'color':'#FFD580','intensity':1.2,'focal':(0,0,0)},
            {'pos':(-5,10,5),'color':'#AAC8FF','intensity':0.45,'focal':(0,0,0)},
            {'pos':(0,-3,0), 'color':'#FFB870','intensity':0.20,'focal':(0,0,0)},
        ],
    },
    {
        'name':'Midday Sun',
        'bg_top':'#1a3a6a','bg_bottom':'#a8c8f0',
        'lights':[
            {'pos':(2,15,3),  'color':'#FFFAF0','intensity':1.4,'focal':(0,0,0)},
            {'pos':(-8,6,-4), 'color':'#C0D8FF','intensity':0.35,'focal':(0,0,0)},
            {'pos':(0,-2,0),  'color':'#E8E4D0','intensity':0.15,'focal':(0,0,0)},
        ],
    },
    {
        'name':'Sunset',
        'bg_top':'#1a1a3a','bg_bottom':'#FF6030',
        'lights':[
            {'pos':(14,1,0),  'color':'#FF7030','intensity':1.3,'focal':(0,0,0)},
            {'pos':(-6,8,4),  'color':'#6040AA','intensity':0.30,'focal':(0,0,0)},
            {'pos':(0,-2,0),  'color':'#FF5520','intensity':0.25,'focal':(0,0,0)},
        ],
    },
    {
        'name':'Overcast',
        'bg_top':'#606875','bg_bottom':'#A8B0B8',
        'lights':[
            {'pos':(0,14,0),  'color':'#D0D8E8','intensity':0.85,'focal':(0,0,0)},
            {'pos':(8,6,8),   'color':'#C8D0D8','intensity':0.40,'focal':(0,0,0)},
            {'pos':(-8,6,-8), 'color':'#C8D0D8','intensity':0.40,'focal':(0,0,0)},
        ],
    },
]
lighting_idx=[0]

PRESETS = {
    '1':{'name':'Overview',     'pos':(5.0,4.5,7.0),'focal':(0.0,0.3,0.0),'fov':58},
    '2':{'name':'Ground level', 'pos':(2.5,0.2,4.0),'focal':(0.0,0.5,0.0),'fov':72},
    '3':{'name':'Aerial',       'pos':(0.0,10.0,0.2),'focal':(0.0,0.0,0.0),'fov':50},
    '4':{'name':'Telephoto',    'pos':(9.0,3.0,1.0),'focal':(0.0,0.5,0.0),'fov':32},
    '5':{'name':'Mountain',     'pos':(1.5,2.5,-4.0),'focal':(0.0,1.0,0.0),'fov':65},
    '6':{'name':'Water edge',   'pos':(1.8,0.1,1.5),'focal':(-0.5,-0.05,-0.5),'fov':78},
}

# =============================================================
# BUILD PLOTTER
# =============================================================
print("\nLaunching interactive scene...")

pl = pv.Plotter(
    title="BIT 2325 — Milestone 3: Phong + Procedural Textures | Interactive",
    window_size=[1400,900],
)
pl.set_background(LIGHTING_MODES[0]['bg_bottom'], top=LIGHTING_MODES[0]['bg_top'])
pl.add_mesh(sky, scalars='sky_rgb', rgb=True, smooth_shading=True,
            show_edges=False, lighting=False, opacity=1.0)

# Start with Phong+Texture mode
terrain_actor = pl.add_mesh(
    shade_meshes[shade_idx[0]],
    scalars='colours', rgb=True, smooth_shading=True,
    show_edges=False, lighting=True,
    specular=0.25, specular_power=20, name='terrain'
)
pl.add_mesh(water_plane, color='#2266AA', opacity=0.62,
            smooth_shading=True, specular=1.0, specular_power=60,
            lighting=True, name='water')

# Vegetation
veg_actors=[]
for i,(kind,mesh,col) in enumerate(tree_meshes+cacti_meshes+rock_meshes):
    a=pl.add_mesh(mesh,color=col,smooth_shading=True,lighting=True,
                  specular=0.05,specular_power=8,name=f'veg_{i}')
    veg_actors.append((f'veg_{i}',a))

def apply_lighting(pl,idx):
    mode=LIGHTING_MODES[idx%len(LIGHTING_MODES)]
    pl.remove_all_lights()
    for lc in mode['lights']:
        pl.add_light(pv.Light(position=lc['pos'],focal_point=lc['focal'],
                               color=lc['color'],intensity=lc['intensity'],positional=False))
    pl.set_background(mode['bg_bottom'],top=mode['bg_top'])
    print(f"\n  Lighting: {mode['name']}")

def apply_preset(pl,key):
    cfg=PRESETS[key]
    cam=Camera(Vector3(*cfg['pos']),Vector3(*cfg['focal']),Vector3(0,1,0),fov_deg=cfg['fov'])
    _=cam.view_matrix; _=cam.projection_matrix
    pl.camera.position=cfg['pos']; pl.camera.focal_point=cfg['focal']
    pl.camera.up=(0,1,0); pl.camera.view_angle=float(cfg['fov'])
    f=1/math.tan(math.radians(cfg['fov']/2))
    print(f"\n  [{cfg['name']}] FOV={cfg['fov']}° f={f:.3f}")

apply_lighting(pl,0); apply_preset(pl,'1')

# =============================================================
# CALLBACKS
# =============================================================
show_normals=[False]; normals_actor=[None]
show_wf=[False]; show_veg=[True]

def cycle_shading():
    """Cycle: Flat biome → Procedural texture → Phong+Texture"""
    shade_idx[0]=(shade_idx[0]+1)%3
    pl.remove_actor('terrain')
    pl.add_mesh(shade_meshes[shade_idx[0]],scalars='colours',rgb=True,
                smooth_shading=True,show_edges=False,lighting=True,
                specular=0.25,specular_power=20,name='terrain')
    print(f"\n  Shading mode: {shade_modes[shade_idx[0]]}")
    pl.render()

def toggle_normals():
    show_normals[0]=not show_normals[0]
    if show_normals[0]:
        step=10; starts=[]; ends=[]
        for j in range(0,D,step):
            for i in range(0,W,step):
                v=terrain.get_vertex(i,j); n=terrain.get_normal(i,j); sc=0.10
                starts.append([v.x*2,v.y,v.z*2])
                ends.append([v.x*2+n.x*sc,v.y+n.y*sc,v.z*2+n.z*sc])
        pts_flat=[]
        for s,e in zip(starts,ends): pts_flat.extend([s,e])
        lines=pv.MultipleLines(points=np.array(pts_flat))
        normals_actor[0]=pl.add_mesh(lines,color='red',line_width=1.5,name='normals')
        print("\n  Normals: ON")
    else:
        if normals_actor[0]: pl.remove_actor('normals')
        print("\n  Normals: OFF")
    pl.render()

def toggle_wireframe():
    show_wf[0]=not show_wf[0]
    pl.remove_actor('terrain')
    style='wireframe' if show_wf[0] else 'surface'
    pl.add_mesh(shade_meshes[shade_idx[0]],scalars='colours',rgb=True,
                smooth_shading=True,style=style,lighting=True,name='terrain')
    print(f"\n  Wireframe: {'ON' if show_wf[0] else 'OFF'}")
    pl.render()

def toggle_vegetation():
    show_veg[0]=not show_veg[0]
    for name,actor in veg_actors: actor.SetVisibility(show_veg[0])
    print(f"\n  Vegetation: {'ON' if show_veg[0] else 'OFF'}")
    pl.render()

def cycle_lighting():
    lighting_idx[0]=(lighting_idx[0]+1)%len(LIGHTING_MODES)
    apply_lighting(pl,lighting_idx[0]); pl.render()

def screenshot():
    fname="m3_screenshot.png"; pl.screenshot(fname)
    print(f"\n  Screenshot: {fname}")

def print_artifacts():
    """Person D — print artifact analysis to terminal"""
    print("\n[Person D] --- Artifact Analysis ---")
    sig=lambda x: math.sin(x*20)*0.5+0.5
    res=ArtifactAnalysis.compare_sampling_strategies(sig,(0,1))
    print("  Sampling strategy comparison:")
    for name,r in res.items():
        print(f"    {name:<15} RMSE={r['rmse']:.5f}  samples={r['samples']}")
    nyq=Sampler.nyquist_check(200*200,1920)
    print(f"\n  Nyquist check (terrain 200×200 @ 1920px):")
    print(f"    Min sample rate: {nyq['min_sample_rate']}")
    print(f"    Supersample needed: {nyq['supersample']}×")
    print(f"    Satisfies Nyquist: {nyq['satisfies_nyquist']}")

def reset_cam(): apply_preset(pl,'1'); pl.render()

for key,fn in [('x','cycle_shading'),('X','cycle_shading'),
               ('n','toggle_normals'),('N','toggle_normals'),
               ('w','toggle_wireframe'),('W','toggle_wireframe'),
               ('t','toggle_vegetation'),('T','toggle_vegetation'),
               ('l','cycle_lighting'),('L','cycle_lighting'),
               ('s','screenshot'),('S','screenshot'),
               ('i','print_artifacts'),('I','print_artifacts'),
               ('r','reset_cam'),('R','reset_cam')]:
    pl.add_key_event(key,locals()[fn])

for k in PRESETS:
    def _mk(pk):
        def _f(): apply_preset(pl,pk); pl.render()
        return _f
    pl.add_key_event(k,_mk(k))

# =============================================================
# HUD
# =============================================================
hud=(
    "BIT 2325 — Milestone 3: Rendering & Signal Processing\n"
    "MOUSE:  Left=rotate  Right=zoom  Middle=pan\n\n"
    "KEYS:\n"
    "  1-6  Camera presets\n"
    "  X    Shading: Flat / Procedural / Phong+Tex\n"
    "  L    Lighting modes\n"
    "  N    Surface normals\n"
    "  W    Wireframe\n"
    "  T    Toggle vegetation\n"
    "  I    Print artifact analysis\n"
    "  S    Screenshot\n"
    "  R    Reset camera  |  Q Quit"
)
pl.add_text(hud,position='lower_left',font_size=9,
            color='white',font='courier',shadow=True)

pipeline=TransformPipeline()
cam_ref=Camera(Vector3(5,4.5,7),Vector3(0,0.3,0),Vector3(0,1,0),fov_deg=58)
pipeline.set_view(cam_ref.view_matrix); pipeline.set_projection(cam_ref.projection_matrix)
analysis=NumericalAnalysis.analyse_pipeline(pipeline)

stats_text=(
    f"MILESTONE 3 SYSTEM\n"
    f"  Shading  : {shade_modes[shade_idx[0]]}\n"
    f"  Vertices : {D*W:,}\n"
    f"  Octaves  : 8  Seed: 77\n\n"
    f"PHONG MODEL  (Person A)\n"
    f"  c = ka*La + kd*Ld*(n·l) + ks*Ls*(r·e)^p\n"
    f"  ka=0.28  kd=0.72  ks=0.38  p=28\n\n"
    f"PROCEDURAL TEXTURES  (Person B)\n"
    f"  Sand: ripple+grain+streak noise\n"
    f"  Grass: patch+blade+dry fBm\n"
    f"  Rock: strata+crack network\n"
    f"  Snow: grain+hollow shadows\n\n"
    f"PIPELINE\n"
    f"  MVP κ = {analysis['mvp']['condition']:.4f}  (well conditioned)\n"
    f"  Person C: AA | Person D: Artifacts"
)
pl.add_text(stats_text,position='upper_right',font_size=8.5,
            color='#aaddff',font='courier',shadow=True)

print("\nCONTROLS:")
print("  X    → cycle shading modes (Flat / Procedural / Phong+Texture)")
print("  L    → cycle lighting")
print("  I    → print artifact analysis to terminal")
print("  1-6  → camera presets")
print("  N/W/T/S/R/Q → normals/wireframe/veg/screenshot/reset/quit")
print("─"*58)
print("Window launching...")

pl.show(auto_close=False)
print("\nScene closed.")
