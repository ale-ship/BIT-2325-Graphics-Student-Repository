# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 5 — Dynamics & Animation  |  Milestone 6 overlays
# Topic: Procedural Content Generation Systems
#
# Authors:  Esther Achieng Otieno  (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu   (SCT221-C004-0217/2024)
#           Wendy Wachira          (SCT221-C004-0194/2024)
#           Alexander Somba        (SCT221-C004-0680/2023)
# Date  : May 2026 | JKUAT
#
# HOW TO RUN:
#   python3 milestone5_scene.py
#
# WHAT IS NEW (on top of Milestones 1–4):
#   M (Person A) — Start / stop terrain morph (C² spline)
#   +/-           — Scrub morph time manually
#   P (Person B) — Cycle particle system  (off/leaves/dust/embers/all)
#   C (Person C) — Drop wave disturbance at centre
#   D (Person D) — Toggle deform mode (left-click to raise)
#   Shift+D       — Deform crater (left-click to lower)
#   A  (M6)      — Toggle Adaptive Fractal Noise octave overlay
#   E  (M6)      — Step ecosystem simulation + show biome colours
#
# ALL MILESTONE 2 KEYS STILL WORK:
#   1-6  Camera presets  |  L  Lighting  |  N  Normals
#   W  Wireframe         |  T  Vegetation|  S  Screenshot
#   R  Reset camera      |  Q  Quit
# =============================================================

import sys, os, math, time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core    import Vector3, Matrix4, Terrain, PerlinNoise
from milestone2_core import Camera, TransformPipeline, NumericalAnalysis
from milestone5_core import (KeyframeAnimator, ParticleSystem, WaveSimulator,
                             TerrainDeformer, PTYPE_LEAF, PTYPE_DUST, PTYPE_EMBER)
from milestone6_core import AdaptiveFractalNoise, EcosystemSimulator

pv.global_theme.allow_empty_mesh = True


# =============================================================
# SECTION 1 — TERRAIN GENERATION  (Person A — unchanged from M2)
# =============================================================

print("=" * 58)
print("  BIT 2325 — Milestone 5 + 6: Dynamics & Research")
print("  Procedural Content Generation System")
print("=" * 58)

SEED   = 77
W, D   = 200, 200
HSCALE = 2.8
SCALE  = 6.0

print("\n[M1] Generating base terrain (seed=77)...")
terrain = Terrain(width=W, depth=D, scale=SCALE,
                  height_scale=HSCALE, octaves=8,
                  persistence=0.52, lacunarity=2.1, seed=SEED)
stats  = terrain.stats()
hgrid  = np.array(terrain.height_grid(), dtype=np.float64)  # (D,W)

X_lin = np.linspace(-2, 2, W)
Z_lin = np.linspace(-2, 2, D)
XX, ZZ = np.meshgrid(X_lin, Z_lin)

vert_nor = np.zeros((D * W, 3))
for j in range(D):
    for i in range(W):
        n = terrain.get_normal(i, j)
        vert_nor[j*W+i] = [n.x, n.y, n.z]

print(f"  {W}×{D} = {stats['vertex_count']:,} vertices  "
      f"h=[{stats['min_height']:.3f}, {stats['max_height']:.3f}]")


# =============================================================
# FAST NUMPY-VECTORIZED COLOURING  (replaces M2 Python loop)
# =============================================================

def _vblend(c1, c2, t_arr):
    t_arr = np.clip(t_arr, 0, 1)[:, None]
    return c1[None, :] * (1 - t_arr) + c2[None, :] * t_arr

def terrain_colours_fast(hgrid_in, normals_in, XX_in, ZZ_in, height_scale=HSCALE):
    """
    Numpy-vectorized terrain colouring — identical colour rules to M2
    but operates on entire (D*W,) arrays at once for real-time updates.
    """
    h  = hgrid_in.flatten(order='C') / height_scale
    ny = normals_in[:, 1]
    wx = XX_in.flatten(order='C')
    wz = ZZ_in.flatten(order='C')
    noise = (np.sin(wx * 8.3 + wz * 5.7) * 0.5 +
             np.cos(wx * 3.1 - wz * 9.2) * 0.5) * 0.04

    # Reference colours (same as M2 terrain_colour())
    DEEP_WATER    = np.array([0.08, 0.18, 0.38])
    SHALLOW_WATER = np.array([0.14, 0.38, 0.60])
    WET_SAND      = np.array([0.72, 0.60, 0.40])
    DRY_SAND      = np.array([0.85, 0.72, 0.50])
    WARM_SAND     = np.array([0.90, 0.68, 0.42])
    SCRUB_GRASS   = np.array([0.45, 0.42, 0.22])
    GRASS         = np.array([0.32, 0.50, 0.18])
    FOREST        = np.array([0.12, 0.28, 0.08])
    ROCK          = np.array([0.52, 0.47, 0.40])
    DARK_ROCK     = np.array([0.35, 0.30, 0.25])
    SNOW          = np.array([0.92, 0.93, 0.95])

    N   = len(h)
    out = np.zeros((N, 3), dtype=np.float64)

    masks_and_colours = [
        (h < -0.35,
         lambda m: np.tile(DEEP_WATER, (m.sum(), 1))),
        ((h >= -0.35) & (h < -0.10),
         lambda m: _vblend(DEEP_WATER, SHALLOW_WATER, (h[m]+0.35)/0.25)),
        ((h >= -0.10) & (h <  0.00),
         lambda m: _vblend(WET_SAND,   DRY_SAND,      (h[m]+0.10)/0.10)),
        ((h >=  0.00) & (h <  0.08),
         lambda m: _vblend(DRY_SAND,   WARM_SAND,     h[m]/0.08 + noise[m])),
        ((h >=  0.08) & (h <  0.18),
         lambda m: _vblend(WARM_SAND,  SCRUB_GRASS,   (h[m]-0.08)/0.10 + noise[m])),
        ((h >=  0.18) & (h <  0.32),
         lambda m: _vblend(SCRUB_GRASS,GRASS,          (h[m]-0.18)/0.14)),
        ((h >=  0.32) & (h <  0.48),
         lambda m: _vblend(GRASS,      FOREST,         (h[m]-0.32)/0.16)),
        ((h >=  0.48) & (h <  0.62),
         lambda m: _vblend(FOREST,     ROCK,           (h[m]-0.48)/0.14)),
        ((h >=  0.62) & (h <  0.78),
         lambda m: _vblend(ROCK,       DARK_ROCK,      (h[m]-0.62)/0.16)),
        (h >=  0.78,
         lambda m: _vblend(DARK_ROCK,  SNOW,           (h[m]-0.78)/0.22)),
    ]
    for mask_cond, colour_fn in masks_and_colours:
        if np.any(mask_cond):
            out[mask_cond] = colour_fn(mask_cond)

    # Slope darkening (steep faces → rock colour)
    slope_mask = (ny < 0.6) & (h > 0.15)
    if np.any(slope_mask):
        t_sl = np.clip(ny[slope_mask] / 0.6, 0, 1)
        out[slope_mask] = _vblend(DARK_ROCK, out[slope_mask], t_sl)

    # Ambient occlusion approximation
    ao = np.clip(0.75 + h * 0.5, 0.75, 1.0)
    out *= ao[:, None]
    return np.clip(out, 0, 1).astype(np.float32)


def compute_normals_fast(hgrid_2d, dx=0.04):
    """Numpy-gradient surface normals — replaces M2's per-vertex loop."""
    gz, gx = np.gradient(hgrid_2d, dx)
    normals = np.stack([-gx, np.ones_like(gx), -gz], axis=-1)
    mag     = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / (mag + 1e-10)).reshape(-1, 3).astype(np.float32)


print("[M2] Computing initial colours (vectorized)...")
colours  = terrain_colours_fast(hgrid, vert_nor, XX, ZZ)
vert_nor_np = vert_nor.astype(np.float32)


# =============================================================
# SECTION 2 — TERRAIN PYVISTA MESH  (same structure as M2)
# =============================================================

terrain_grid = pv.StructuredGrid(XX * 2, hgrid, ZZ * 2)
terrain_grid['colours'] = colours
terrain_grid['height']  = hgrid.flatten(order='C').astype(np.float32)
terrain_grid.point_data['normals'] = vert_nor_np
terrain_grid.point_data.active_normals_name = 'normals'


# =============================================================
# SECTION 3 — M5 SYSTEM INITIALISATION
# =============================================================

# ── Person A: KeyframeAnimator ────────────────────────────────
print("\n[M5-A] Building KeyframeAnimator (4 keyframes × seeds)...")
animator = KeyframeAnimator(width=W, depth=D, scale=SCALE,
                            height_scale=HSCALE, octaves=8,
                            persistence=0.52, lacunarity=2.1)
KEYFRAME_SEEDS = [(0.0, 77), (3.5, 42), (7.0, 13), (10.5, 77)]
for t, s in KEYFRAME_SEEDS:
    animator.add_keyframe(t, s)
    print(f"   Keyframe t={t:.1f}s  seed={s}")
animator.build_splines()
print(f"   Splines built. Duration={animator.duration:.1f}s  "
      f"(looping C² interpolation)")

# Precompute colours for start/end keyframes (for faster interpolation)
_H0_flat  = animator.get_frame(0.0).flatten(order='C')
_H35_flat = animator.get_frame(3.5).flatten(order='C')

morph_active = [False]
morph_t      = [0.0]          # current animation time
morph_start  = [0.0]          # wall-clock time when morph started

# ── Person B: ParticleSystem ──────────────────────────────────
print("[M5-B] Initialising ParticleSystem (800 max particles)...")
ps = ParticleSystem(terrain_hgrid=hgrid, world_extent=4.0,
                    max_particles=800, seed=42)

PARTICLE_MODES = ['off', 'leaves', 'dust', 'embers', 'all']
particle_mode  = [0]   # index into PARTICLE_MODES

# Fixed-size particle buffer (dead particles hidden at y=-100)
_part_pos  = np.zeros((ps.N, 3), dtype=np.float32)
_part_pos[:, 1] = -100.0
_part_col  = np.zeros((ps.N, 3), dtype=np.float32)
particle_pd = pv.PolyData(_part_pos)
particle_pd['colours'] = _part_col

# ── Person C: WaveSimulator ───────────────────────────────────
print("[M5-C] Initialising WaveSimulator (64×64 grid)...")
ws = WaveSimulator(size=64, wave_speed=1.2, dt=0.016,
                   damping=0.996, seed=77)
ws.add_drop(32, 32, amplitude=0.06)
ws.add_drop(20, 44, amplitude=0.04)

SEA_LEVEL = -0.12
N_wave    = ws.N
XW_lin    = np.linspace(-2.0, 2.0, N_wave)
ZW_lin    = np.linspace(-2.0, 2.0, N_wave)
XW, ZW    = np.meshgrid(XW_lin, ZW_lin)

disp0 = ws.get_displacement() * 0.06
water_grid = pv.StructuredGrid(XW, SEA_LEVEL + disp0, ZW)
water_grid['wave_h'] = (SEA_LEVEL + disp0).flatten(order='C').astype(np.float32)

# ── Person D: TerrainDeformer ─────────────────────────────────
print("[M5-D] Initialising TerrainDeformer...")
deformer      = TerrainDeformer(width=W, depth=D, world_extent=4.0)
deform_mode   = [False]   # whether left-click raises terrain
deform_crater = [False]   # Shift+D makes craters

# ── M6: Adaptive Fractal Noise ────────────────────────────────
print("[M6-A+B] Preparing AdaptiveFractalNoise...")
afn           = AdaptiveFractalNoise(width=W, depth=D, scale=SCALE,
                                     height_scale=HSCALE,
                                     oct_min=2, oct_max=10,
                                     curvature_weight=0.55)
afn_overlay   = [False]
_afn_colours  = [None]   # cached octave map colours

# ── M6: EcosystemSimulator ────────────────────────────────────
print("[M6-C+D] Preparing EcosystemSimulator...")
eco           = EcosystemSimulator(terrain, noise_seed_T=17, noise_seed_M=31)
eco_overlay   = [False]
eco_step_n    = [0]

print("\n[M5] All systems initialised.\n")


# =============================================================
# SECTION 4 — LIGHTING (unchanged from M2)
# =============================================================

LIGHTING_MODES = [
    {
        'name'      : 'Golden Hour (warm afternoon)',
        'bg_top'    : '#4a6fa5', 'bg_bottom': '#e8c090',
        'lights': [
            {'pos':(12, 4,-8),  'color':'#FFD580','intensity':1.2,'focal':(0,0,0)},
            {'pos':(-5,10, 5),  'color':'#AAC8FF','intensity':0.45,'focal':(0,0,0)},
            {'pos':(0, -3, 0),  'color':'#FFB870','intensity':0.20,'focal':(0,0,0)},
        ],
        'ambient': 0.35,
        'wind'   : ([1.0, 0.0, 0.3], 0.8),
    },
    {
        'name'      : 'Midday Sun (bright)',
        'bg_top'    : '#1a3a6a', 'bg_bottom': '#a8c8f0',
        'lights': [
            {'pos':(2, 15, 3),  'color':'#FFFAF0','intensity':1.4,'focal':(0,0,0)},
            {'pos':(-8, 6,-4),  'color':'#C0D8FF','intensity':0.35,'focal':(0,0,0)},
            {'pos':(0, -2, 0),  'color':'#E8E4D0','intensity':0.15,'focal':(0,0,0)},
        ],
        'ambient': 0.45,
        'wind'   : ([0.6, 0.0, 0.8], 0.5),
    },
    {
        'name'      : 'Sunset (dramatic)',
        'bg_top'    : '#1a1a3a', 'bg_bottom': '#FF6030',
        'lights': [
            {'pos':(14, 1, 0),  'color':'#FF7030','intensity':1.3,'focal':(0,0,0)},
            {'pos':(-6, 8, 4),  'color':'#6040AA','intensity':0.30,'focal':(0,0,0)},
            {'pos':(0, -2, 0),  'color':'#FF5520','intensity':0.25,'focal':(0,0,0)},
        ],
        'ambient': 0.18,
        'wind'   : ([1.0, 0.0,-0.3], 1.4),
    },
    {
        'name'      : 'Overcast (soft diffuse)',
        'bg_top'    : '#606875', 'bg_bottom': '#A8B0B8',
        'lights': [
            {'pos':(0, 14, 0),  'color':'#D0D8E8','intensity':0.85,'focal':(0,0,0)},
            {'pos':(8,  6, 8),  'color':'#C8D0D8','intensity':0.40,'focal':(0,0,0)},
            {'pos':(-8, 6,-8),  'color':'#C8D0D8','intensity':0.40,'focal':(0,0,0)},
        ],
        'ambient': 0.60,
        'wind'   : ([0.3, 0.0, 0.5], 0.3),
    },
]
lighting_idx = [0]


# =============================================================
# SECTION 5 — CAMERA PRESETS (unchanged from M2)
# =============================================================

PRESETS = {
    '1': {'name':'Overview',       'pos':(5.0, 4.5, 7.0),'focal':(0.0,0.3,0.0),'fov':58},
    '2': {'name':'Ground level',   'pos':(2.5, 0.2, 4.0),'focal':(0.0,0.5,0.0),'fov':72},
    '3': {'name':'Aerial',         'pos':(0.0,10.0, 0.2),'focal':(0.0,0.0,0.0),'fov':50},
    '4': {'name':'Telephoto side', 'pos':(9.0, 3.0, 1.0),'focal':(0.0,0.5,0.0),'fov':32},
    '5': {'name':'Mountain face',  'pos':(1.5, 2.5,-4.0),'focal':(0.0,1.0,0.0),'fov':65},
    '6': {'name':'Water edge',     'pos':(1.8, 0.1, 1.5),'focal':(-0.5,-0.05,-0.5),'fov':78},
}

def apply_preset(pl, key):
    cfg = PRESETS[key]
    cam = Camera(Vector3(*cfg['pos']), Vector3(*cfg['focal']),
                 Vector3(0,1,0), fov_deg=cfg['fov'])
    _ = cam.view_matrix; _ = cam.projection_matrix
    pl.camera.position    = cfg['pos']
    pl.camera.focal_point = cfg['focal']
    pl.camera.up          = (0.0, 1.0, 0.0)
    pl.camera.view_angle  = float(cfg['fov'])
    print(f"\n  [{cfg['name']}]  pos={cfg['pos']}  FOV={cfg['fov']}°")


# =============================================================
# SECTION 6 — VEGETATION (verbatim from M2, Person B)
# =============================================================

def blend(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(float(a)*(1-t) + float(b)*t for a, b in zip(c1, c2))

def place_trees(count=320, seed=42):
    rng = np.random.default_rng(seed)
    meshes = []
    placed = 0; attempts = 0
    while placed < count and attempts < count * 12:
        attempts += 1
        gi = rng.integers(2, W-2); gj = rng.integers(2, D-2)
        h = hgrid[gj, gi]; hn = h / HSCALE
        ny = float(vert_nor[gj*W+gi][1])
        if not (0.18 < hn < 0.52) or ny < 0.75: continue
        wx = XX[gj, gi]; wz = ZZ[gj, gi]
        trunk_h = rng.uniform(0.10, 0.22); trunk_r = rng.uniform(0.008, 0.018)
        canopy_r = rng.uniform(0.06, 0.14)
        lean_x = rng.uniform(-0.02, 0.02); lean_z = rng.uniform(-0.02, 0.02)
        g_val = rng.uniform(0.28, 0.42); b_val = rng.uniform(0.05, 0.14)
        canopy_col = (rng.uniform(0.05, 0.16), g_val, b_val)
        trunk_col  = (rng.uniform(0.28, 0.38), rng.uniform(0.20, 0.28), 0.10)
        base_y = h
        trunk = pv.Cylinder(
            center=(wx+lean_x*trunk_h/2, base_y+trunk_h/2, wz+lean_z*trunk_h/2),
            direction=(lean_x,1.0,lean_z), radius=trunk_r, height=trunk_h, resolution=6)
        cy1 = base_y+trunk_h+canopy_r*0.6; cy2 = base_y+trunk_h+canopy_r*1.1
        canopy1 = pv.Sphere(radius=canopy_r, center=(wx+lean_x,cy1,wz+lean_z),
                            theta_resolution=8, phi_resolution=8)
        canopy2 = pv.Sphere(radius=canopy_r*0.7, center=(wx+lean_x*0.5,cy2,wz+lean_z*0.5),
                            theta_resolution=8, phi_resolution=8)
        meshes += [('trunk',trunk,trunk_col),('canopy',canopy1,canopy_col),
                   ('canopy2',canopy2,canopy_col)]
        placed += 1
    return meshes

def place_cacti(count=200, seed=99):
    rng = np.random.default_rng(seed)
    meshes = []; placed = 0; attempts = 0
    while placed < count and attempts < count * 10:
        attempts += 1
        gi = rng.integers(2, W-2); gj = rng.integers(2, D-2)
        h = hgrid[gj, gi]; hn = h / HSCALE
        ny = float(vert_nor[gj*W+gi][1])
        if not (-0.05 < hn < 0.17) or ny < 0.80: continue
        wx = XX[gj, gi]; wz = ZZ[gj, gi]
        col = (rng.uniform(0.55,0.72), rng.uniform(0.65,0.80), rng.uniform(0.08,0.18))
        trunk_h = rng.uniform(0.06, 0.14); trunk_r = rng.uniform(0.006, 0.012)
        trunk = pv.Cylinder(center=(wx,h+trunk_h/2,wz), direction=(0,1,0),
                            radius=trunk_r, height=trunk_h, resolution=5)
        meshes.append(('cactus', trunk, col))
        if trunk_h > 0.09:
            for side in [-1,1]:
                arm_len = rng.uniform(0.025, 0.045)
                arm_start = h + trunk_h * rng.uniform(0.5, 0.75)
                h_arm = pv.Cylinder(center=(wx+side*arm_len/2,arm_start,wz),
                                    direction=(1,0,0), radius=trunk_r*0.75,
                                    height=arm_len, resolution=5)
                v_tip = pv.Cylinder(center=(wx+side*arm_len,arm_start+arm_len/2,wz),
                                    direction=(0,1,0), radius=trunk_r*0.65,
                                    height=arm_len, resolution=5)
                meshes += [('cactus_arm',h_arm,col),('cactus_tip',v_tip,col)]
        placed += 1
    return meshes

def place_rocks(count=150, seed=55):
    rng = np.random.default_rng(seed)
    meshes = []; placed = 0; attempts = 0
    while placed < count and attempts < count * 10:
        attempts += 1
        gi = rng.integers(2, W-2); gj = rng.integers(2, D-2)
        h = hgrid[gj, gi]; hn = h / HSCALE
        if not (0.05 < hn < 0.70): continue
        wx = XX[gj, gi]; wz = ZZ[gj, gi]
        r    = rng.uniform(0.012, 0.055)
        rock = pv.Sphere(radius=r, center=(wx,h+r*0.45,wz),
                         theta_resolution=8, phi_resolution=6)
        pts  = rock.points.copy()
        pts[:,1] = (pts[:,1]-(h+r*0.45))*0.55 + (h+r*0.45)
        angle = rng.uniform(0, 360)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        px_r = pts[:,0]*ca - pts[:,2]*sa; pz_r = pts[:,0]*sa + pts[:,2]*ca
        pts[:,0], pts[:,2] = px_r, pz_r
        rock.points = pts
        grey = rng.uniform(0.38, 0.56); warm = rng.uniform(0.00, 0.06)
        col = (grey+warm, grey, max(0,grey-warm*0.5))
        meshes.append(('rock', rock, col))
        placed += 1
    return meshes

print("[M2] Placing vegetation (verbatim from M2)...")
tree_meshes  = place_trees()
cacti_meshes = place_cacti()
rock_meshes  = place_rocks()


# =============================================================
# SECTION 7 — ATMOSPHERE (verbatim from M2, Person C)
# =============================================================

sky = pv.Sphere(radius=18, theta_resolution=24, phi_resolution=24)
sky_colours = np.zeros((len(sky.points), 3))
for k, pt in enumerate(sky.points):
    y_norm = (pt[1] + 18) / 36
    if y_norm > 0.5:
        t = (y_norm - 0.5) / 0.5
        sky_colours[k] = np.array([0.65,0.78,0.92])*(1-t) + np.array([0.28,0.48,0.80])*t
    else:
        t = y_norm / 0.5
        sky_colours[k] = np.array([0.88,0.76,0.62])*(1-t) + np.array([0.65,0.78,0.92])*t
sky['sky_rgb'] = sky_colours


# =============================================================
# SECTION 8 — BUILD PLOTTER
# =============================================================

print("\n[Scene] Launching plotter...")

pl = pv.Plotter(
    title       = "BIT 2325 — Milestone 5+6 | Dynamics & Research",
    window_size = [1440, 900],
)

mode = LIGHTING_MODES[0]
pl.set_background(mode['bg_bottom'], top=mode['bg_top'])

# Sky
pl.add_mesh(sky, scalars='sky_rgb', rgb=True, smooth_shading=True,
            show_edges=False, lighting=False)

# Terrain
terrain_actor = pl.add_mesh(
    terrain_grid, scalars='colours', rgb=True,
    smooth_shading=True, show_edges=False,
    lighting=True, specular=0.18, specular_power=12,
    name='terrain',
)

# ── GLSL Water Shader (Person C) ──────────────────────────────
# The WaveSimulator computes wave physics (CPU).
# GLSL handles per-fragment specular, normal perturbation, and
# Fresnel-based colour blending on the GPU.

WATER_FRAG_GLSL = """
//VTK::Color::Impl
  // ── GLSL Water Fragment Shader ──────────────────────────
  // Animated per-fragment water with Fresnel + specular.
  //
  // anim_time uniform is updated from Python each frame.
  // vertexMC  = vertex position in model (world) coordinates.
  //
  // Wave-perturbed normal (approximation of du/dx, du/dz):
  float tx = vertexMC.x * 5.8 + anim_time * 2.1;
  float tz = vertexMC.z * 4.4 + anim_time * 1.75;
  float tx2= vertexMC.x * 12.0 - anim_time * 3.3;
  float tz2= vertexMC.z *  9.5 + anim_time * 2.9;
  vec3 wave_n = normalize(vec3(
      -0.07*cos(tx) - 0.04*cos(tx2),
       1.0,
      -0.07*cos(tz) - 0.04*cos(tz2)
  ));

  // Sun direction (matches Golden Hour light)
  vec3 sun_d  = normalize(vec3(12.0, 4.0, -8.0));
  vec3 view_d = normalize(-vertexVCVSOutput.xyz);
  vec3 H      = normalize(sun_d + view_d);

  // Specular (Blinn-Phong, high shininess)
  float spec  = pow(max(dot(wave_n, H), 0.0), 90.0);

  // Fresnel (Schlick approximation)
  //   F(θ) = F0 + (1-F0)*(1-cosθ)^5
  float F0      = 0.04;
  float cos_th  = max(dot(wave_n, view_d), 0.0);
  float fresnel = F0 + (1.0 - F0) * pow(1.0 - cos_th, 5.0);

  // Colour: deep blue → shallow teal by Fresnel
  vec3 deep_col    = vec3(0.03, 0.13, 0.35);
  vec3 shallow_col = vec3(0.10, 0.45, 0.72);
  vec3 sky_refl    = vec3(0.48, 0.68, 0.90);

  // Reflect more sky colour when viewed at grazing angles
  vec3 water_col = mix(
      mix(deep_col, shallow_col, 1.0 - fresnel),
      sky_refl,
      fresnel * 0.5
  );

  // Sun highlight
  water_col += vec3(1.0, 0.98, 0.9) * spec * 0.85;

  ambientColor  = water_col * 0.20;
  diffuseColor  = water_col * 0.70;
  specularColor = vec3(spec * 0.75);
  opacity       = 0.72;
"""

water_actor = pl.add_mesh(
    water_grid, scalars='wave_h',
    cmap='Blues_r', clim=[-0.18, 0.05],
    smooth_shading=True, show_scalar_bar=False,
    opacity=0.70, specular=1.0, specular_power=60,
    lighting=True, name='water',
)

# Apply GLSL shader
glsl_ok        = False
water_uniforms = None
try:
    sp = water_actor.GetShaderProperty()
    sp.AddFragmentShaderReplacement(
        "//VTK::Color::Impl", True, WATER_FRAG_GLSL, False)
    water_uniforms = sp.GetFragmentCustomUniforms()
    water_uniforms.SetUniformf("anim_time", 0.0)
    glsl_ok = True
    print("  [GLSL] Water fragment shader active (Fresnel + specular).")
except Exception as e:
    print(f"  [GLSL] Fallback to vertex colours (VTK {e})")

# ── Particle point cloud ──────────────────────────────────────
particle_actor = pl.add_mesh(
    particle_pd, scalars='colours', rgb=True,
    point_size=8, render_points_as_spheres=True,
    lighting=False, name='particles',
)

# ── Vegetation (verbatim M2 add_object_batch) ─────────────────
veg_actors = []
def add_object_batch(pl, meshes, prefix):
    actors = []
    for i, (kind, mesh, col) in enumerate(meshes):
        spec  = 0.05 if 'canopy' in kind else 0.15
        actor = pl.add_mesh(mesh, color=col, smooth_shading=True,
                            lighting=True, specular=spec, specular_power=8,
                            name=f"{prefix}_{i}")
        actors.append((f"{prefix}_{i}", actor))
    return actors

veg_actors += add_object_batch(pl, tree_meshes,  'tree')
veg_actors += add_object_batch(pl, cacti_meshes, 'cactus')
veg_actors += add_object_batch(pl, rock_meshes,  'rock')


# =============================================================
# SECTION 9 — LIGHTING & APPLY INITIAL PRESET
# =============================================================

def apply_lighting(pl, idx):
    m = LIGHTING_MODES[idx % len(LIGHTING_MODES)]
    pl.remove_all_lights()
    for lc in m['lights']:
        pl.add_light(pv.Light(position=lc['pos'], focal_point=lc['focal'],
                              color=lc['color'], intensity=lc['intensity'],
                              positional=False))
    pl.set_background(m['bg_bottom'], top=m['bg_top'])
    # Update particle wind to match lighting mood
    wd, ws_speed = m['wind']
    ps.set_wind(wd, ws_speed)
    print(f"\n  Lighting: {m['name']}")

apply_lighting(pl, 0)
apply_preset(pl, '1')


# =============================================================
# SECTION 10 — INTERACTIVE CALLBACKS  (M2 keys + M5 keys)
# =============================================================

show_normals   = [False]
show_wireframe = [False]
show_veg       = [True]
normals_actor  = [None]

# ── M2 callbacks (unchanged) ──────────────────────────────────

def toggle_normals():
    show_normals[0] = not show_normals[0]
    if show_normals[0]:
        step = 10; starts, ends = [], []
        for j in range(0, D, step):
            for i in range(0, W, step):
                v = terrain.get_vertex(i, j); n = terrain.get_normal(i, j)
                sc = 0.10
                starts.append([v.x*2, v.y, v.z*2])
                ends.append([v.x*2+n.x*sc, v.y+n.y*sc, v.z*2+n.z*sc])
        pts_flat = []
        for s, e in zip(starts, ends): pts_flat.extend([s, e])
        lines = pv.MultipleLines(points=np.array(pts_flat))
        normals_actor[0] = pl.add_mesh(lines, color='red', line_width=1.5, name='normals')
        print("\n  Normals: ON")
    else:
        if normals_actor[0]: pl.remove_actor('normals')
        print("\n  Normals: OFF")
    pl.render()

def toggle_wireframe():
    show_wireframe[0] = not show_wireframe[0]
    pl.remove_actor('terrain')
    style = 'wireframe' if show_wireframe[0] else 'surface'
    pl.add_mesh(terrain_grid, scalars='colours', rgb=True,
                smooth_shading=True, style=style,
                lighting=True, name='terrain')
    print(f"\n  Wireframe: {'ON' if show_wireframe[0] else 'OFF'}")
    pl.render()

def toggle_vegetation():
    show_veg[0] = not show_veg[0]
    for name, actor in veg_actors: actor.SetVisibility(show_veg[0])
    print(f"\n  Vegetation: {'ON' if show_veg[0] else 'OFF'}")
    pl.render()

def cycle_lighting():
    lighting_idx[0] = (lighting_idx[0] + 1) % len(LIGHTING_MODES)
    apply_lighting(pl, lighting_idx[0]); pl.render()

def screenshot():
    fname = f"m5_scene_{int(time.time())}.png"
    pl.screenshot(fname); print(f"\n  Screenshot: {fname}")

def reset_cam():
    apply_preset(pl, '1'); pl.render()

# ── M5 callbacks ──────────────────────────────────────────────

def toggle_morph():
    """M key — start/stop live terrain morphing."""
    morph_active[0] = not morph_active[0]
    if morph_active[0]:
        morph_start[0] = time.time() - morph_t[0]
        print(f"\n  [M5-A] Morph STARTED  (t={morph_t[0]:.2f}s)")
    else:
        print(f"\n  [M5-A] Morph PAUSED   (t={morph_t[0]:.2f}s)")

def scrub_forward():
    """+ key — advance morph time by 0.25s."""
    morph_t[0] = (morph_t[0] + 0.25) % animator.duration
    _apply_morph_frame(morph_t[0])
    print(f"\n  [M5-A] Scrub → t={morph_t[0]:.2f}s")

def scrub_backward():
    """- key — rewind morph time by 0.25s."""
    morph_t[0] = (morph_t[0] - 0.25) % animator.duration
    _apply_morph_frame(morph_t[0])
    print(f"\n  [M5-A] Scrub ← t={morph_t[0]:.2f}s")

def _apply_morph_frame(t):
    """Core terrain update: spline → heights → normals → colours → GPU."""
    new_hgrid = animator.get_frame(t)                  # (D,W) interpolated heights
    new_norms = compute_normals_fast(new_hgrid)        # (D*W,3)

    # Choose colour source based on overlays
    if afn_overlay[0] and _afn_colours[0] is not None:
        new_cols = _afn_colours[0]
    elif eco_overlay[0]:
        eco_rgb = eco.get_biome_colours().reshape(-1, 3).astype(np.float32)
        new_cols = eco_rgb
    else:
        deform_delta = deformer.get_deformation()
        combined_h   = new_hgrid + deform_delta
        new_cols = terrain_colours_fast(combined_h, new_norms, XX, ZZ)

    # Apply deformer displacement on top of spline
    deform_delta = deformer.get_deformation()
    combined_h   = new_hgrid + deform_delta

    # Update PyVista mesh — only the Y (height) column changes
    pts = terrain_grid.points.copy()
    pts[:, 1] = combined_h.flatten(order='C')
    terrain_grid.points = pts
    terrain_grid['colours'] = new_cols

def cycle_particles():
    """P key — cycle particle type."""
    particle_mode[0] = (particle_mode[0] + 1) % len(PARTICLE_MODES)
    m = PARTICLE_MODES[particle_mode[0]]
    print(f"\n  [M5-B] Particles: {m.upper()}")

def drop_wave():
    """C key — drop wave disturbance at scene centre."""
    ws.add_drop(N_wave//2, N_wave//2, amplitude=0.10, radius=4)
    ws.add_drop(N_wave//3, N_wave//3, amplitude=0.06, radius=3)
    print("\n  [M5-C] Wave disturbance dropped at centre")

def toggle_deform():
    """D key — toggle terrain deform mode (raise on click)."""
    deform_mode[0]   = not deform_mode[0]
    deform_crater[0] = False
    if deform_mode[0]:
        pl.enable_surface_picking(callback=_deform_pick_cb,
                                  left_clicking=True,
                                  show_message=False)
        print("\n  [M5-D] Deform RAISE mode  (left-click terrain to raise)")
    else:
        pl.disable_picking()
        print("\n  [M5-D] Deform mode OFF")

def toggle_deform_crater():
    """Shift+D — toggle crater deform mode."""
    deform_crater[0] = not deform_crater[0]
    deform_mode[0]   = deform_crater[0]
    if deform_crater[0]:
        pl.enable_surface_picking(callback=_deform_pick_cb,
                                  left_clicking=True,
                                  show_message=False)
        print("\n  [M5-D] Deform CRATER mode  (left-click terrain to lower)")
    else:
        pl.disable_picking()
        print("\n  [M5-D] Deform mode OFF")

def _deform_pick_cb(point):
    """Called when user clicks terrain in deform mode."""
    if point is None: return
    wx, _, wz = float(point[0]), float(point[1]), float(point[2])
    amp = -0.45 if deform_crater[0] else 0.45
    deformer.deform(wx, wz, amplitude=amp, sigma=0.30)
    print(f"\n  [M5-D] {'Crater' if amp<0 else 'Bump'} at ({wx:.2f}, {wz:.2f})")

def toggle_afn():
    """A key — toggle Adaptive Fractal Noise octave overlay."""
    afn_overlay[0] = not afn_overlay[0]
    if afn_overlay[0]:
        eco_overlay[0] = False   # mutually exclusive
        cam_pos = pl.camera.position
        print("\n  [M6-A+B] Computing AFN octave map (this takes ~3s)...")
        curv  = AdaptiveFractalNoise.compute_curvature(hgrid)
        dist  = AdaptiveFractalNoise.compute_distance_map(
            W, D, np.array(cam_pos), world_extent=4.0)
        c_n   = curv / (curv.max() + 1e-10)
        d_n   = dist / (dist.max() + 1e-10)
        blend_f = afn.alpha * c_n + (1 - afn.alpha) * (1 - d_n)
        blend_f = np.clip(blend_f, 0, 1)
        oct_map = afn.oct_min + np.round(
            blend_f * (afn.oct_max - afn.oct_min)
        ).astype(int)

        # Map octave count → heatmap colour (blue=low, red=high)
        t_oct = (oct_map - afn.oct_min) / max(afn.oct_max - afn.oct_min, 1)
        t_oct_flat = t_oct.flatten(order='C')
        afn_cols = np.zeros((D*W, 3), dtype=np.float32)
        afn_cols[:, 0] = t_oct_flat         # red channel
        afn_cols[:, 2] = 1 - t_oct_flat     # blue channel
        afn_cols[:, 1] = 0.15               # slight green

        total_a = int(np.sum(oct_map))
        total_f = D * W * afn.oct_max
        saving  = (1 - total_a/total_f)*100

        _afn_colours[0] = afn_cols
        terrain_grid['colours'] = afn_cols
        print(f"  [M6-A+B] Octave range: [{oct_map.min()}, {oct_map.max()}]  "
              f"mean={oct_map.mean():.1f}  saving={saving:.1f}%")
        print("           (Red=max octaves / Blue=min octaves)")
    else:
        _afn_colours[0] = None
        terrain_grid['colours'] = colours   # restore M2 colours
        print("\n  [M6-A+B] AFN overlay OFF")
    pl.render()

def step_ecosystem():
    """E key — advance ecosystem 5 steps and show biome colours."""
    eco_overlay[0] = True
    afn_overlay[0] = False
    eco.step(n=5, dt=0.5)
    eco_step_n[0] += 5
    eco_rgb = eco.get_biome_colours().reshape(-1, 3).astype(np.float32)
    terrain_grid['colours'] = eco_rgb
    print(f"\n  [M6-C+D] Ecosystem step {eco_step_n[0]}")
    print(eco.summary())
    pl.render()

def reset_eco():
    """Shift+E — turn off ecosystem overlay."""
    eco_overlay[0] = False
    terrain_grid['colours'] = colours
    print("\n  [M6-C+D] Ecosystem overlay OFF")
    pl.render()

# Wire up all keys
for key, fn in [
    ('n','toggle_normals'), ('N','toggle_normals'),
    ('w','toggle_wireframe'),('W','toggle_wireframe'),
    ('t','toggle_vegetation'),('T','toggle_vegetation'),
    ('l','cycle_lighting'), ('L','cycle_lighting'),
    ('s','screenshot'),     ('S','screenshot'),
    ('r','reset_cam'),      ('R','reset_cam'),
    ('m','toggle_morph'),   ('M','toggle_morph'),
    ('p','cycle_particles'),('P','cycle_particles'),
    ('c','drop_wave'),      ('C','drop_wave'),
    ('d','toggle_deform'),  ('D','toggle_deform'),
    ('a','toggle_afn'),     ('A','toggle_afn'),
    ('e','step_ecosystem'), ('E','step_ecosystem'),
]:
    pl.add_key_event(key, locals()[fn])

pl.add_key_event('plus',  scrub_forward)
pl.add_key_event('equal', scrub_forward)    # = is + without shift
pl.add_key_event('minus', scrub_backward)

for k in PRESETS:
    def _make_fn(pk):
        def _fn(): apply_preset(pl, pk); pl.render()
        return _fn
    pl.add_key_event(k, _make_fn(k))


# =============================================================
# SECTION 11 — TIMER CALLBACK  (~30fps animation loop)
# =============================================================

_frame_count  = [0]
_anim_start   = [time.time()]
_last_wave_wind = [0.0]

def _update_particles():
    """Update particle positions and colours in the fixed-size buffer."""
    mode_name = PARTICLE_MODES[particle_mode[0]]
    if mode_name == 'off':
        _part_pos[:, 1] = -100.0
        _part_col[:, :]  = 0.0
    else:
        # Filter alive particles by active mode
        all_alive = np.where(ps.alive)[0]
        visible   = np.zeros(ps.N, dtype=bool)
        for i in all_alive:
            pt = int(ps.ptype[i])
            if (mode_name == 'all'
                or (mode_name == 'leaves' and pt == PTYPE_LEAF)
                or (mode_name == 'dust'   and pt == PTYPE_DUST)
                or (mode_name == 'embers' and pt == PTYPE_EMBER)):
                visible[i] = True

        _part_pos[:] = ps.pos.copy()
        _part_pos[~visible, 1] = -100.0  # hide non-visible particles
        _part_col[:] = 0.0
        if np.any(visible):
            vis_idx = np.where(visible)[0]
            _part_col[vis_idx] = ps.get_colours()  # alive colours

    particle_pd.points = _part_pos.copy()
    particle_pd['colours'] = _part_col.copy()

def animation_step():
    """Master timer callback — drives all M5 systems each frame."""
    _frame_count[0] += 1
    now = time.time()
    t_wall = now - _anim_start[0]

    # ── Person A: Terrain morph ───────────────────────────────
    if morph_active[0]:
        morph_t[0] = (now - morph_start[0]) % animator.duration
        _apply_morph_frame(morph_t[0])
    else:
        # Still apply deformer ripples even when morph paused
        deformer.step(dt=0.033)
        if np.any(deformer.get_deformation() != 0):
            _apply_morph_frame(morph_t[0])

    # ── Person B: Particles ───────────────────────────────────
    if PARTICLE_MODES[particle_mode[0]] != 'off':
        ps.step(dt=0.033)
    _update_particles()

    # ── Person C: Wave simulation ─────────────────────────────
    ws.step(n_steps=2)
    if t_wall - _last_wave_wind[0] > 0.8:   # add ripple every 0.8s
        ws.add_wind_ripple(strength=0.004)
        _last_wave_wind[0] = t_wall

    disp = ws.get_displacement() * 0.06
    wpts = water_grid.points.copy()
    wpts[:, 1] = (SEA_LEVEL + disp).flatten(order='C')
    water_grid.points = wpts
    water_grid['wave_h'] = wpts[:, 1].astype(np.float32)

    # ── GLSL time uniform ─────────────────────────────────────
    if glsl_ok and water_uniforms is not None:
        water_uniforms.SetUniformf("anim_time", float(t_wall))

    # ── Person D: Deformer propagation ────────────────────────
    deformer.step(dt=0.033)

    pl.render()

# Register timer (~30fps)
try:
    pl.add_timer_event(max_steps=10_000_000, duration=33,
                       callback=animation_step)
    print("  Timer registered (30fps animation loop)")
except Exception:
    print("  Note: add_timer_event not available — morph is key-driven only.")


# =============================================================
# SECTION 12 — HUD
# =============================================================

hud = (
    "BIT 2325 — Procedural Terrain  (Milestones 5 + 6)\n"
    "Alexander Somba | SCT221-C004-0680/2023\n"
    "\n"
    "MOUSE:  Left=rotate   Right=zoom   Middle=pan\n"
    "\n"
    "M2 KEYS:\n"
    "  1-6  Camera presets    L  Lighting mode\n"
    "  N  Normals   W  Wireframe   T  Vegetation\n"
    "  S  Screenshot   R  Reset camera\n"
    "\n"
    "M5 KEYS:\n"
    "  M    Start/stop terrain morph (C² spline)\n"
    "  +/-  Scrub morph time forward/back\n"
    "  P    Cycle particles (off/leaves/dust/embers/all)\n"
    "  C    Drop wave disturbance\n"
    "  D    Toggle deform-raise mode\n"
    "\n"
    "M6 KEYS:\n"
    "  A    Adaptive Fractal Noise overlay\n"
    "  E    Ecosystem step (+5 steps)\n"
    "\n"
    "  Q / ESC  Quit"
)
pl.add_text(hud, position='lower_left', font_size=8.5,
            color='white', font='courier', shadow=True)

pipeline = TransformPipeline()
cam_ref  = Camera(Vector3(5,4.5,7), Vector3(0,0.3,0), Vector3(0,1,0), fov_deg=58)
pipeline.set_view(cam_ref.view_matrix)
pipeline.set_projection(cam_ref.projection_matrix)
analysis = NumericalAnalysis.analyse_pipeline(pipeline)

stats_text = (
    f"SYSTEM  (M1 → M6)\n"
    f"  Terrain : {W}×{D} = {stats['vertex_count']:,} verts\n"
    f"  Keyframes: {len(KEYFRAME_SEEDS)} (t=0→{animator.duration:.0f}s loop)\n"
    f"  Particles: 800 max  (3 types)\n"
    f"  Wave grid: {N_wave}×{N_wave}  CFL r={ws.r:.3f}\n"
    f"  GLSL water: {'YES' if glsl_ok else 'NO'}\n"
    f"\n"
    f"PIPELINE (M2)\n"
    f"  MVP κ : {analysis['mvp']['condition']:.4f}\n"
    f"  FOV   : 58°  f={1/math.tan(math.radians(29)):.3f}\n"
    f"  Octaves: 8   Seed: {SEED}\n"
    f"\n"
    f"M6 NOVEL\n"
    f"  AFN oct range: [{afn.oct_min}, {afn.oct_max}]\n"
    f"  Ecosystem cells: {W*D:,}\n"
    f"\n"
    f"Person A — Terrain + Morph\n"
    f"Person B — Vegetation + Particles\n"
    f"Person C — Lighting + Water + GLSL\n"
    f"Person D — Rocks + Colour + Deform"
)
pl.add_text(stats_text, position='upper_right', font_size=8.5,
            color='#aaddff', font='courier', shadow=True)


# =============================================================
# LAUNCH
# =============================================================

print()
print("CONTROLS:")
print("  M          → start/stop terrain morph")
print("  +/-        → scrub morph time")
print("  P          → cycle particles")
print("  C          → drop wave disturbance")
print("  D          → deform-raise  (left-click terrain)")
print("  A          → AFN octave overlay (M6)")
print("  E          → ecosystem step (M6)")
print("  1-6 L N W T S R  → all M2 controls")
print("─" * 58)
print("Window launching — press M to start the terrain morph!")
print()

pl.show(auto_close=False)
print("\nScene closed.")
