# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 2 — Enhanced Interactive 3D Scene
# Topic: Procedural Content Generation Systems
#
# Group Members:
#           Esther Achieng Otieno (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu (SCT221-C004-0217/2024)
#           Wendy Wachira (SCT221-C004-0194/2024)
#           Alexander Somba (SCT221-C004-0680/2023)ing
#
# Date: May 2026 | JKUAT
#
# HOW TO RUN:
#   python3 milestone2_scene.py
#
# CONTROLS:
#   Mouse left drag    — rotate
#   Mouse right drag   — zoom
#   Middle drag        — pan
#   Scroll             — zoom
#   1-6                — camera presets
#   L                  — cycle lighting (Day/Sunset/Dusk/Night)
#   N                  — toggle surface normals
#   W                  — toggle wireframe
#   T                  — toggle vegetation
#   R                  — reset camera
#   S                  — screenshot
#   Q / ESC            — quit
# =============================================================

import sys, os, math
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core    import Vector3, Matrix4, Terrain
from milestone2_core import Camera, TransformPipeline, NumericalAnalysis

# ── suppress VTK warnings ────────────────────────────────────
pv.global_theme.allow_empty_mesh = True


# =============================================================
# SECTION 1 — TERRAIN GENERATION (Person A)
# Uses Milestone 1 system unchanged
# =============================================================

print("=" * 58)
print("  BIT 2325 — Milestone 2: Enhanced 3D Scene")
print("  Procedural Content Generation System")
print("=" * 58)
print()
print("[Person A] Generating terrain...")

terrain = Terrain(
    width        = 200,
    depth        = 200,
    scale        = 6.0,
    height_scale = 2.8,
    octaves      = 8,
    persistence  = 0.52,
    lacunarity   = 2.1,
    seed         = 77,
)
stats = terrain.stats()
W, D  = terrain.width, terrain.depth
print(f"  Resolution : {W}×{D} = {stats['vertex_count']:,} vertices")
print(f"  Height     : [{stats['min_height']:.3f}, {stats['max_height']:.3f}]")


# ── Convert terrain to numpy arrays ──────────────────────────
hgrid = np.array(terrain.height_grid())         # (D, W) heights
X_lin = np.linspace(-2, 2, W)
Z_lin = np.linspace(-2, 2, D)
XX, ZZ = np.meshgrid(X_lin, Z_lin)

# Collect all vertex positions and normals
vert_pos = np.zeros((D * W, 3))
vert_nor = np.zeros((D * W, 3))
for j in range(D):
    for i in range(W):
        v = terrain.get_vertex(i, j)
        n = terrain.get_normal(i, j)
        # scale X,Z to [-2,2] world units
        vert_pos[j*W+i] = [v.x*2, v.y, v.z*2]
        vert_nor[j*W+i] = [n.x, n.y, n.z]


# =============================================================
# SECTION 2 — TERRAIN COLOURING (Person A + Person D)
# Rich blended colours matching reference image
# =============================================================

def blend(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return np.array(c1)*(1-t) + np.array(c2)*t

def terrain_colour(height, normal, world_x, world_z):
    """
    Multi-factor terrain colouring:
      - Height determines base biome
      - Slope (normal.y) darkens steep faces
      - Small noise adds micro-variation
    """
    hs = terrain.height_scale
    h  = height / hs           # normalise to [-1, 1]
    ny = float(normal[1])      # y-component of normal = flatness

    # Small positional noise for colour variation
    noise_val = (math.sin(world_x * 8.3 + world_z * 5.7) * 0.5 +
                 math.cos(world_x * 3.1 - world_z * 9.2) * 0.5) * 0.04

    # Base colours
    DEEP_WATER    = [0.08, 0.18, 0.38]
    SHALLOW_WATER = [0.14, 0.38, 0.60]
    WET_SAND      = [0.72, 0.60, 0.40]
    DRY_SAND      = [0.85, 0.72, 0.50]
    WARM_SAND     = [0.90, 0.68, 0.42]   # reference image warmth
    SCRUB_GRASS   = [0.45, 0.42, 0.22]
    GRASS         = [0.32, 0.50, 0.18]
    DARK_GRASS    = [0.22, 0.38, 0.12]
    FOREST        = [0.12, 0.28, 0.08]
    ROCK          = [0.52, 0.47, 0.40]
    DARK_ROCK     = [0.35, 0.30, 0.25]
    SNOW          = [0.92, 0.93, 0.95]

    # Height-based base colour
    if h < -0.35:
        c = DEEP_WATER
    elif h < -0.10:
        t = (h + 0.35) / 0.25
        c = blend(DEEP_WATER, SHALLOW_WATER, t)
    elif h < 0.00:
        t = (h + 0.10) / 0.10
        c = blend(WET_SAND, DRY_SAND, t)
    elif h < 0.08:
        t = h / 0.08
        c = blend(DRY_SAND, WARM_SAND, t + noise_val)
    elif h < 0.18:
        t = (h - 0.08) / 0.10
        c = blend(WARM_SAND, SCRUB_GRASS, t + noise_val)
    elif h < 0.32:
        t = (h - 0.18) / 0.14
        c = blend(SCRUB_GRASS, GRASS, t)
    elif h < 0.48:
        t = (h - 0.32) / 0.16
        c = blend(GRASS, FOREST, t)
    elif h < 0.62:
        t = (h - 0.48) / 0.14
        c = blend(FOREST, ROCK, t)
    elif h < 0.78:
        t = (h - 0.62) / 0.16
        c = blend(ROCK, DARK_ROCK, t)
    else:
        t = (h - 0.78) / 0.22
        c = blend(DARK_ROCK, SNOW, t)

    # Slope darkening — steep faces are darker (rock-like)
    slope_factor = max(0.0, ny)
    if slope_factor < 0.6 and h > 0.15:
        c = blend(DARK_ROCK, c, slope_factor / 0.6)

    # Ambient occlusion approximation — valleys are darker
    ao = max(0.75, min(1.0, 0.75 + h * 0.5))
    c = [v * ao for v in c]

    return np.clip(c, 0, 1)

print("[Person A+D] Computing terrain colours...")
colours = np.zeros((D * W, 3))
for j in range(D):
    for i in range(W):
        h   = hgrid[j, i]
        nor = vert_nor[j*W+i]
        wx  = XX[j, i]
        wz  = ZZ[j, i]
        colours[j*W+i] = terrain_colour(h, nor, wx, wz)

# Build PyVista mesh
terrain_grid = pv.StructuredGrid(XX*2, hgrid, ZZ*2)
terrain_grid['colours'] = colours
terrain_grid['height']  = hgrid.flatten(order='C')
terrain_grid.point_data['normals'] = vert_nor
terrain_grid.point_data.active_normals_name = 'normals'


# =============================================================
# SECTION 3 — WATER (Person C)
# Layered water plane with transparency and specular
# =============================================================

print("[Person C] Building water plane...")

SEA_LEVEL = -0.12

# Main water surface
water_plane = pv.Plane(
    center     = (0, SEA_LEVEL, 0),
    direction  = (0, 1, 0),
    i_size     = 4, j_size = 4,
    i_resolution = 80, j_resolution = 80
)
# Slight wave displacement using noise
wpts = water_plane.points.copy()
for k in range(len(wpts)):
    x, z = wpts[k, 0], wpts[k, 2]
    wave = (math.sin(x * 4.1 + 0.3) * math.cos(z * 3.7 + 0.8)) * 0.008
    wpts[k, 1] += wave
water_plane.points = wpts


# =============================================================
# SECTION 4 — VEGETATION (Person B)
# Procedural tree + cactus placement driven by terrain noise
# =============================================================

print("[Person B] Placing vegetation...")

def place_trees(terrain, hgrid, XX, ZZ, W, D, count=300, seed=42):
    """
    Place trees on forest/grass biome areas.
    Position driven by Perlin noise — trees cluster naturally.
    Each tree: trunk (cylinder) + canopy (sphere).
    Returns a list of PyVista meshes.
    """
    rng    = np.random.default_rng(seed)
    meshes = []
    placed = 0
    attempts = 0

    while placed < count and attempts < count * 12:
        attempts += 1
        # Random grid position
        gi = rng.integers(2, W-2)
        gj = rng.integers(2, D-2)
        h  = hgrid[gj, gi]
        hn = h / terrain.height_scale

        # Only place in forest/grass zone, on flat ground
        ny = float(vert_nor[gj*W+gi][1])
        if not (0.18 < hn < 0.52) or ny < 0.75:
            continue

        wx = XX[gj, gi]
        wz = ZZ[gj, gi]

        # Tree parameters with variation
        trunk_h  = rng.uniform(0.10, 0.22)
        trunk_r  = rng.uniform(0.008, 0.018)
        canopy_r = rng.uniform(0.06, 0.14)
        lean_x   = rng.uniform(-0.02, 0.02)
        lean_z   = rng.uniform(-0.02, 0.02)

        # Colour variation — dark green forest like reference
        g_val = rng.uniform(0.28, 0.42)
        b_val = rng.uniform(0.05, 0.14)
        canopy_col = (rng.uniform(0.05, 0.16), g_val, b_val)
        trunk_col  = (rng.uniform(0.28, 0.38), rng.uniform(0.20, 0.28), 0.10)

        base_y = h

        # Trunk
        trunk = pv.Cylinder(
            center    = (wx+lean_x*trunk_h/2, base_y + trunk_h/2, wz+lean_z*trunk_h/2),
            direction = (lean_x, 1.0, lean_z),
            radius    = trunk_r,
            height    = trunk_h,
            resolution= 6,
        )

        # Canopy — two overlapping spheres for density
        cy1 = base_y + trunk_h + canopy_r * 0.6
        cy2 = base_y + trunk_h + canopy_r * 1.1
        canopy1 = pv.Sphere(radius=canopy_r,     center=(wx+lean_x, cy1, wz+lean_z), theta_resolution=8, phi_resolution=8)
        canopy2 = pv.Sphere(radius=canopy_r*0.7, center=(wx+lean_x*0.5, cy2, wz+lean_z*0.5), theta_resolution=8, phi_resolution=8)

        meshes.append(('trunk',   trunk,   trunk_col))
        meshes.append(('canopy',  canopy1, canopy_col))
        meshes.append(('canopy2', canopy2, canopy_col))
        placed += 1

    print(f"    Trees placed: {placed}")
    return meshes


def place_cacti(terrain, hgrid, XX, ZZ, W, D, count=180, seed=99):
    """
    Place cacti on sand/scrub biome areas.
    Saguaro-style: main trunk + two arms.
    """
    rng    = np.random.default_rng(seed)
    meshes = []
    placed = 0
    attempts = 0

    while placed < count and attempts < count * 10:
        attempts += 1
        gi = rng.integers(2, W-2)
        gj = rng.integers(2, D-2)
        h  = hgrid[gj, gi]
        hn = h / terrain.height_scale
        ny = float(vert_nor[gj*W+gi][1])

        # Only on sand zone, flat ground
        if not (-0.05 < hn < 0.17) or ny < 0.80:
            continue

        wx  = XX[gj, gi]
        wz  = ZZ[gj, gi]
        col = (rng.uniform(0.55, 0.72),   # warm yellow-green
               rng.uniform(0.65, 0.80),
               rng.uniform(0.08, 0.18))

        trunk_h = rng.uniform(0.06, 0.14)
        trunk_r = rng.uniform(0.006, 0.012)
        base_y  = h

        # Main trunk
        trunk = pv.Cylinder(
            center    = (wx, base_y + trunk_h/2, wz),
            direction = (0, 1, 0),
            radius    = trunk_r,
            height    = trunk_h,
            resolution= 5,
        )
        meshes.append(('cactus', trunk, col))

        # Arms (only on larger cacti)
        if trunk_h > 0.09:
            for side in [-1, 1]:
                arm_len   = rng.uniform(0.025, 0.045)
                arm_start = base_y + trunk_h * rng.uniform(0.5, 0.75)
                # Horizontal arm
                h_arm = pv.Cylinder(
                    center    = (wx + side * arm_len/2, arm_start, wz),
                    direction = (1, 0, 0),
                    radius    = trunk_r * 0.75,
                    height    = arm_len,
                    resolution= 5,
                )
                # Vertical tip
                v_tip = pv.Cylinder(
                    center    = (wx + side * arm_len, arm_start + arm_len/2, wz),
                    direction = (0, 1, 0),
                    radius    = trunk_r * 0.65,
                    height    = arm_len,
                    resolution= 5,
                )
                meshes.append(('cactus_arm', h_arm, col))
                meshes.append(('cactus_tip', v_tip, col))
        placed += 1

    print(f"    Cacti placed: {placed}")
    return meshes


def place_rocks(terrain, hgrid, XX, ZZ, W, D, count=120, seed=55):
    """
    Place rocks on rocky/transition zones.
    Randomly scaled and rotated spheres flattened on Y-axis.
    """
    rng    = np.random.default_rng(seed)
    meshes = []
    placed = 0
    attempts = 0

    while placed < count and attempts < count * 10:
        attempts += 1
        gi = rng.integers(2, W-2)
        gj = rng.integers(2, D-2)
        h  = hgrid[gj, gi]
        hn = h / terrain.height_scale
        ny = float(vert_nor[gj*W+gi][1])

        if not (0.05 < hn < 0.70):
            continue

        wx  = XX[gj, gi]
        wz  = ZZ[gj, gi]
        r   = rng.uniform(0.012, 0.055)
        # Flatten Y slightly — rocks sit on ground
        rock = pv.Sphere(radius=r,
                          center=(wx, h + r*0.45, wz),
                          theta_resolution=8, phi_resolution=6)
        # Squash Y
        pts = rock.points.copy()
        pts[:, 1] = (pts[:, 1] - (h + r*0.45)) * 0.55 + (h + r*0.45)
        # Random rotation
        angle = rng.uniform(0, 360)
        cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        px_r = pts[:,0]*cos_a - pts[:,2]*sin_a
        pz_r = pts[:,0]*sin_a + pts[:,2]*cos_a
        pts[:,0], pts[:,2] = px_r, pz_r
        rock.points = pts

        grey  = rng.uniform(0.38, 0.56)
        warm  = rng.uniform(0.00, 0.06)
        col   = (grey + warm, grey, grey - warm*0.5)
        meshes.append(('rock', rock, tuple(np.clip(col, 0, 1))))
        placed += 1

    print(f"    Rocks placed: {placed}")
    return meshes


tree_meshes  = place_trees(terrain, hgrid, XX, ZZ, W, D, count=320, seed=42)
cacti_meshes = place_cacti(terrain, hgrid, XX, ZZ, W, D, count=200, seed=99)
rock_meshes  = place_rocks(terrain, hgrid, XX, ZZ, W, D, count=150, seed=55)


# =============================================================
# SECTION 5 — ATMOSPHERE (Person C)
# Sky dome and atmospheric haze
# =============================================================

print("[Person C] Building atmosphere...")

# Sky dome — large sphere around scene
sky = pv.Sphere(radius=18, theta_resolution=24, phi_resolution=24)
# Gradient: blue sky at top, warm horizon at equator
sky_colours = np.zeros((len(sky.points), 3))
for k, pt in enumerate(sky.points):
    y_norm = (pt[1] + 18) / 36   # 0 at bottom, 1 at top
    if y_norm > 0.5:
        # Upper sky — deep blue
        t = (y_norm - 0.5) / 0.5
        sky_colours[k] = blend([0.65, 0.78, 0.92], [0.28, 0.48, 0.80], t)
    else:
        # Horizon — warm haze
        t = y_norm / 0.5
        sky_colours[k] = blend([0.88, 0.76, 0.62], [0.65, 0.78, 0.92], t)
sky['sky_rgb'] = sky_colours

# Distant mountains haze plane (slight blue fog at horizon)
haze = pv.Plane(center=(0, 0.8, 0), direction=(0, 1, 0),
                i_size=36, j_size=36, i_resolution=2, j_resolution=2)


# =============================================================
# SECTION 6 — LIGHTING MODES (Person C)
# =============================================================

LIGHTING_MODES = [
    {
        'name'      : 'Golden Hour (warm afternoon)',
        'bg_top'    : '#4a6fa5',
        'bg_bottom' : '#e8c090',
        'lights'    : [
            # Main sun — warm, low angle
            {'pos':(12, 4, -8),  'color':'#FFD580', 'intensity':1.2, 'focal':(0,0,0)},
            # Sky fill — cool blue
            {'pos':(-5, 10, 5),  'color':'#AAC8FF', 'intensity':0.45,'focal':(0,0,0)},
            # Ground bounce — warm
            {'pos':(0, -3, 0),   'color':'#FFB870', 'intensity':0.20,'focal':(0,0,0)},
        ],
        'ambient'   : 0.35,
    },
    {
        'name'      : 'Midday Sun (bright)',
        'bg_top'    : '#1a3a6a',
        'bg_bottom' : '#a8c8f0',
        'lights'    : [
            {'pos':(2,  15, 3),  'color':'#FFFAF0', 'intensity':1.4, 'focal':(0,0,0)},
            {'pos':(-8,  6, -4), 'color':'#C0D8FF', 'intensity':0.35,'focal':(0,0,0)},
            {'pos':(0,  -2, 0),  'color':'#E8E4D0', 'intensity':0.15,'focal':(0,0,0)},
        ],
        'ambient'   : 0.45,
    },
    {
        'name'      : 'Sunset (dramatic)',
        'bg_top'    : '#1a1a3a',
        'bg_bottom' : '#FF6030',
        'lights'    : [
            {'pos':(14,  1, 0),  'color':'#FF7030', 'intensity':1.3, 'focal':(0,0,0)},
            {'pos':(-6,  8, 4),  'color':'#6040AA', 'intensity':0.30,'focal':(0,0,0)},
            {'pos':(0,  -2, 0),  'color':'#FF5520', 'intensity':0.25,'focal':(0,0,0)},
        ],
        'ambient'   : 0.18,
    },
    {
        'name'      : 'Overcast (soft diffuse)',
        'bg_top'    : '#606875',
        'bg_bottom' : '#A8B0B8',
        'lights'    : [
            {'pos':(0,  14, 0),  'color':'#D0D8E8', 'intensity':0.85,'focal':(0,0,0)},
            {'pos':(8,   6, 8),  'color':'#C8D0D8', 'intensity':0.40,'focal':(0,0,0)},
            {'pos':(-8,  6,-8),  'color':'#C8D0D8', 'intensity':0.40,'focal':(0,0,0)},
        ],
        'ambient'   : 0.60,
    },
]

lighting_idx = [0]


# =============================================================
# SECTION 7 — CAMERA PRESETS (Person A)
# All using our derived lookAt + perspective matrices
# =============================================================

PRESETS = {
    '1': {'name':'Overview',        'pos':(5.0, 4.5, 7.0), 'focal':(0.0, 0.3, 0.0), 'fov':58},
    '2': {'name':'Ground level',    'pos':(2.5, 0.2, 4.0), 'focal':(0.0, 0.5, 0.0), 'fov':72},
    '3': {'name':'Aerial',          'pos':(0.0,10.0, 0.2), 'focal':(0.0, 0.0, 0.0), 'fov':50},
    '4': {'name':'Telephoto side',  'pos':(9.0, 3.0, 1.0), 'focal':(0.0, 0.5, 0.0), 'fov':32},
    '5': {'name':'Mountain face',   'pos':(1.5, 2.5,-4.0), 'focal':(0.0, 1.0, 0.0), 'fov':65},
    '6': {'name':'Water edge',      'pos':(1.8, 0.1, 1.5), 'focal':(-0.5,-0.05,-0.5),'fov':78},
}

def apply_preset(pl, key):
    cfg = PRESETS[key]
    cam = Camera(Vector3(*cfg['pos']), Vector3(*cfg['focal']),
                 Vector3(0,1,0), fov_deg=cfg['fov'])
    _ = cam.view_matrix        # trigger our derivation
    _ = cam.projection_matrix
    pl.camera.position    = cfg['pos']
    pl.camera.focal_point = cfg['focal']
    pl.camera.up          = (0.0, 1.0, 0.0)
    pl.camera.view_angle  = float(cfg['fov'])
    f = 1/math.tan(math.radians(cfg['fov']/2))
    print(f"\n  [{cfg['name']}]  pos={cfg['pos']}  FOV={cfg['fov']}°  f={f:.3f}")


# =============================================================
# SECTION 8 — BUILD PLOTTER
# =============================================================

print("\nLaunching interactive scene...")
print("─" * 58)

pl = pv.Plotter(
    title       = "BIT 2325 — Milestone 2 | Procedural Terrain | Group",
    window_size = [1400, 900],
)

mode = LIGHTING_MODES[0]
pl.set_background(mode['bg_bottom'], top=mode['bg_top'])

# Sky dome
pl.add_mesh(sky, scalars='sky_rgb', rgb=True,
            smooth_shading=True, show_edges=False,
            lighting=False, opacity=1.0)

# Terrain
terrain_actor = pl.add_mesh(
    terrain_grid,
    scalars        = 'colours',
    rgb            = True,
    smooth_shading = True,
    show_edges     = False,
    lighting       = True,
    specular       = 0.18,
    specular_power = 12,
    name           = 'terrain',
)

# Water
water_actor = pl.add_mesh(
    water_plane,
    color          = '#2266AA',
    opacity        = 0.62,
    smooth_shading = True,
    specular       = 1.0,
    specular_power = 60,
    lighting       = True,
    name           = 'water',
)

# Vegetation + rocks — add all as single batches per type
print("[Person B+D] Adding scene objects to renderer...")
veg_actors = []

def add_object_batch(pl, meshes, name_prefix):
    actors = []
    for i, (kind, mesh, col) in enumerate(meshes):
        spec  = 0.05 if 'canopy' in kind else 0.15
        actor = pl.add_mesh(
            mesh,
            color          = col,
            smooth_shading = True,
            lighting       = True,
            specular       = spec,
            specular_power = 8,
            name           = f"{name_prefix}_{i}",
        )
        actors.append((f"{name_prefix}_{i}", actor))
    return actors

veg_actors += add_object_batch(pl, tree_meshes,  'tree')
veg_actors += add_object_batch(pl, cacti_meshes, 'cactus')
veg_actors += add_object_batch(pl, rock_meshes,  'rock')

print(f"  Scene objects: {len(veg_actors)} meshes added")

# Initial lighting
def apply_lighting(pl, idx):
    mode = LIGHTING_MODES[idx % len(LIGHTING_MODES)]
    pl.remove_all_lights()
    for lc in mode['lights']:
        light = pv.Light(
            position    = lc['pos'],
            focal_point = lc['focal'],
            color       = lc['color'],
            intensity   = lc['intensity'],
            positional  = False,
        )
        pl.add_light(light)
    pl.set_background(mode['bg_bottom'], top=mode['bg_top'])
    print(f"\n  Lighting: {mode['name']}")

apply_lighting(pl, 0)
apply_preset(pl, '1')


# =============================================================
# SECTION 9 — INTERACTIVE CALLBACKS
# =============================================================

show_normals   = [False]
show_wireframe = [False]
show_veg       = [True]
normals_actor  = [None]

def toggle_normals():
    show_normals[0] = not show_normals[0]
    if show_normals[0]:
        step = 10
        starts, ends = [], []
        for j in range(0, D, step):
            for i in range(0, W, step):
                v  = terrain.get_vertex(i, j)
                n  = terrain.get_normal(i, j)
                sc = 0.10
                starts.append([v.x*2, v.y, v.z*2])
                ends.append([v.x*2 + n.x*sc,
                             v.y   + n.y*sc,
                             v.z*2 + n.z*sc])
        pts_flat = []
        for s, e in zip(starts, ends):
            pts_flat.extend([s, e])
        lines = pv.MultipleLines(points=np.array(pts_flat))
        normals_actor[0] = pl.add_mesh(
            lines, color='red', line_width=1.5, name='normals')
        print("\n  Normals: ON")
    else:
        if normals_actor[0]:
            pl.remove_actor('normals')
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
    for name, actor in veg_actors:
        actor.SetVisibility(show_veg[0])
    print(f"\n  Vegetation: {'ON' if show_veg[0] else 'OFF'}")
    pl.render()

def cycle_lighting():
    lighting_idx[0] = (lighting_idx[0] + 1) % len(LIGHTING_MODES)
    apply_lighting(pl, lighting_idx[0])
    pl.render()

def screenshot():
    fname = "scene_screenshot.png"
    pl.screenshot(fname)
    print(f"\n  Screenshot: {fname}")

def reset_cam():
    apply_preset(pl, '1')
    pl.render()

# key bindings
for key, fn in [('n','toggle_normals'),('N','toggle_normals'),
                ('w','toggle_wireframe'),('W','toggle_wireframe'),
                ('t','toggle_vegetation'),('T','toggle_vegetation'),
                ('l','cycle_lighting'),('L','cycle_lighting'),
                ('s','screenshot'),('S','screenshot'),
                ('r','reset_cam'),('R','reset_cam')]:
    pl.add_key_event(key, locals()[fn])

for k in PRESETS:
    def _make_fn(pk):
        def _fn(): apply_preset(pl, pk); pl.render()
        return _fn
    pl.add_key_event(k, _make_fn(k))


# =============================================================
# SECTION 10 — HUD
# =============================================================

hud = (
    "BIT 2325 — Procedural Terrain  (Milestone 2)\n"
    "Alexander Somba | SCT221-C004-0680/2023\n"
    "\n"
    "MOUSE:  Left=rotate  Right=zoom  Middle=pan\n"
    "\n"
    "KEYS:\n"
    "  1-6  Camera presets\n"
    "  L    Lighting (Golden Hour/Midday/Sunset/Overcast)\n"
    "  N    Surface normals\n"
    "  W    Wireframe\n"
    "  T    Toggle vegetation\n"
    "  S    Screenshot\n"
    "  R    Reset camera\n"
    "  Q    Quit"
)
pl.add_text(hud, position='lower_left', font_size=9,
            color='white', font='courier', shadow=True)

# Stats overlay
pipeline = TransformPipeline()
cam_ref  = Camera(Vector3(5,4.5,7), Vector3(0,0.3,0),
                  Vector3(0,1,0), fov_deg=58)
pipeline.set_view(cam_ref.view_matrix)
pipeline.set_projection(cam_ref.projection_matrix)
analysis = NumericalAnalysis.analyse_pipeline(pipeline)

stats_text = (
    f"SYSTEM  (Milestones 1 + 2)\n"
    f"  Vertices : {stats['vertex_count']:,}\n"
    f"  Trees    : {sum(1 for m in tree_meshes if m[0]=='trunk')}\n"
    f"  Cacti    : {sum(1 for m in cacti_meshes if m[0]=='cactus')}\n"
    f"  Rocks    : {len(rock_meshes)}\n"
    f"\n"
    f"PIPELINE\n"
    f"  MVP κ  : {analysis['mvp']['condition']:.4f}\n"
    f"  FOV    : 58°  f={1/math.tan(math.radians(29)):.3f}\n"
    f"  Octaves: 8   Seed: 77\n"
    f"\n"
    f"Person A — Terrain + Camera\n"
    f"Person B — Vegetation\n"
    f"Person C — Lighting + Water\n"
    f"Person D — Rocks + Colour"
)
pl.add_text(stats_text, position='upper_right', font_size=8.5,
            color='#aaddff', font='courier', shadow=True)


# =============================================================
# LAUNCH
# =============================================================

print()
print("CONTROLS:")
print("  Mouse left drag  → rotate")
print("  Mouse right drag → zoom")
print("  Middle drag      → pan")
print("  1-6              → camera presets")
print("  L                → cycle lighting")
print("  N                → surface normals")
print("  W                → wireframe")
print("  T                → toggle vegetation")
print("  S                → screenshot")
print("  Q / ESC          → quit")
print("─" * 58)
print("Window launching...")

pl.show(auto_close=False)
print("\nScene closed.")
