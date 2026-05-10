# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 1 — Visualisation Script
# Topic: Procedural Content Generation Systems
#
# Authors:   Esther Achieng Otieno (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu (SCT221-C004-0217/2024)
#           Wendy Wachira (SCT221-C004-0194/2024)
#           Alexander Somba (SCT221-C004-0680/2023)
# Date  : May 2026 | JKUAT
#
# Generates all figures for the Milestone 1 report.
# Run: python3 milestone1_visualise.py
# =============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core import Vector3, Matrix4, PerlinNoise, Terrain

OUT = "./figures"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# ── Shared colour map — terrain biomes ───────────────────────
BIOME_COLOURS = {
    'deep_water':    '#1a3f6f',
    'shallow_water': '#2e6da4',
    'sand':          '#c2b280',
    'grass':         '#5a8a3c',
    'forest':        '#2d5a1b',
    'rock':          '#7a7060',
    'snow':          '#f0f0f0',
}

def height_to_colour(h, h_min, h_max, height_scale):
    """Map a height value to a biome RGB colour."""
    norm = h / height_scale
    if norm < -0.3:  c = BIOME_COLOURS['deep_water']
    elif norm < -0.05: c = BIOME_COLOURS['shallow_water']
    elif norm < 0.02:  c = BIOME_COLOURS['sand']
    elif norm < 0.25:  c = BIOME_COLOURS['grass']
    elif norm < 0.45:  c = BIOME_COLOURS['forest']
    elif norm < 0.65:  c = BIOME_COLOURS['rock']
    else:              c = BIOME_COLOURS['snow']
    return mcolors.to_rgb(c)


# =============================================================
# FIGURE 1 — 3D Terrain Surface (main showcase)
# =============================================================
print("Generating Figure 1 — 3D Terrain Surface...")

terrain = Terrain(width=128, depth=128, scale=5.5,
                  height_scale=3.0, octaves=7, seed=42)

hgrid = np.array(terrain.height_grid())
W, D  = terrain.width, terrain.depth

X = np.linspace(-1, 1, W)
Z = np.linspace(-1, 1, D)
XX, ZZ = np.meshgrid(X, Z)
YY = hgrid

# Build biome colour array
colours = np.zeros((*hgrid.shape, 3))
h_min, h_max = hgrid.min(), hgrid.max()
for j in range(D):
    for i in range(W):
        colours[j, i] = height_to_colour(
            hgrid[j,i], h_min, h_max, terrain.height_scale)

fig = plt.figure(figsize=(14, 9))
ax  = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(XX, YY, ZZ,
                        facecolors=colours,
                        linewidth=0, antialiased=True,
                        shade=True, alpha=0.97)

# Atmospheric lighting effect
ax.view_init(elev=35, azim=-55)
ax.set_xlabel("X", labelpad=10, fontsize=11)
ax.set_ylabel("Height", labelpad=10, fontsize=11)
ax.set_zlabel("Z", labelpad=10, fontsize=11)
ax.set_title("Procedural Terrain — Milestone 1\nfBm Perlin Noise  |  128×128 vertices  |  7 octaves",
             fontsize=13, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(fc=c, label=b.replace('_',' ').title())
                   for b,c in BIOME_COLOURS.items()]
ax.legend(handles=legend_elements, loc='upper left',
          fontsize=9, framealpha=0.9, ncol=1)

ax.set_facecolor('#0a0a1a')
fig.patch.set_facecolor('#0a0a1a')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.title.set_color('white')
save(fig, "fig1_terrain_3d")


# =============================================================
# FIGURE 2 — Noise Analysis: single vs octave layers
# =============================================================
print("Generating Figure 2 — Noise Layer Analysis...")

noise_gen = PerlinNoise(seed=42)
N = 256
x_vals = np.linspace(0, 5, N)
y_vals = np.linspace(0, 5, N)
XX2, YY2 = np.meshgrid(x_vals, y_vals)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Figure 2 — Fractal Brownian Motion: Building Terrain Layer by Layer",
             fontsize=13, fontweight='bold')

octave_results = []
for oct_count in range(1, 8):
    result = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            result[j,i] = noise_gen.octave_noise(
                x_vals[i], y_vals[j],
                octaves=oct_count,
                persistence=0.5,
                lacunarity=2.0)
    octave_results.append(result)

for idx in range(7):
    row, col = idx // 4, idx % 4
    ax = axes[row][col]
    im = ax.imshow(octave_results[idx], cmap='terrain',
                   origin='lower', interpolation='bilinear')
    ax.set_title(f"Octave {idx+1}", fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Last panel — frequency spectrum annotation
ax = axes[1][3]
ax.set_facecolor('#f5f5f5')
ax.axis('off')
ax.text(0.5, 0.95, "fBm Formula", ha='center', va='top',
        fontsize=11, fontweight='bold', transform=ax.transAxes)
formula_lines = [
    "H(x,y) = Σᵢ aᵢ · noise(x·fᵢ, y·fᵢ)",
    "",
    "aᵢ = persistenceⁱ",
    "fᵢ = lacunarityⁱ",
    "",
    "persistence = 0.5",
    "lacunarity  = 2.0",
    "octaves     = 7",
    "",
    "Each octave adds detail",
    "at half the amplitude",
    "and double the frequency.",
]
for k, line in enumerate(formula_lines):
    ax.text(0.1, 0.82 - k*0.07, line, ha='left', va='top',
            fontsize=9, transform=ax.transAxes,
            fontfamily='monospace' if '=' in line else 'sans-serif')

plt.tight_layout()
save(fig, "fig2_noise_octaves")


# =============================================================
# FIGURE 3 — Height Map + Biome Map (2D top-down views)
# =============================================================
print("Generating Figure 3 — Height and Biome Maps...")

terrain2 = Terrain(width=256, depth=256, scale=5.0,
                   height_scale=3.0, octaves=6, seed=42)
hgrid2   = np.array(terrain2.height_grid())

# Biome colour image
biome_img = np.zeros((256, 256, 3))
for j in range(256):
    for i in range(256):
        biome_img[j,i] = height_to_colour(
            hgrid2[j,i], hgrid2.min(), hgrid2.max(),
            terrain2.height_scale)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 3 — Top-Down Terrain Analysis  |  256×256 vertices",
             fontsize=12, fontweight='bold')

# Height map
im1 = axes[0].imshow(hgrid2, cmap='terrain',
                      origin='lower', interpolation='bilinear')
axes[0].set_title("Height Map\n(brighter = higher)", fontsize=10, fontweight='bold')
axes[0].set_xlabel("X grid"); axes[0].set_ylabel("Z grid")
plt.colorbar(im1, ax=axes[0], label="Height (units)")

# Biome map
axes[1].imshow(biome_img, origin='lower', interpolation='nearest')
axes[1].set_title("Biome Map\n(colour-coded by terrain type)", fontsize=10, fontweight='bold')
axes[1].set_xlabel("X grid")
legend_elements = [plt.Rectangle((0,0),1,1, color=c, label=b.replace('_',' ').title())
                   for b,c in BIOME_COLOURS.items()]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=7.5)

# Height distribution histogram
all_h = hgrid2.flatten()
axes[2].hist(all_h, bins=80, color='steelblue', alpha=0.8, edgecolor='white', lw=0.3)
axes[2].set_title("Height Distribution\n(histogram of all vertices)",
                  fontsize=10, fontweight='bold')
axes[2].set_xlabel("Height value"); axes[2].set_ylabel("Vertex count")
axes[2].axvline(all_h.mean(), color='red', lw=2, label=f"Mean = {all_h.mean():.3f}")
axes[2].axvline(0, color='orange', lw=1.5, ls='--', label="Sea level")
axes[2].legend(fontsize=9); axes[2].grid(axis='y', alpha=0.4)

plt.tight_layout()
save(fig, "fig3_maps")


# =============================================================
# FIGURE 4 — Surface Normals Visualisation
# =============================================================
print("Generating Figure 4 — Surface Normals...")

terrain3 = Terrain(width=32, depth=32, scale=4.0,
                   height_scale=2.0, octaves=5, seed=42)
hgrid3   = np.array(terrain3.height_grid())
W3, D3   = terrain3.width, terrain3.depth
X3 = np.linspace(-1,1,W3); Z3 = np.linspace(-1,1,D3)
XX3, ZZ3 = np.meshgrid(X3, Z3)

fig = plt.figure(figsize=(13, 6))
fig.suptitle("Figure 4 — Surface Normal Vectors\n"
             "Computed via cross product of tangent vectors: n = (∂H/∂x) × (∂H/∂z)",
             fontsize=11, fontweight='bold')

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XX3, hgrid3, ZZ3,
                alpha=0.55, cmap='terrain',
                linewidth=0, antialiased=True)

# Draw normals as quivers (every 4th vertex)
step = 4
for j in range(0, D3, step):
    for i in range(0, W3, step):
        v  = terrain3.get_vertex(i, j)
        n  = terrain3.get_normal(i, j)
        scale = 0.12
        ax.quiver(v.x, v.y, v.z,
                  n.x * scale, n.y * scale, n.z * scale,
                  color='red', arrow_length_ratio=0.4,
                  linewidth=0.8, alpha=0.8)

ax.set_title("Red arrows = surface normals (perpendicular to terrain)",
             fontsize=9, color='#333333')
ax.set_xlabel("X"); ax.set_ylabel("Height"); ax.set_zlabel("Z")
ax.view_init(elev=30, azim=-60)
plt.tight_layout()
save(fig, "fig4_normals")


# =============================================================
# FIGURE 5 — Mathematical Foundations Dashboard
# =============================================================
print("Generating Figure 5 — Mathematical Foundations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Figure 5 — Mathematical Foundations of the Terrain System",
             fontsize=13, fontweight='bold')

# Panel 1: Dot product visualisation
ax = axes[0][0]
a_v = np.array([2, 1])
b_v = np.array([1, 2])
ax.annotate('', xy=a_v, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5))
ax.annotate('', xy=b_v, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=2.5))
proj = np.dot(a_v, b_v/np.linalg.norm(b_v)) * (b_v/np.linalg.norm(b_v))
ax.annotate('', xy=proj, xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, linestyle='dashed'))
ax.text(a_v[0]+0.1, a_v[1], 'a = (2,1)', fontsize=9, color='steelblue')
ax.text(b_v[0]+0.1, b_v[1], 'b = (1,2)', fontsize=9, color='firebrick')
theta = np.degrees(np.arccos(np.dot(a_v,b_v)/(np.linalg.norm(a_v)*np.linalg.norm(b_v))))
ax.text(0.5, 0.5, f"a·b = {np.dot(a_v,b_v)}\n|a||b|cos θ\nθ = {theta:.1f}°",
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round', fc='lightyellow'))
ax.set_xlim(-0.5, 3); ax.set_ylim(-0.5, 3)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_title("Dot Product: a·b = |a||b|cosθ", fontsize=10, fontweight='bold')

# Panel 2: Cross product
ax = axes[0][1]
o = np.array([0,0,0])
av = np.array([1,0,0]); bv = np.array([0,1,0])
cv = np.cross(av, bv)
for vec, col, lbl in [(av,'steelblue','a=(1,0,0)'),(bv,'firebrick','b=(0,1,0)'),(cv,'darkgreen','a×b=(0,0,1)')]:
    ax.annotate('', xy=vec[:2]+np.array([0.05,0.05]),
                xytext=o[:2],
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.text(vec[0]+0.08, vec[1]+0.05, lbl, color=col, fontsize=9)
ax.text(0.5, -0.3, "a×b ⊥ both a and b\nUsed to compute terrain normals",
        ha='center', fontsize=8.5, style='italic')
ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_title("Cross Product: Surface Normals", fontsize=10, fontweight='bold')

# Panel 3: Smoothstep fade curve
ax = axes[0][2]
t = np.linspace(0, 1, 300)
linear = t
cubic  = 3*t**2 - 2*t**3           # original smoothstep
quintic= 6*t**5 - 15*t**4 + 10*t**3 # Perlin's improved
ax.plot(t, linear,  'b--', lw=1.5, label='Linear (artifacts)', alpha=0.7)
ax.plot(t, cubic,   'orange', lw=2, label='Cubic smoothstep')
ax.plot(t, quintic, 'red', lw=2.5, label="Perlin's quintic (C²)")
ax.set_title("Fade / Smoothstep Curves\nfor Noise Interpolation", fontsize=10, fontweight='bold')
ax.set_xlabel("t"); ax.set_ylabel("f(t)")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.text(0.5, 0.15, "f(t) = 6t⁵ − 15t⁴ + 10t³",
        ha='center', fontsize=9, color='red',
        bbox=dict(boxstyle='round', fc='#fff0f0'))

# Panel 4: 1D noise slice
ax = axes[1][0]
noise_gen2 = PerlinNoise(seed=42)
xs = np.linspace(0, 6, 500)
n1 = [noise_gen2.noise(x, 0.5) for x in xs]
n_oct = [noise_gen2.octave_noise(x, 0.5, octaves=k) for x in xs
         for k in [1]][0:0]
for oct_n, col, lbl in [(1,'lightblue','1 octave'),
                         (3,'steelblue','3 octaves'),
                         (6,'navy','6 octaves')]:
    vals = [noise_gen2.octave_noise(x, 0.5, octaves=oct_n) for x in xs]
    ax.plot(xs, vals, color=col, lw=1.5+oct_n*0.1, label=lbl, alpha=0.9)
ax.set_title("1D Noise Slice\n(varying octave count)", fontsize=10, fontweight='bold')
ax.set_xlabel("x"); ax.set_ylabel("noise value")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', lw=0.8, ls='--')

# Panel 5: Matrix transformation demo
ax = axes[1][1]
# Show a square before and after transformation
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float)
T_2d = np.array([[1.5, 0.3],[0.2, 1.2]])
transformed = (T_2d @ square.T).T
ax.fill(square[:,0], square[:,1], alpha=0.3, color='steelblue', label='Original')
ax.plot(square[:,0], square[:,1], 'b-o', ms=5, lw=2)
ax.fill(transformed[:,0], transformed[:,1], alpha=0.3, color='firebrick', label='Transformed')
ax.plot(transformed[:,0], transformed[:,1], 'r-o', ms=5, lw=2)
for i, (s, t) in enumerate(zip(square[:-1], transformed[:-1])):
    ax.annotate('', xy=t, xytext=s,
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.8, alpha=0.6))
ax.set_title("Matrix Transformation\n(scale + shear demo)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
ax.set_xlim(-0.2, 2); ax.set_ylim(-0.2, 1.6)

# Panel 6: Biome distribution pie
ax = axes[1][2]
terrain_pie = Terrain(width=128, depth=128, scale=5.0, height_scale=3.0, octaves=6, seed=42)
hgrid_pie   = np.array(terrain_pie.height_grid())
biome_counts = {b:0 for b in BIOME_COLOURS}
for j in range(128):
    for i in range(128):
        bm = terrain_pie.classify_biome(hgrid_pie[j,i])
        biome_counts[bm] += 1
labels = [b.replace('_',' ').title() for b in biome_counts if biome_counts[b]>0]
sizes  = [biome_counts[b] for b in biome_counts if biome_counts[b]>0]
colors_pie = [BIOME_COLOURS[b] for b in biome_counts if biome_counts[b]>0]
ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize':8})
ax.set_title("Biome Coverage Distribution\n(128×128 terrain)", fontsize=10, fontweight='bold')

plt.tight_layout()
save(fig, "fig5_maths_dashboard")


# =============================================================
# FIGURE 6 — Multi-seed terrain comparison
# =============================================================
print("Generating Figure 6 — Multi-Seed Terrain Comparison...")

seeds = [42, 123, 777, 2026]
seeds_label = ["Seed 42\n(main terrain)", "Seed 123", "Seed 777", "Seed 2026"]

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Figure 6 — Procedural Reproducibility: Same Algorithm, Different Seeds\n"
             "Demonstrates infinite unique terrain generation from one system",
             fontsize=12, fontweight='bold')

for idx, (seed, label) in enumerate(zip(seeds, seeds_label)):
    t = Terrain(width=64, depth=64, scale=5.0, height_scale=3.0, octaves=6, seed=seed)
    h = np.array(t.height_grid())
    X_s = np.linspace(-1,1,64); Z_s = np.linspace(-1,1,64)
    XX_s, ZZ_s = np.meshgrid(X_s, Z_s)

    ax = fig.add_subplot(2, 4, idx+1, projection='3d')
    colours_s = np.zeros((*h.shape,3))
    for j in range(64):
        for i in range(64):
            colours_s[j,i] = height_to_colour(h[j,i], h.min(), h.max(), 3.0)
    ax.plot_surface(XX_s, h, ZZ_s, facecolors=colours_s,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_title(label, fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=30, azim=-50)

    # 2D top-down view
    ax2 = fig.add_subplot(2, 4, idx+5)
    bi_img = np.zeros((64,64,3))
    for j in range(64):
        for i in range(64):
            bi_img[j,i] = height_to_colour(h[j,i], h.min(), h.max(), 3.0)
    ax2.imshow(bi_img, origin='lower', interpolation='bilinear')
    ax2.set_title(f"Top-down  |  seed={seed}", fontsize=8.5)
    ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout()
save(fig, "fig6_multi_seed")


# =============================================================
# FIGURE 7 — System Architecture Diagram
# =============================================================
print("Generating Figure 7 — System Architecture...")

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0,13); ax.set_ylim(0,8); ax.axis('off')
fig.suptitle("Figure 7 — Milestone 1 System Architecture\n"
             "Procedural Content Generation: Terrain System",
             fontsize=12, fontweight='bold')

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, w, h, label, sublabel, fc, ec='#333333'):
    ax.add_patch(FancyBboxPatch((x,y), w, h,
                                boxstyle="round,pad=0.1",
                                fc=fc, ec=ec, lw=1.8, zorder=3))
    ax.text(x+w/2, y+h*0.65, label, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='white', zorder=4)
    ax.text(x+w/2, y+h*0.28, sublabel, ha='center', va='center',
            fontsize=7.5, color='#dddddd', zorder=4, style='italic')

def draw_arrow(ax, x1,y1,x2,y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2),
                zorder=2)

# Foundation layer
draw_box(ax, 0.3, 0.4, 3.8, 1.4, "Vector3", "x, y, z | dot | cross\nlength | normalise | lerp", '#1a5276')
draw_box(ax, 4.6, 0.4, 3.8, 1.4, "Matrix4", "4×4 homogeneous\ntranslate | scale | rotate", '#1a5276')
draw_box(ax, 8.9, 0.4, 3.8, 1.4, "Math Utilities", "smoothstep | lerp\nfade | permutation", '#1a5276')

# Noise layer
draw_box(ax, 2.5, 2.4, 8.0, 1.5, "PerlinNoise", "Gradient noise | octave fBm\ncr(p) — colour as function of 3D position", '#117a65')

# Terrain layer
draw_box(ax, 2.0, 4.4, 9.0, 1.5, "Terrain", "height_map | normals | biome classification\ngenerate() | get_vertex() | stats()", '#6e2f8a')

# Output layer
draw_box(ax, 0.3, 6.3, 3.5, 1.2, "3D Surface Plot", "Matplotlib fig1, fig4,\nfig6 — 3D renders", '#1f618d')
draw_box(ax, 4.3, 6.3, 4.0, 1.2, "2D Analysis Maps", "Height map | biome map\nhistogram | figs 2,3,5", '#1f618d')
draw_box(ax, 9.0, 6.3, 3.5, 1.2, "System Report", "Derivations\nstats | analysis", '#1f618d')

# Arrows
draw_arrow(ax, 2.2,  1.8, 3.8,  2.4)   # Vector3 → Noise
draw_arrow(ax, 6.5,  1.8, 6.5,  2.4)   # Matrix4 → Noise
draw_arrow(ax, 10.8, 1.8, 9.0,  2.4)   # Utils → Noise
draw_arrow(ax, 6.5,  3.9, 6.5,  4.4)   # Noise → Terrain
draw_arrow(ax, 3.5,  5.9, 2.0,  7.5)   # Terrain → 3D
draw_arrow(ax, 6.5,  5.9, 6.3,  7.5)   # Terrain → 2D
draw_arrow(ax, 9.5,  5.9, 10.8, 7.5)   # Terrain → Report

# Layer labels
ax.text(0.05, 1.05, "LAYER 1\nFoundations", fontsize=8, color='#1a5276', fontweight='bold')
ax.text(0.05, 3.05, "LAYER 2\nNoise Engine", fontsize=8, color='#117a65', fontweight='bold')
ax.text(0.05, 5.0,  "LAYER 3\nTerrain System", fontsize=8, color='#6e2f8a', fontweight='bold')
ax.text(0.05, 6.85, "LAYER 4\nOutput", fontsize=8, color='#1f618d', fontweight='bold')

plt.tight_layout()
save(fig, "fig7_architecture")

print("\nAll Milestone 1 figures generated in ./figures/")
print("Total: 7 figures")
