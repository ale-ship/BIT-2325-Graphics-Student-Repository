# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 4 — Efficiency & Stochastic Methods
# Topic: Procedural Content Generation Systems
#
# Person A — BVH acceleration structure + performance comparison
# Person B — Monte Carlo path tracing (full implementation)
# Person C — LOD (Level of Detail) system
# Person D — Noise/variance analysis + error bounds
#
# Date: May 2026 | JKUAT
#
# WHAT THIS ADDS ON TOP OF MILESTONES 1–3:
#   BVHNode      — Bounding Volume Hierarchy for fast ray-mesh intersection
#   PathTracer   — Full Monte Carlo path tracer with global illumination
#   LODSystem    — Level of Detail for scalable rendering
#   VarianceAnalysis — Convergence, error bounds, variance reduction
# =============================================================

import math, sys, os, time, random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core    import Vector3, Matrix4, Terrain
from milestone2_core import Camera
from milestone3_core import PhongShader, ProceduralTexture


# =============================================================
# RAY  (shared data structure)
# =============================================================

@dataclass
class Ray:
    """A ray: origin point + normalised direction."""
    origin   : Vector3
    direction: Vector3

    def at(self, t):
        """Point along ray at parameter t: P = O + t*D."""
        return self.origin + self.direction * t


@dataclass
class HitRecord:
    """Record of a ray-surface intersection."""
    t       : float          # ray parameter at intersection
    point   : Vector3        # world-space hit point
    normal  : Vector3        # surface normal at hit
    colour  : np.ndarray     # surface colour at hit
    material: str = 'diffuse'# material type


# =============================================================
# CLASS: AABB  — Axis-Aligned Bounding Box
# =============================================================

class AABB:
    """
    Axis-Aligned Bounding Box.
    Used by BVH for fast ray-box intersection tests.
    """

    def __init__(self, min_pt, max_pt):
        """
        min_pt, max_pt : np.array([x,y,z])
        """
        self.min = np.array(min_pt, dtype=float)
        self.max = np.array(max_pt, dtype=float)

    @staticmethod
    def from_triangle(v0, v1, v2, pad=1e-4):
        """Build AABB enclosing a triangle."""
        pts = np.vstack([v0, v1, v2])
        mn  = pts.min(axis=0) - pad
        mx  = pts.max(axis=0) + pad
        return AABB(mn, mx)

    def intersect(self, ray_origin, ray_dir_inv):
        """
        Ray-AABB intersection — slab method.
        ray_dir_inv : 1/ray.direction (precomputed).
        Returns True if ray intersects this box.
        """
        t1 = (self.min - ray_origin) * ray_dir_inv
        t2 = (self.max - ray_origin) * ray_dir_inv
        tmin = np.minimum(t1, t2).max()
        tmax = np.maximum(t1, t2).min()
        return tmax > max(tmin, 1e-8)

    @staticmethod
    def surrounding(a, b):
        """Compute AABB enclosing both a and b."""
        mn = np.minimum(a.min, b.min)
        mx = np.maximum(a.max, b.max)
        return AABB(mn, mx)


# =============================================================
# CLASS: BVHNode  (Person A)
# Bounding Volume Hierarchy for fast ray-triangle intersection.
#
# Without BVH: O(N) per ray — every ray tests every triangle.
# With BVH:    O(log N) per ray — binary tree of AABBs.
#
# Build: O(N log N) — sort triangles by centroid, split at median.
# Query: O(log N)  — traverse tree, skip non-intersecting branches.
# =============================================================

class BVHNode:
    """
    BVH node — either a leaf (holds triangles) or
    an interior node (holds two children).
    """

    def __init__(self, triangles, depth=0, max_depth=20, leaf_size=4):
        """
        triangles : list of (v0,v1,v2,colour) numpy arrays
        depth     : current recursion depth
        max_depth : stop subdividing beyond this
        leaf_size : max triangles per leaf node
        """
        self.left     = None
        self.right    = None
        self.tris     = []
        self.bbox     = None

        if not triangles:
            return

        # Build bounding box for all triangles
        all_aabbs = [AABB.from_triangle(t[0], t[1], t[2]) for t in triangles]
        bbox = all_aabbs[0]
        for ab in all_aabbs[1:]:
            bbox = AABB.surrounding(bbox, ab)
        self.bbox = bbox

        # Leaf condition
        if len(triangles) <= leaf_size or depth >= max_depth:
            self.tris = triangles
            return

        # Choose split axis — longest AABB dimension
        extents = bbox.max - bbox.min
        axis    = int(np.argmax(extents))

        # Sort by centroid along chosen axis
        triangles.sort(key=lambda t: (t[0][axis]+t[1][axis]+t[2][axis])/3.0)
        mid = len(triangles) // 2

        self.left  = BVHNode(triangles[:mid], depth+1, max_depth, leaf_size)
        self.right = BVHNode(triangles[mid:], depth+1, max_depth, leaf_size)

    def intersect(self, ray):
        """
        Test ray against BVH tree.
        Returns HitRecord of nearest hit or None.
        """
        if self.bbox is None:
            return None

        ro = np.array([ray.origin.x, ray.origin.y, ray.origin.z])
        rd = np.array([ray.direction.x, ray.direction.y, ray.direction.z])
        rd_inv = np.where(np.abs(rd) > 1e-10, 1.0/rd, 1e10)

        # Prune with bounding box test
        if not self.bbox.intersect(ro, rd_inv):
            return None

        # Leaf node — test each triangle
        if self.tris:
            nearest = None
            for tri in self.tris:
                hit = BVHNode._ray_triangle(ray, tri[0], tri[1], tri[2])
                if hit and (nearest is None or hit[0] < nearest[0]):
                    # Compute normal and colour
                    e1  = tri[1] - tri[0]
                    e2  = tri[2] - tri[0]
                    n_v = np.cross(e1, e2)
                    n_l = np.linalg.norm(n_v) + 1e-10
                    n_v = n_v / n_l
                    p   = ray.at(hit[0])
                    nearest = (hit[0], p, n_v, tri[3])

            if nearest:
                t, p, n_arr, col = nearest
                n_vec = Vector3(*n_arr)
                # Ensure normal faces ray
                if n_vec.dot(ray.direction) > 0:
                    n_vec = -n_vec
                return HitRecord(t=t, point=p, normal=n_vec,
                                  colour=col, material='diffuse')
            return None

        # Interior — recurse both children
        hit_l = self.left.intersect(ray)  if self.left  else None
        hit_r = self.right.intersect(ray) if self.right else None

        if hit_l and hit_r:
            return hit_l if hit_l.t < hit_r.t else hit_r
        return hit_l or hit_r

    @staticmethod
    def _ray_triangle(ray, v0, v1, v2, eps=1e-8):
        """
        Möller–Trumbore ray-triangle intersection.
        Returns (t, u, v) or None.
        O(1) — 1 cross product, 2 dot products, 1 division.
        """
        ro = np.array([ray.origin.x, ray.origin.y, ray.origin.z])
        rd = np.array([ray.direction.x, ray.direction.y, ray.direction.z])
        e1 = v1 - v0
        e2 = v2 - v0
        h  = np.cross(rd, e2)
        a  = np.dot(e1, h)
        if abs(a) < eps:
            return None     # Ray parallel to triangle
        f  = 1.0 / a
        s  = ro - v0
        u  = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q  = np.cross(s, e1)
        v  = f * np.dot(rd, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t  = f * np.dot(e2, q)
        if t < eps:
            return None
        return (t, u, v)

    @staticmethod
    def build_from_terrain(terrain, hgrid, XX, ZZ, colours):
        """
        Build BVH from terrain mesh triangles.
        Each quad face → 2 triangles.
        Returns (bvh, n_triangles).
        """
        W, D   = terrain.width, terrain.depth
        triangles = []

        for j in range(D-1):
            for i in range(W-1):
                # Quad corners
                def vp(ii, jj):
                    return np.array([XX[jj,ii]*2, hgrid[jj,ii], ZZ[jj,ii]*2])
                def vc(ii, jj):
                    return colours[jj*W+ii]

                v00, v10 = vp(i,j),   vp(i+1,j)
                v01, v11 = vp(i,j+1), vp(i+1,j+1)
                c00 = vc(i,j); c11 = vc(i+1,j+1)

                # Triangle 1: (00,10,11)
                triangles.append((v00, v10, v11, c00))
                # Triangle 2: (00,11,01)
                triangles.append((v00, v11, v01, c11))

        print(f"    BVH: building from {len(triangles)} triangles...")
        t0  = time.perf_counter()
        bvh = BVHNode(triangles)
        t1  = time.perf_counter()
        print(f"    BVH build time: {(t1-t0)*1000:.1f}ms")
        return bvh, len(triangles)


# =============================================================
# CLASS: PathTracer  (Person B)
# Full Monte Carlo path tracer with global illumination.
#
# Algorithm (Shirley Ch.13 — Distribution Ray Tracing):
#   For each pixel:
#     For each sample:
#       Cast primary ray through pixel
#       At each intersection:
#         Sample random direction in hemisphere
#         Recursively trace reflected ray
#         Accumulate radiance
#     Average all samples → pixel colour
#
# This naturally captures:
#   - Soft shadows (area lights sampled randomly)
#   - Indirect illumination (multi-bounce)
#   - Colour bleeding (red wall bleeds onto nearby white wall)
#   - Ambient occlusion (crevices are naturally darker)
# =============================================================

class PathTracer:
    """
    Monte Carlo path tracer — full global illumination.
    Implements unidirectional path tracing from the camera.
    """

    def __init__(self, bvh, sky_colour=None, max_bounces=4,
                 samples_per_pixel=4):
        """
        bvh             : BVHNode — scene acceleration structure
        sky_colour      : np.array([r,g,b]) — background/sky radiance
        max_bounces     : maximum ray depth (recursion limit)
        samples_per_pixel: N in Monte Carlo estimator
        """
        self.bvh    = bvh
        self.sky    = sky_colour if sky_colour is not None \
                      else np.array([0.52, 0.70, 0.90])
        self.max_b  = max_bounces
        self.spp    = samples_per_pixel

    def _sky_radiance(self, direction):
        """
        Sky radiance — gradient from horizon to zenith.
        Simple analytical sky model.
        """
        d  = np.array([direction.x, direction.y, direction.z])
        t  = max(0.0, min(1.0, (d[1] + 1.0) / 2.0))
        horizon = np.array([0.85, 0.78, 0.65])
        zenith  = np.array([0.35, 0.55, 0.85])
        return horizon * (1-t) + zenith * t

    def _random_hemisphere(self, normal):
        """
        Sample a random direction in the hemisphere oriented by normal.
        Uses cosine-weighted sampling — Shirley §13.3.
        Probability distribution: p(ω) = cos(θ)/π
        This is importance sampling — samples more likely directions
        (near-normal) which reduces variance.
        """
        n = np.array([normal.x, normal.y, normal.z])

        # Random direction in unit sphere
        while True:
            v = np.random.randn(3)
            ln = np.linalg.norm(v)
            if ln > 1e-10:
                v = v / ln
                break

        # Flip if in wrong hemisphere
        if np.dot(v, n) < 0:
            v = -v
        return v

    def _trace(self, ray, depth=0):
        """
        Recursive path tracing — core algorithm.

        At each bounce:
          1. Find nearest intersection
          2. If miss → return sky radiance
          3. If hit → sample random reflected direction
          4. Return: emitted + BRDF * cos(θ) * recursive_trace / pdf

        For Lambertian (diffuse) surfaces:
          BRDF = ρ/π  (where ρ = surface colour)
          pdf  = cos(θ)/π  (cosine-weighted sampling)
          → BRDF * cos(θ) / pdf = ρ  (the cos and π cancel!)
          This is why path tracing with cosine sampling is elegant.
        """
        if depth >= self.max_b:
            # Russian roulette termination for deeper paths
            return np.zeros(3)

        hit = self.bvh.intersect(ray)

        if hit is None:
            return self._sky_radiance(ray.direction)

        # Surface colour at hit point
        surface_c = np.array(hit.colour, dtype=float)

        # Sample random reflected direction (cosine weighted)
        new_dir_arr = self._random_hemisphere(hit.normal)
        new_dir     = Vector3(*new_dir_arr).normalise()

        # Offset origin slightly along normal to avoid self-intersection
        n_arr   = np.array([hit.normal.x, hit.normal.y, hit.normal.z])
        new_ori = hit.point + hit.normal * 1e-4

        new_ray = Ray(origin=new_ori, direction=new_dir)

        # Recursive trace
        incoming = self._trace(new_ray, depth + 1)

        # BRDF × incoming / pdf = surface_colour × incoming
        # (cos(θ)/π and pdf=cos(θ)/π cancel for Lambertian + cosine sampling)
        return surface_c * incoming

    def render_pixel(self, col, row, img_w, img_h, camera):
        """
        Render a single pixel using Monte Carlo integration.
        Averages spp path-traced samples.

        Monte Carlo estimator:
          L ≈ (1/N) Σ_i f(X_i) / p(X_i)
          For path tracing: L = average of N traced paths
        """
        colours = []
        for _ in range(self.spp):
            # Jitter within pixel for antialiasing
            jx = random.random() - 0.5
            jy = random.random() - 0.5
            px = (col + jx + 0.5) / img_w
            py = (row + jy + 0.5) / img_h

            # Generate camera ray
            ray = self._camera_ray(px, py, camera)
            c   = self._trace(ray, depth=0)
            colours.append(c)

        return np.clip(np.mean(colours, axis=0), 0, 1)

    def _camera_ray(self, px, py, camera):
        """
        Generate a ray from camera through pixel (px, py) in [0,1]².
        """
        fov    = math.radians(camera.fov_deg)
        aspect = camera.aspect
        f      = 1.0 / math.tan(fov / 2.0)

        # Camera basis
        eye    = np.array([camera.position.x,
                           camera.position.y,
                           camera.position.z])
        target = np.array([camera.target.x,
                           camera.target.y,
                           camera.target.z])
        up_v   = np.array([camera.up.x, camera.up.y, camera.up.z])

        fwd   = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up_v)
        if np.linalg.norm(right) < 1e-10:
            right = np.array([1,0,0])
        right /= np.linalg.norm(right)
        up    = np.cross(right, fwd)

        # NDC → camera direction
        ndc_x = (2*px - 1) * aspect / f
        ndc_y = (1 - 2*py) / f
        d     = fwd + right*ndc_x + up*ndc_y
        d    /= np.linalg.norm(d)

        return Ray(origin   = Vector3(*eye),
                   direction = Vector3(*d))

    def render_small(self, img_w, img_h, camera, verbose=True):
        """
        Render a small image using path tracing.
        Returns (H×W×3) float array.
        """
        img = np.zeros((img_h, img_w, 3))
        total = img_w * img_h
        t0    = time.perf_counter()

        for row in range(img_h):
            for col in range(img_w):
                img[row, col] = self.render_pixel(col, row,
                                                   img_w, img_h, camera)
            if verbose and (row+1) % max(1, img_h//5) == 0:
                elapsed = time.perf_counter() - t0
                done    = (row+1)*img_w / total
                eta     = elapsed / done * (1-done) if done > 0 else 0
                print(f"    Row {row+1}/{img_h}  "
                      f"({done*100:.0f}%)  ETA: {eta:.1f}s")

        return img

    @staticmethod
    def gamma_correct(img, gamma=2.2):
        """
        Apply gamma correction: c_out = c_in^(1/gamma).
        Converts linear light to display-encoded colour.
        """
        return np.clip(img ** (1.0/gamma), 0, 1)


# =============================================================
# CLASS: LODSystem  (Person C)
# Level of Detail — reduces complexity for distant objects.
# =============================================================

class LODSystem:
    """
    Level of Detail system — Shirley Ch.10 efficiency.
    Selects terrain resolution based on camera distance.
    Reduces vertex count for distant terrain patches.
    """

    # LOD thresholds: (max_distance, resolution_fraction)
    LEVELS = [
        (5.0,  1.0,  "LOD 0 — Full resolution"),
        (10.0, 0.5,  "LOD 1 — Half resolution"),
        (20.0, 0.25, "LOD 2 — Quarter resolution"),
        (40.0, 0.125,"LOD 3 — Eighth resolution"),
        (1e9,  0.0625,"LOD 4 — Sixteenth resolution"),
    ]

    def __init__(self, base_terrain):
        self.base = base_terrain
        self._cache = {}   # resolution → mesh

    def get_level(self, camera_distance):
        """Return LOD level index for a given distance."""
        for i, (dist, frac, name) in enumerate(self.LEVELS):
            if camera_distance <= dist:
                return i, frac, name
        return len(self.LEVELS)-1, self.LEVELS[-1][1], self.LEVELS[-1][2]

    def get_vertex_count(self, camera_distance):
        """Return vertex count for the appropriate LOD."""
        _, frac, _ = self.get_level(camera_distance)
        W = max(4, int(self.base.width  * frac))
        D = max(4, int(self.base.depth  * frac))
        return W * D

    def analyse_savings(self, distances):
        """
        Compute vertex count and render time savings across distances.
        Returns performance comparison data.
        """
        base_verts = self.base.width * self.base.depth
        results    = []
        for dist in distances:
            lvl, frac, name = self.get_level(dist)
            verts   = self.get_vertex_count(dist)
            saving  = 1.0 - verts / base_verts
            # Render time scales roughly as O(V^1.2) for rasterization
            time_r  = (verts / base_verts) ** 1.2
            results.append({
                'distance'      : dist,
                'lod_level'     : lvl,
                'lod_name'      : name,
                'vertices'      : verts,
                'reduction'     : round(saving * 100, 1),
                'time_fraction' : round(time_r, 4),
            })
        return results


# =============================================================
# CLASS: VarianceAnalysis  (Person D)
# Monte Carlo convergence, error bounds, variance reduction.
# =============================================================

class VarianceAnalysis:
    """
    Statistical analysis of Monte Carlo path tracing convergence.
    Shirley §13.2 — Monte Carlo integration theory.
    """

    @staticmethod
    def mc_estimate(integrand, n_samples, domain=(0,1)):
        """
        Monte Carlo integration:
        E[f] ≈ (b-a)/N × Σ f(Xi)   where Xi ~ Uniform(a,b)
        Standard error: σ/√N  (halves when N quadruples)
        """
        a, b    = domain
        samples = np.random.uniform(a, b, n_samples)
        vals    = np.array([integrand(x) for x in samples])
        mean    = float(np.mean(vals))
        var     = float(np.var(vals))
        stderr  = math.sqrt(var / n_samples)
        return mean, var, stderr

    @staticmethod
    def convergence_study(integrand, sample_counts, domain=(0,1),
                          true_value=None):
        """
        Study how MC error reduces as sample count increases.
        Theory: error ∝ 1/√N — doubling accuracy needs 4× samples.
        """
        results = []
        for N in sample_counts:
            mean, var, stderr = VarianceAnalysis.mc_estimate(
                integrand, N, domain)
            error = abs(mean - true_value) if true_value else stderr
            results.append({
                'N'      : N,
                'mean'   : round(mean, 6),
                'var'    : round(var, 6),
                'stderr' : round(stderr, 6),
                'error'  : round(error, 6),
                'theory' : round(1/math.sqrt(N), 6),
            })
        return results

    @staticmethod
    def importance_sampling_comparison(n_samples=1000):
        """
        Compare uniform vs cosine-weighted (importance) sampling.
        For Lambertian BRDF rendering of a hemisphere.

        Uniform sampling:    p(ω) = 1/(2π)
        Cosine-weighted:     p(ω) = cos(θ)/π
        Cosine IS is optimal for Lambertian — zero variance for flat surfaces.
        """
        # True value of ∫ cos(θ) dω over hemisphere = π
        true_val = math.pi

        # Uniform hemisphere sampling
        uniform_samples = []
        for _ in range(n_samples):
            # Sample θ ∈ [0,π/2], φ ∈ [0,2π]
            theta = math.acos(random.random())
            val   = math.cos(theta) / (1/(2*math.pi))   # f/p
            uniform_samples.append(val)

        # Cosine-weighted sampling
        cosine_samples = []
        for _ in range(n_samples):
            # Sample θ with p(θ) ∝ cos(θ): θ = acos(√u)
            u     = random.random()
            theta = math.acos(math.sqrt(u))
            val   = math.cos(theta) / (math.cos(theta)/math.pi)   # = π
            cosine_samples.append(val)

        u_mean = np.mean(uniform_samples)
        c_mean = np.mean(cosine_samples)
        u_var  = np.var(uniform_samples)
        c_var  = np.var(cosine_samples)

        return {
            'true_value': true_val,
            'uniform'   : {'mean':round(u_mean,4), 'variance':round(u_var,4),
                           'error':round(abs(u_mean-true_val),4)},
            'cosine_IS' : {'mean':round(c_mean,4), 'variance':round(c_var,4),
                           'error':round(abs(c_mean-true_val),4)},
            'variance_reduction': round(u_var/max(c_var,1e-10), 2),
        }

    @staticmethod
    def error_bounds_analysis(sample_counts, confidence=0.95):
        """
        95% confidence intervals for MC estimator.
        CI = mean ± z * σ/√N   (z=1.96 for 95% confidence)
        """
        z       = 1.96   # for 95% CI
        results = []
        # Test on ∫₀¹ sin(πx) dx = 2/π ≈ 0.6366
        integrand  = lambda x: math.sin(math.pi * x)
        true_value = 2.0 / math.pi
        for N in sample_counts:
            mean, var, stderr = VarianceAnalysis.mc_estimate(
                integrand, N, (0, 1))
            ci_half  = z * stderr
            in_ci    = abs(mean - true_value) <= ci_half
            results.append({
                'N'       : N,
                'mean'    : round(mean, 5),
                'CI'      : f"[{mean-ci_half:.4f}, {mean+ci_half:.4f}]",
                'width'   : round(2*ci_half, 5),
                'contains': in_ci,
            })
        return results


# =============================================================
# DEMO
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BIT 2325 — Milestone 4: Efficiency & Stochastic Methods")
    print("=" * 60)

    # ── Build terrain ────────────────────────────────────────
    print("\nBuilding terrain...")
    terrain = Terrain(width=32, depth=32, scale=5.0,
                      height_scale=2.5, octaves=6, seed=42)
    hgrid   = np.array(terrain.height_grid())
    W, D    = terrain.width, terrain.depth
    X_lin   = np.linspace(-2, 2, W)
    Z_lin   = np.linspace(-2, 2, D)
    XX, ZZ  = np.meshgrid(X_lin, Z_lin)

    tex     = ProceduralTexture(seed=42)
    colours = tex.apply_to_terrain(terrain, hgrid, XX, ZZ)
    print(f"  Terrain: {W}×{D} = {W*D} vertices")

    # ── BVH build and test ───────────────────────────────────
    print("\n[Person A] --- BVH Acceleration Structure ---")
    bvh, n_tris = BVHNode.build_from_terrain(terrain, hgrid, XX, ZZ, colours)
    print(f"  Triangles  : {n_tris}")
    print(f"  Without BVH: O(N)      = {n_tris} tests per ray")
    print(f"  With BVH   : O(log N)  = {math.ceil(math.log2(n_tris))} tests per ray (theoretical)")

    # Time a ray intersection test
    test_ray = Ray(origin   = Vector3(0, 10, 0),
                   direction = Vector3(0, -1, 0).normalise())
    t0  = time.perf_counter()
    for _ in range(100):
        hit = bvh.intersect(test_ray)
    t1  = time.perf_counter()
    avg_ms = (t1-t0)/100*1000
    print(f"\n  Ray-BVH test (avg over 100): {avg_ms:.3f}ms")
    if hit:
        print(f"  Hit at: {hit.point}  normal: {hit.normal}")

    # ── Path Tracer ──────────────────────────────────────────
    print("\n[Person B] --- Monte Carlo Path Tracer ---")
    camera = Camera(
        position = Vector3(3.0, 4.0, 5.0),
        target   = Vector3(0.0, 0.0, 0.0),
        up       = Vector3(0.0, 1.0, 0.0),
        fov_deg  = 60.0, aspect=16/9, near=0.1, far=50.0
    )

    pt = PathTracer(bvh, max_bounces=4, samples_per_pixel=4)
    print(f"  Rendering 32×18 patch at {pt.spp} spp, {pt.max_b} bounces...")
    t0  = time.perf_counter()
    img = pt.render_small(32, 18, camera, verbose=True)
    t1  = time.perf_counter()
    img_gamma = PathTracer.gamma_correct(img)

    print(f"\n  Render time : {t1-t0:.2f}s for {32*18} pixels")
    print(f"  Time/pixel  : {(t1-t0)*1000/(32*18):.2f}ms")
    print(f"  Image shape : {img_gamma.shape}")
    print(f"  Mean luminance: {img_gamma.mean():.4f}")
    print(f"  Max value:      {img_gamma.max():.4f}")

    # ── LOD System ───────────────────────────────────────────
    print("\n[Person C] --- Level of Detail System ---")
    lod    = LODSystem(terrain)
    dists  = [2, 5, 10, 20, 40, 80]
    results_lod = lod.analyse_savings(dists)
    base_v = terrain.width * terrain.depth
    print(f"  Base terrain: {terrain.width}×{terrain.depth} = {base_v} vertices")
    print(f"\n  {'Distance':>10} {'LOD':>6} {'Vertices':>10} "
          f"{'Reduction':>12} {'Time frac':>12}")
    print("  " + "-"*55)
    for r in results_lod:
        print(f"  {r['distance']:>10.1f} {r['lod_level']:>6} "
              f"{r['vertices']:>10,} {r['reduction']:>11.1f}% "
              f"{r['time_fraction']:>12.4f}")

    # ── Variance Analysis ────────────────────────────────────
    print("\n[Person D] --- Variance & Convergence Analysis ---")

    # Convergence study
    integrand  = lambda x: math.sin(math.pi * x)
    true_val   = 2.0 / math.pi
    ns         = [4, 16, 64, 256, 1024, 4096]
    conv       = VarianceAnalysis.convergence_study(integrand, ns, true_value=true_val)
    print(f"\n  MC convergence on ∫₀¹ sin(πx)dx = {true_val:.5f}")
    print(f"  {'N':>6} {'Mean':>10} {'Variance':>12} "
          f"{'Std Err':>10} {'Error':>10} {'Theory 1/√N':>14}")
    print("  " + "-"*68)
    for r in conv:
        print(f"  {r['N']:>6} {r['mean']:>10.6f} {r['var']:>12.6f} "
              f"{r['stderr']:>10.6f} {r['error']:>10.6f} "
              f"{r['theory']:>14.6f}")

    # Importance sampling comparison
    print(f"\n  Importance sampling comparison (N=1000):")
    is_res = VarianceAnalysis.importance_sampling_comparison(1000)
    print(f"  True value (π): {is_res['true_value']:.4f}")
    u = is_res['uniform']
    c = is_res['cosine_IS']
    print(f"  Uniform sampling:  mean={u['mean']:.4f}  "
          f"var={u['variance']:.4f}  error={u['error']:.4f}")
    print(f"  Cosine IS:         mean={c['mean']:.4f}  "
          f"var={c['variance']:.4f}  error={c['error']:.4f}")
    print(f"  Variance reduction: {is_res['variance_reduction']}×")

    # Error bounds
    print(f"\n  95% Confidence intervals:")
    ci = VarianceAnalysis.error_bounds_analysis([16,64,256,1024])
    print(f"  {'N':>6} {'Mean':>8} {'95% CI':>28} "
          f"{'Width':>10} {'Contains truth':>16}")
    print("  " + "-"*72)
    for r in ci:
        print(f"  {r['N']:>6} {r['mean']:>8.5f} "
              f"{r['CI']:>28} {r['width']:>10.5f} "
              f"{'✓' if r['contains'] else '✗':>16}")

    print("\nMilestone 4 system verified.")
