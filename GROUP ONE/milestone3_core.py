# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 3 — Rendering & Signal Processing
# Topic: Procedural Content Generation Systems
#
# Person A — Phong shading + normal perturbation
# Person B — Procedural texture synthesis
# Person C — Antialiasing (MSAA simulation + comparison)
# Person D — Artifact analysis (aliasing, noise, distortion)
#
# Date: May 2026 | JKUAT
#
# WHAT THIS ADDS ON TOP OF MILESTONES 1 & 2:
#   PhongShader       — full Phong illumination model from scratch
#   ProceduralTexture — multi-layer noise-based surface textures
#   Sampler           — MSAA, stratified, jittered sampling
#   ArtifactAnalysis  — aliasing, noise variance, distortion metrics
# =============================================================

import math
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core    import Vector3, Matrix4, Terrain
from milestone2_core import Camera, TransformPipeline, NumericalAnalysis


# =============================================================
# CLASS: PhongShader  (Person A)
# Full Phong illumination model from first principles.
#
# Phong = Ambient + Diffuse + Specular
#   c = ka*ca + kd*cd*max(0,n·l) + ks*cs*(r·e)^p
# =============================================================

class PhongShader:
    """
    Full Phong shading model — Shirley & Marschner Ch.10.
    Computes per-vertex colour given surface and light properties.
    """

    def __init__(self):
        # Light sources: list of dicts
        self.lights = []
        # Ambient light
        self.ambient_colour    = np.array([0.15, 0.15, 0.20])
        self.ambient_intensity = 1.0

    def add_light(self, position, colour, intensity=1.0,
                  light_type='directional'):
        """
        Add a light source.
        position  : Vector3 — direction (directional) or world pos (point)
        colour    : np.array([r,g,b]) normalised 0-1
        intensity : scalar multiplier
        type      : 'directional' or 'point'
        """
        self.lights.append({
            'position'  : position,
            'colour'    : np.array(colour) * intensity,
            'type'      : light_type,
        })

    def shade(self, position, normal, surface_colour,
              eye_pos, ka=0.3, kd=0.8, ks=0.4, shininess=32):
        """
        Compute Phong illuminated colour at a surface point.

        Phong formula (Shirley Eq. 10.6):
          c = ka*La + Σ_lights [ kd*Ld*max(0, n·l) + ks*Ls*(r·e)^p ]

        Where:
          n = surface normal (unit vector)
          l = unit vector from surface toward the light
          r = reflection of l about n: r = 2(n·l)n - l
          e = unit vector from surface toward eye/camera
          p = shininess exponent (controls highlight sharpness)

        Parameters:
          position        : Vector3 — surface point in world space
          normal          : Vector3 — unit surface normal
          surface_colour  : np.array([r,g,b]) — base diffuse colour
          eye_pos         : Vector3 — camera position
          ka, kd, ks      : ambient, diffuse, specular coefficients
          shininess       : Phong exponent p
        """
        n   = normal.normalise()
        e   = (eye_pos - position).normalise()
        c   = np.array(surface_colour, dtype=float)

        # ── Ambient term ──────────────────────────────────────
        ambient = ka * self.ambient_colour * c

        # ── Diffuse + Specular per light ──────────────────────
        diffuse_total  = np.zeros(3)
        specular_total = np.zeros(3)

        for light in self.lights:
            lp = light['position']
            lc = light['colour']

            # Light direction
            if light['type'] == 'directional':
                l = lp.normalise()   # lp is already a direction
            else:
                l = (lp - position).normalise()   # toward point light

            # Dot product: n · l (Lambert's cosine law)
            ndotl = max(0.0, float(n.dot(l)))

            if ndotl > 0:
                # Diffuse — kd * Ld * (n·l) * surface_colour
                diffuse_total += kd * lc * ndotl * c

                # Reflection vector: r = 2(n·l)n - l
                r_vec  = (n * (2.0 * ndotl)) - l
                r_norm = r_vec.normalise()

                # Specular — ks * Ls * (r·e)^p
                rdote  = max(0.0, float(r_norm.dot(e)))
                spec   = ks * lc * (rdote ** shininess)
                specular_total += spec

        result = ambient + diffuse_total + specular_total
        return np.clip(result, 0.0, 1.0)

    def shade_array(self, positions, normals, colours,
                    eye_pos, material_props=None):
        """
        Shade an array of vertices efficiently.
        Returns Nx3 array of shaded colours.
        Uses vectorised numpy operations for speed.

        This is what gets called per-frame in the rendering pipeline.
        """
        N   = len(positions)
        out = np.zeros((N, 3))

        # Default material
        if material_props is None:
            material_props = {
                'ka': 0.30, 'kd': 0.75, 'ks': 0.35, 'shininess': 32
            }
        ka = material_props['ka']
        kd = material_props['kd']
        ks = material_props['ks']
        p  = material_props['shininess']

        # Eye direction array (E - P) normalised
        ep = np.array([eye_pos.x, eye_pos.y, eye_pos.z])
        E  = ep - positions                             # (N,3)
        E  = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

        # Ambient
        amb = ka * self.ambient_colour * colours        # (N,3)
        out += amb

        for light in self.lights:
            lp = light['position']
            lc = light['colour']

            if light['type'] == 'directional':
                L = np.array([lp.x, lp.y, lp.z])
                L = L / (np.linalg.norm(L) + 1e-10)
                L = np.tile(L, (N, 1))                  # (N,3)
            else:
                lp_arr = np.array([lp.x, lp.y, lp.z])
                L      = lp_arr - positions
                L      = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-10)

            # n · l
            NdotL = np.sum(normals * L, axis=1)         # (N,)
            NdotL = np.clip(NdotL, 0.0, 1.0)

            # Diffuse
            diff = kd * NdotL[:, None] * lc * colours   # (N,3)
            out += diff

            # Reflection: r = 2(n·l)n - l
            R     = 2.0 * NdotL[:, None] * normals - L  # (N,3)
            R_len = np.linalg.norm(R, axis=1, keepdims=True) + 1e-10
            R     = R / R_len

            # r · e
            RdotE = np.sum(R * E, axis=1)
            RdotE = np.clip(RdotE, 0.0, 1.0)
            spec  = ks * lc * (RdotE[:, None] ** p)
            out  += spec

        return np.clip(out, 0.0, 1.0)


# =============================================================
# CLASS: ProceduralTexture  (Person B)
# Multi-layer noise-based surface texture synthesis.
#
# Generates per-pixel colour from mathematical functions.
# No image files — purely computed from position.
# =============================================================

class ProceduralTexture:
    """
    Procedural texture synthesis — Shirley & Marschner §11.5.
    Defines cr(p) — colour as a function of 3D position p.
    No texture memory required — infinite resolution.
    """

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        # Pre-generate permutation table (Perlin-style)
        perm = np.arange(256, dtype=int)
        np.random.shuffle(perm)
        self.perm = np.concatenate([perm, perm])  # doubled

    def _hash(self, ix, iy):
        """Consistent hash for grid point (ix, iy)."""
        return self.perm[(ix + self.perm[iy % 256]) % 256]

    def _smooth(self, t):
        """Quintic smoothstep — same as PerlinNoise._fade()."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _noise2d(self, x, y):
        """Simple value noise in 2D."""
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        dx, dy = x - x0, y - y0
        # Values at corners
        v00 = (self._hash(x0,   y0)   / 255.0) * 2 - 1
        v10 = (self._hash(x0+1, y0)   / 255.0) * 2 - 1
        v01 = (self._hash(x0,   y0+1) / 255.0) * 2 - 1
        v11 = (self._hash(x0+1, y0+1) / 255.0) * 2 - 1
        # Smooth interpolate
        u, v = self._smooth(dx), self._smooth(dy)
        return (v00*(1-u)*(1-v) + v10*u*(1-v) +
                v01*(1-u)*v     + v11*u*v)

    def _fbm(self, x, y, octaves=4, persistence=0.5, lacunarity=2.0):
        """Fractal Brownian Motion — same principle as terrain fBm."""
        value = 0.0; amp = 1.0; freq = 1.0; max_v = 0.0
        for _ in range(octaves):
            value += self._noise2d(x*freq, y*freq) * amp
            max_v += amp
            amp   *= persistence
            freq  *= lacunarity
        return value / max_v

    # ── Texture recipes ───────────────────────────────────────

    def sand_texture(self, x, z):
        """
        Desert sand — warm base with subtle ripple variation.
        Simulates wind-blown sand grain patterns.
        cr(p) = base_sand + noise_ripple + micro_grain
        """
        # Large-scale ripple
        ripple = self._fbm(x * 1.8, z * 1.8, octaves=3) * 0.12
        # Micro grain
        grain  = self._noise2d(x * 22, z * 22) * 0.04
        # Anisotropic streaks (wind direction)
        streak = math.sin(x * 6.0 + self._fbm(x, z, 2) * 1.5) * 0.05

        r = np.clip(0.87 + ripple + grain + streak, 0, 1)
        g = np.clip(0.72 + ripple * 0.8 + grain * 0.7, 0, 1)
        b = np.clip(0.48 + ripple * 0.5, 0, 1)
        return np.array([r, g, b])

    def grass_texture(self, x, z, height_variation=0.0):
        """
        Grass — green base with blade variation and dry patches.
        cr(p) = base_green * (1 + noise_patch) + dry_blend
        """
        patch  = self._fbm(x * 3.5, z * 3.5, octaves=4) * 0.15
        blade  = self._noise2d(x * 40, z * 40) * 0.05
        dry    = max(0, self._fbm(x * 0.8, z * 0.8, octaves=2))

        r = np.clip(0.28 + patch * 0.3 + blade + dry * 0.18, 0, 1)
        g = np.clip(0.52 + patch * 0.5 + blade + dry * 0.05, 0, 1)
        b = np.clip(0.15 + patch * 0.1, 0, 1)
        return np.array([r, g, b])

    def rock_texture(self, x, z, y=0.0):
        """
        Rock face — layered grey with strata and weathering.
        Simulates geological strata via horizontal stripe noise.
        """
        # Horizontal strata
        strata = math.sin(y * 8.0 + self._noise2d(x*2, z*2) * 1.5) * 0.08
        # Surface crack network
        crack  = abs(self._fbm(x * 6, z * 6, octaves=5)) * 0.12
        # Base grey with slight warmth
        base   = 0.45 + strata - crack * 0.5
        warm   = self._noise2d(x * 3, z * 3) * 0.04

        r = np.clip(base + warm * 0.08, 0, 1)
        g = np.clip(base + warm * 0.03, 0, 1)
        b = np.clip(base - warm * 0.02, 0, 1)
        return np.array([r, g, b])

    def snow_texture(self, x, z):
        """
        Snow — near-white with subtle blue-grey shadows in hollows.
        """
        grain  = self._noise2d(x * 30, z * 30) * 0.03
        shadow = self._fbm(x * 2, z * 2, octaves=3) * 0.06
        r = np.clip(0.94 + grain - shadow * 0.3, 0, 1)
        g = np.clip(0.95 + grain - shadow * 0.2, 0, 1)
        b = np.clip(0.97 + grain - shadow * 0.1, 0, 1)
        return np.array([r, g, b])

    def water_texture(self, x, z, time=0.0):
        """
        Animated water — ripple pattern varies with time.
        time parameter enables animation in Milestone 5.
        """
        ripple = (math.sin(x * 5.0 + time * 2.0) *
                  math.cos(z * 4.5 - time * 1.5)) * 0.06
        foam   = max(0, self._fbm(x * 8, z * 8, octaves=3) - 0.3) * 0.15
        r = np.clip(0.10 + ripple * 0.3 + foam, 0, 1)
        g = np.clip(0.38 + ripple * 0.4 + foam, 0, 1)
        b = np.clip(0.72 + ripple * 0.5 + foam, 0, 1)
        return np.array([r, g, b])

    def apply_to_terrain(self, terrain, hgrid, XX, ZZ):
        """
        Apply procedural textures to all terrain vertices.
        Returns Nx3 colour array ready for PyVista.
        """
        W, D   = terrain.width, terrain.depth
        colours = np.zeros((D*W, 3))
        hs = terrain.height_scale

        for j in range(D):
            for i in range(W):
                h  = hgrid[j, i]
                hn = h / hs
                x, z = float(XX[j, i]), float(ZZ[j, i])
                y    = float(h)

                # Height-based texture selection with blending
                if hn < -0.30:
                    c = self.water_texture(x, z)
                elif hn < -0.05:
                    t  = (hn + 0.30) / 0.25
                    c  = (1-t)*self.water_texture(x, z) + t*self.sand_texture(x, z)
                elif hn < 0.05:
                    c  = self.sand_texture(x, z)
                elif hn < 0.15:
                    t  = (hn - 0.05) / 0.10
                    c  = (1-t)*self.sand_texture(x,z) + t*self.grass_texture(x,z)
                elif hn < 0.45:
                    c  = self.grass_texture(x, z, hn)
                elif hn < 0.60:
                    t  = (hn - 0.45) / 0.15
                    c  = (1-t)*self.grass_texture(x,z) + t*self.rock_texture(x,z,y)
                elif hn < 0.80:
                    c  = self.rock_texture(x, z, y)
                else:
                    t  = (hn - 0.80) / 0.20
                    c  = (1-t)*self.rock_texture(x,z,y) + t*self.snow_texture(x,z)

                colours[j*W+i] = np.clip(c, 0, 1)
        return colours


# =============================================================
# CLASS: Sampler  (Person C)
# Antialiasing via multiple sampling strategies.
#
# Shirley §8.3 — antialiasing through box filtering.
# Shirley §9.5 — sampling theory and Nyquist theorem.
# =============================================================

class Sampler:
    """
    Sampling strategies for antialiasing.
    Demonstrates MSAA, stratified, and jittered sampling.
    """

    @staticmethod
    def regular_grid(n):
        """
        Regular n×n grid samples within a pixel.
        Simple but produces structured aliasing patterns.
        Returns list of (dx, dy) offsets in [-0.5, 0.5].
        """
        samples = []
        for i in range(n):
            for j in range(n):
                dx = (i + 0.5) / n - 0.5
                dy = (j + 0.5) / n - 0.5
                samples.append((dx, dy))
        return samples

    @staticmethod
    def stratified(n, jitter=True, seed=42):
        """
        Stratified (jittered) sampling — Shirley §13.4.
        Divides pixel into n×n strata, one sample per stratum.
        Jitter adds random offset within each stratum.
        Breaks up regular grid aliasing patterns.
        """
        rng     = np.random.default_rng(seed)
        samples = []
        for i in range(n):
            for j in range(n):
                if jitter:
                    dx = (i + rng.random()) / n - 0.5
                    dy = (j + rng.random()) / n - 0.5
                else:
                    dx = (i + 0.5) / n - 0.5
                    dy = (j + 0.5) / n - 0.5
                samples.append((dx, dy))
        return samples

    @staticmethod
    def poisson_disk(n_samples, seed=42):
        """
        Poisson disk sampling — minimum distance between samples.
        Best distribution for antialiasing — avoids clumping.
        Uses Mitchell's best-candidate algorithm.
        """
        rng        = np.random.default_rng(seed)
        samples    = []
        candidates = 20   # candidates per sample (quality vs speed)

        for _ in range(n_samples):
            best      = None
            best_dist = -1
            for _ in range(candidates):
                cx = rng.random() - 0.5
                cy = rng.random() - 0.5
                if not samples:
                    best = (cx, cy); break
                min_d = min(math.sqrt((cx-sx)**2+(cy-sy)**2)
                            for sx,sy in samples)
                if min_d > best_dist:
                    best_dist = min_d
                    best = (cx, cy)
            samples.append(best)
        return samples

    @staticmethod
    def box_filter(samples):
        """
        Box filter — average all samples equally.
        c_pixel = (1/N) Σ c_i
        Shirley §9.3 — simplest reconstruction filter.
        """
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @staticmethod
    def tent_filter(samples, offsets):
        """
        Tent (linear) filter — weight by distance from centre.
        f(x) = 1 - |x| for |x| < 1.
        C0 continuous — smoother than box.
        """
        if not samples:
            return 0.0
        total_w = 0.0; total_v = 0.0
        for v, (dx, dy) in zip(samples, offsets):
            d = math.sqrt(dx**2 + dy**2) / 0.5   # normalise to [0,1]
            w = max(0.0, 1.0 - d)
            total_v += v * w; total_w += w
        return total_v / (total_w + 1e-10)

    @staticmethod
    def gaussian_filter(samples, offsets, sigma=0.5):
        """
        Gaussian filter — bell curve weights.
        f(x) = exp(-x²/2σ²)
        Shirley §9.3 — best quality AA filter.
        """
        if not samples:
            return 0.0
        total_w = 0.0; total_v = 0.0
        for v, (dx, dy) in zip(samples, offsets):
            d = math.sqrt(dx**2 + dy**2)
            w = math.exp(-(d**2) / (2 * sigma**2))
            total_v += v * w; total_w += w
        return total_v / (total_w + 1e-10)

    @staticmethod
    def nyquist_check(polygon_density, render_width):
        """
        Nyquist theorem check — Shirley §9.5.
        sample_rate >= 2 * max_spatial_frequency.
        Returns minimum supersample factor needed.
        """
        freq    = polygon_density / render_width
        min_sr  = 2 * freq
        ss      = math.ceil(min_sr)
        return {
            'frequency'   : round(freq, 4),
            'min_sample_rate': round(min_sr, 4),
            'supersample' : max(1, ss),
            'satisfies_nyquist': ss <= 8,
        }


# =============================================================
# CLASS: ArtifactAnalysis  (Person D)
# Measures aliasing, noise variance, and distortion.
# =============================================================

class ArtifactAnalysis:
    """
    Quantitative analysis of rendering artifacts.
    Shirley §8.3 — antialiasing theory.
    """

    @staticmethod
    def aliasing_error(signal_high_res, signal_low_res):
        """
        Measure aliasing as RMSE between high-res and low-res renders.
        High RMSE = severe aliasing.
        """
        hr = np.array(signal_high_res, dtype=float)
        lr = np.array(signal_low_res,  dtype=float)
        if len(hr) != len(lr):
            lr = np.interp(np.linspace(0,1,len(hr)),
                          np.linspace(0,1,len(lr)), lr)
        return float(np.sqrt(np.mean((hr - lr)**2)))

    @staticmethod
    def noise_variance(samples):
        """
        Variance of a set of samples.
        High variance = noisy render.
        σ² = (1/N) Σ (xi - μ)²
        """
        arr = np.array(samples)
        return float(np.var(arr))

    @staticmethod
    def noise_snr(signal, noise_samples):
        """
        Signal-to-noise ratio in dB.
        SNR = 10 * log10(signal² / noise_variance)
        Higher = cleaner render.
        """
        var = ArtifactAnalysis.noise_variance(noise_samples)
        if var < 1e-12:
            return float('inf')
        return 10 * math.log10((signal**2) / var)

    @staticmethod
    def mse(img_a, img_b):
        """Mean Squared Error between two images."""
        a = np.array(img_a, dtype=float)
        b = np.array(img_b, dtype=float)
        return float(np.mean((a - b)**2))

    @staticmethod
    def psnr(img_a, img_b, max_val=1.0):
        """
        Peak Signal-to-Noise Ratio.
        PSNR = 10 * log10(MAX² / MSE)
        >40 dB = excellent quality
        30-40 dB = good
        <30 dB = noticeable artifacts
        """
        m = ArtifactAnalysis.mse(img_a, img_b)
        if m < 1e-12:
            return float('inf')
        return 10 * math.log10((max_val**2) / m)

    @staticmethod
    def compare_sampling_strategies(signal_fn, x_range, n_samples=64):
        """
        Compare aliasing error across sampling strategies.
        signal_fn : callable (x) → float
        """
        import time
        xs_ref = np.linspace(x_range[0], x_range[1], n_samples*8)
        ref    = np.array([signal_fn(x) for x in xs_ref])

        results = {}
        for strategy_name, xs in [
            ('Regular 1×',  np.linspace(x_range[0], x_range[1], n_samples)),
            ('Regular 4×',  np.linspace(x_range[0], x_range[1], n_samples*4)),
            ('Stratified',  np.sort(np.random.uniform(*x_range, n_samples))),
        ]:
            t0     = time.perf_counter()
            vals   = np.array([signal_fn(x) for x in xs])
            t1     = time.perf_counter()
            interp = np.interp(xs_ref, xs, vals)
            rmse   = float(np.sqrt(np.mean((ref - interp)**2)))
            results[strategy_name] = {
                'rmse'    : round(rmse, 6),
                'time_ms' : round((t1-t0)*1000, 3),
                'samples' : len(xs),
            }
        return results


# =============================================================
# DEMO
# =============================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  BIT 2325 — Milestone 3: Rendering & Signal Processing")
    print("=" * 58)

    # ── Phong Shader demo ────────────────────────────────────
    print("\n[Person A] --- Phong Shader ---")
    shader = PhongShader()
    shader.add_light(Vector3(5, 10, 5),  np.array([1.0, 0.95, 0.85]), 1.2)
    shader.add_light(Vector3(-4, 6, -3), np.array([0.6, 0.7, 1.0]),  0.4)

    eye   = Vector3(4, 5, 8)
    pos   = Vector3(0.3, 1.2, 0.1)
    norm  = Vector3(0.1, 0.9, 0.2).normalise()
    sand  = np.array([0.87, 0.72, 0.50])

    c_phong = shader.shade(pos, norm, sand, eye,
                            ka=0.3, kd=0.75, ks=0.4, shininess=32)
    print(f"  Surface colour (sand):  {np.round(sand, 3)}")
    print(f"  Phong shaded colour:    {np.round(c_phong, 3)}")

    # Test vectorised version
    N = 1000
    positions = np.random.randn(N, 3)
    normals   = np.random.randn(N, 3)
    normals   = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    colours   = np.random.rand(N, 3)
    shaded    = shader.shade_array(positions, normals, colours, eye)
    print(f"  Vectorised shade of {N} vertices: output shape={shaded.shape}  ✓")

    # ── Procedural Texture demo ──────────────────────────────
    print("\n[Person B] --- Procedural Textures ---")
    tex = ProceduralTexture(seed=42)
    test_pts = [(0.3, 0.5), (1.2, 0.8), (-0.5, 1.1)]
    for x, z in test_pts:
        s = tex.sand_texture(x, z)
        g = tex.grass_texture(x, z)
        r = tex.rock_texture(x, z, 0.5)
        print(f"  ({x:.1f},{z:.1f})  sand={np.round(s,3)}  "
              f"grass={np.round(g,3)}  rock={np.round(r,3)}")

    # ── Sampler demo ─────────────────────────────────────────
    print("\n[Person C] --- Sampling Strategies ---")
    regular    = Sampler.regular_grid(4)
    stratified = Sampler.stratified(4, jitter=True)
    poisson    = Sampler.poisson_disk(16)
    print(f"  Regular 4×4:     {len(regular)} samples")
    print(f"  Stratified 4×4:  {len(stratified)} samples")
    print(f"  Poisson disk 16: {len(poisson)} samples")

    # Nyquist check for our terrain
    nyq = Sampler.nyquist_check(polygon_density=200*200, render_width=1920)
    print(f"\n  Nyquist check (200×200 terrain, 1920px render):")
    print(f"    frequency      = {nyq['frequency']}")
    print(f"    min_sample_rate= {nyq['min_sample_rate']}")
    print(f"    supersample    = {nyq['supersample']}×")

    # Filter comparison
    vals    = [0.8, 0.6, 0.9, 0.4]
    offsets = regular[:4]
    print(f"\n  Filter comparison on 4 samples {vals}:")
    print(f"    Box filter:      {Sampler.box_filter(vals):.4f}")
    print(f"    Tent filter:     {Sampler.tent_filter(vals, offsets):.4f}")
    print(f"    Gaussian filter: {Sampler.gaussian_filter(vals, offsets):.4f}")

    # ── Artifact Analysis demo ───────────────────────────────
    print("\n[Person D] --- Artifact Analysis ---")
    # High freq signal — edge aliasing
    sig = lambda x: math.sin(x * 20) * 0.5 + 0.5
    results = ArtifactAnalysis.compare_sampling_strategies(sig, (0, 1))
    print("  Aliasing error comparison (high-frequency signal):")
    for name, r in results.items():
        print(f"    {name:<15} RMSE={r['rmse']:.6f}  "
              f"time={r['time_ms']:.3f}ms  samples={r['samples']}")

    # PSNR test
    img_a = np.random.rand(100)
    img_b = img_a + np.random.randn(100) * 0.05
    psnr  = ArtifactAnalysis.psnr(img_a, img_b)
    print(f"\n  PSNR test (5% noise):  {psnr:.2f} dB  "
          f"({'good' if psnr > 30 else 'noisy'})")

    # Noise variance
    mc_samples = np.random.rand(1000) * 0.1
    var  = ArtifactAnalysis.noise_variance(mc_samples)
    snr  = ArtifactAnalysis.noise_snr(0.5, mc_samples)
    print(f"  Monte Carlo variance:  {var:.6f}")
    print(f"  Monte Carlo SNR:       {snr:.2f} dB")

    print("\nMilestone 3 system verified.")
