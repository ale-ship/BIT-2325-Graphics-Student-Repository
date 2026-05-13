# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 1 — Representation & Foundations
# Topic: Procedural Content Generation Systems
#
# System: Procedural Terrain Generator
# Authors:  Esther Achieng Otieno (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu (SCT221-C004-0217/2024)
#           Wendy Wachira (SCT221-C004-0194/2024)
#           Alexander Somba (SCT221-C004-0680/2023)
# Date  : May 2026 | JKUAT
# =============================================================
#
# WHAT THIS FILE CONTAINS:
#   Vector3   — 3D vector with all mathematical operations
#   Matrix4   — 4x4 homogeneous transformation matrix
#   Noise     — Perlin-style gradient noise from scratch
#   Terrain   — Procedural terrain built on the above
#
# NO black-box use of core algorithms.
# All maths implemented from first principles.
# =============================================================

import math
import random


# =============================================================
# CLASS: Vector3
# Mathematical 3D vector — the fundamental geometric primitive.
#
# A vector v = (x, y, z) represents a point or direction in
# 3D space. All operations are derived from first principles.
# =============================================================

class Vector3:
    """3D vector with full mathematical operations."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ── Arithmetic operators ──────────────────────────────────

    def __add__(self, other):
        """Vector addition: (a+b)_i = a_i + b_i"""
        return Vector3(self.x + other.x,
                       self.y + other.y,
                       self.z + other.z)

    def __sub__(self, other):
        """Vector subtraction: (a-b)_i = a_i - b_i"""
        return Vector3(self.x - other.x,
                       self.y - other.y,
                       self.z - other.z)

    def __mul__(self, scalar):
        """Scalar multiplication: s*v = (s*x, s*y, s*z)"""
        return Vector3(self.x * scalar,
                       self.y * scalar,
                       self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """Scalar division"""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar,
                       self.y / scalar,
                       self.z / scalar)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __repr__(self):
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    # ── Core vector operations ────────────────────────────────

    def dot(self, other):
        """
        Dot product: a · b = ax*bx + ay*by + az*bz
        Geometric meaning: |a||b|cos(θ)
        Used in: lighting, projection, angle computation
        """
        return (self.x * other.x +
                self.y * other.y +
                self.z * other.z)

    def cross(self, other):
        """
        Cross product: a × b = (ay*bz - az*by,
                                 az*bx - ax*bz,
                                 ax*by - ay*bx)
        Geometric meaning: vector perpendicular to both a and b
        Used in: computing surface normals for terrain
        """
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        """
        Euclidean length (magnitude):
        |v| = sqrt(x² + y² + z²)
        Derived from Pythagorean theorem in 3D.
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def length_squared(self):
        """|v|² — faster than length() when only comparison needed"""
        return self.x**2 + self.y**2 + self.z**2

    def normalise(self):
        """
        Unit vector: v̂ = v / |v|
        Produces a vector with the same direction but length = 1.
        Essential for lighting normals — they must be unit vectors.
        """
        mag = self.length()
        if mag < 1e-10:
            return Vector3(0, 0, 0)   # zero vector edge case
        return self / mag

    def lerp(self, other, t):
        """
        Linear interpolation: lerp(a, b, t) = a + t*(b - a)
        t=0 returns a, t=1 returns b, t=0.5 returns midpoint.
        Used in: noise smoothing, animation blending.
        """
        return self + (other - self) * t

    def reflect(self, normal):
        """
        Reflection vector: r = v - 2*(v·n)*n
        Reflects this vector about a normal n.
        Used in: specular lighting, ray tracing.
        """
        n = normal.normalise()
        return self - n * (2.0 * self.dot(n))

    def distance_to(self, other):
        """Euclidean distance between two points"""
        return (self - other).length()

    def to_list(self):
        return [self.x, self.y, self.z]

    def to_tuple(self):
        return (self.x, self.y, self.z)


# =============================================================
# CLASS: Matrix4
# 4×4 homogeneous transformation matrix.
#
# Using homogeneous coordinates [x, y, z, w] allows translation,
# rotation, and scale to all be expressed as matrix multiplication.
# This is the standard representation in all graphics pipelines.
# =============================================================

class Matrix4:
    """4×4 homogeneous transformation matrix."""

    def __init__(self, data=None):
        """
        data: flat list of 16 floats, row-major order.
        Index mapping: data[row*4 + col]
        """
        if data is None:
            # Default: identity matrix
            self.data = [
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1
            ]
        else:
            if len(data) != 16:
                raise ValueError("Matrix4 requires exactly 16 values")
            self.data = [float(v) for v in data]

    def get(self, row, col):
        return self.data[row * 4 + col]

    def set(self, row, col, value):
        self.data[row * 4 + col] = float(value)

    def __repr__(self):
        rows = []
        for r in range(4):
            row = [f"{self.get(r,c):8.4f}" for c in range(4)]
            rows.append("[ " + "  ".join(row) + " ]")
        return "\n".join(rows)

    def __mul__(self, other):
        """
        Matrix multiplication: C = A × B
        C_ij = Σ_k A_ik * B_kj

        This is how transformation pipelines are built —
        combining translate, rotate, scale into one matrix.
        """
        if isinstance(other, Matrix4):
            result = Matrix4()
            for r in range(4):
                for c in range(4):
                    val = 0.0
                    for k in range(4):
                        val += self.get(r, k) * other.get(k, c)
                    result.set(r, c, val)
            return result

        elif isinstance(other, Vector3):
            # Apply matrix to a 3D point (w=1) or direction (w=0)
            x, y, z, w = other.x, other.y, other.z, 1.0
            nx = self.get(0,0)*x + self.get(0,1)*y + self.get(0,2)*z + self.get(0,3)*w
            ny = self.get(1,0)*x + self.get(1,1)*y + self.get(1,2)*z + self.get(1,3)*w
            nz = self.get(2,0)*x + self.get(2,1)*y + self.get(2,2)*z + self.get(2,3)*w
            nw = self.get(3,0)*x + self.get(3,1)*y + self.get(3,2)*z + self.get(3,3)*w
            if abs(nw) > 1e-10:
                return Vector3(nx/nw, ny/nw, nz/nw)
            return Vector3(nx, ny, nz)

        raise TypeError(f"Cannot multiply Matrix4 by {type(other)}")

    def transpose(self):
        """
        Transpose: M^T where M^T_ij = M_ji
        Used for: converting between row/column major,
                  computing inverse of rotation matrices.
        """
        result = Matrix4()
        for r in range(4):
            for c in range(4):
                result.set(c, r, self.get(r, c))
        return result

    def determinant(self):
        """
        3×3 cofactor expansion for the 4×4 determinant.
        Used to check if matrix is invertible (det ≠ 0).
        """
        d = self.data
        # Cofactor expansion along first row
        def det3(a,b,c,d,e,f,g,h,i):
            return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

        return (d[0]*det3(d[5],d[6],d[7],d[9],d[10],d[11],d[13],d[14],d[15])
               -d[1]*det3(d[4],d[6],d[7],d[8],d[10],d[11],d[12],d[14],d[15])
               +d[2]*det3(d[4],d[5],d[7],d[8],d[9], d[11],d[12],d[13],d[15])
               -d[3]*det3(d[4],d[5],d[6],d[8],d[9], d[10],d[12],d[13],d[14]))

    # ── Factory methods — standard transformation matrices ────

    @staticmethod
    def identity():
        """Identity matrix I — applying it changes nothing."""
        return Matrix4()

    @staticmethod
    def translation(tx, ty, tz):
        """
        Translation matrix:
        | 1  0  0  tx |
        | 0  1  0  ty |
        | 0  0  1  tz |
        | 0  0  0   1 |
        Moves a point by (tx, ty, tz).
        Requires homogeneous coordinates — impossible with 3×3.
        """
        m = Matrix4()
        m.set(0,3, tx)
        m.set(1,3, ty)
        m.set(2,3, tz)
        return m

    @staticmethod
    def scale(sx, sy, sz):
        """
        Scale matrix:
        | sx  0   0   0 |
        |  0  sy  0   0 |
        |  0  0   sz  0 |
        |  0  0   0   1 |
        """
        m = Matrix4()
        m.set(0,0, sx)
        m.set(1,1, sy)
        m.set(2,2, sz)
        return m

    @staticmethod
    def rotation_x(angle_deg):
        """
        Rotation about X-axis by angle θ:
        | 1    0      0    0 |
        | 0  cos θ  -sin θ  0 |
        | 0  sin θ   cos θ  0 |
        | 0    0      0    1 |
        """
        t  = math.radians(angle_deg)
        c, s = math.cos(t), math.sin(t)
        m = Matrix4()
        m.set(1,1,  c); m.set(1,2, -s)
        m.set(2,1,  s); m.set(2,2,  c)
        return m

    @staticmethod
    def rotation_y(angle_deg):
        """
        Rotation about Y-axis by angle θ:
        |  cos θ  0  sin θ  0 |
        |    0    1    0    0 |
        | -sin θ  0  cos θ  0 |
        |    0    0    0    1 |
        """
        t  = math.radians(angle_deg)
        c, s = math.cos(t), math.sin(t)
        m = Matrix4()
        m.set(0,0,  c); m.set(0,2,  s)
        m.set(2,0, -s); m.set(2,2,  c)
        return m

    @staticmethod
    def rotation_z(angle_deg):
        """Rotation about Z-axis."""
        t  = math.radians(angle_deg)
        c, s = math.cos(t), math.sin(t)
        m = Matrix4()
        m.set(0,0,  c); m.set(0,1, -s)
        m.set(1,0,  s); m.set(1,1,  c)
        return m


# =============================================================
# CLASS: PerlinNoise
# Gradient noise from first principles — Ken Perlin (1985).
#
# Core idea:
#   1. Define a random gradient vector at each integer grid point
#   2. For any point p, find its surrounding grid cell
#   3. Compute dot product of gradient with offset vector
#   4. Smoothly interpolate between the four corners
#
# This produces smooth, continuous, natural-looking variation —
# exactly what terrain generation needs.
# =============================================================

class PerlinNoise:
    """
    2D Perlin gradient noise implementation from scratch.
    Reference: Perlin, K. (1985). An image synthesizer. SIGGRAPH.
    """

    # Gradient vectors — unit vectors pointing in 8 directions
    # These give the noise its directional character
    GRADIENTS = [
        ( 1,  1), (-1,  1), ( 1, -1), (-1, -1),
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
    ]

    def __init__(self, seed=42):
        random.seed(seed)
        # Permutation table: random shuffle of 0-255
        # Doubled to avoid index wrapping issues
        perm = list(range(256))
        random.shuffle(perm)
        self.perm = perm + perm   # length 512
        self.seed = seed

    def _gradient(self, ix, iy):
        """
        Return the gradient vector at integer grid point (ix, iy).
        Uses permutation table to ensure consistency.
        """
        idx = self.perm[(ix + self.perm[iy % 256]) % 256] % 8
        return self.GRADIENTS[idx]

    @staticmethod
    def _fade(t):
        """
        Smoothstep function: f(t) = 6t⁵ - 15t⁴ + 10t³
        This is Ken Perlin's improved fade curve (2002).
        It has zero first AND second derivatives at t=0 and t=1,
        producing C² continuity — smoother than the original cubic.
        Without this the terrain would have visible grid artifacts.
        """
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a, b, t):
        """Linear interpolation between a and b."""
        return a + t * (b - a)

    def _dot_grad(self, ix, iy, x, y):
        """
        Dot product of gradient at (ix,iy) with offset vector (x-ix, y-iy).
        This is what gives Perlin noise its smooth gradient character.
        """
        gx, gy = self._gradient(ix, iy)
        return gx * (x - ix) + gy * (y - iy)

    def noise(self, x, y):
        """
        Evaluate 2D Perlin noise at point (x, y).
        Returns a value approximately in [-1, 1].

        Algorithm:
        1. Find integer cell containing (x,y)
        2. Compute fractional offsets within the cell
        3. Get gradients at all 4 corners
        4. Dot each gradient with its offset vector
        5. Smoothly interpolate the four dot products
        """
        # Step 1: Integer cell coordinates
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        # Step 2: Fractional offsets
        dx = x - x0
        dy = y - y0

        # Step 3&4: Dot products at four corners
        n00 = self._dot_grad(x0, y0, x, y)
        n10 = self._dot_grad(x1, y0, x, y)
        n01 = self._dot_grad(x0, y1, x, y)
        n11 = self._dot_grad(x1, y1, x, y)

        # Step 5: Smooth interpolation
        u = self._fade(dx)
        v = self._fade(dy)
        # Interpolate along x, then along y
        nx0 = self._lerp(n00, n10, u)
        nx1 = self._lerp(n01, n11, u)
        return self._lerp(nx0, nx1, v)

    def octave_noise(self, x, y, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        Fractal Brownian Motion (fBm) — layered Perlin noise.

        Each octave adds a higher-frequency, lower-amplitude layer.
        This mimics how real terrain has both large mountains (low freq)
        and small rocks/details (high freq).

        Formula:
            height = Σ(i=0 to octaves) amplitude_i * noise(x * freq_i, y * freq_i)

        Parameters:
            octaves     : number of noise layers (more = more detail)
            persistence : amplitude multiplier per octave (0.5 = halved each time)
            lacunarity  : frequency multiplier per octave (2.0 = doubled each time)
        """
        value     = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0   # for normalisation

        for _ in range(octaves):
            value     += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence   # each octave is quieter
            frequency *= lacunarity    # each octave is higher frequency

        # Normalise to [-1, 1]
        return value / max_value


# =============================================================
# CLASS: Terrain
# Procedural terrain built on Vector3, Matrix4, PerlinNoise.
#
# The terrain is a height-map: a 2D grid where each cell stores
# a Vector3 position. Height is determined by octave noise.
# =============================================================

class Terrain:
    """
    Procedural terrain generator.
    Produces a 3D height-mapped surface from noise.
    """

    def __init__(self, width=64, depth=64, scale=5.0,
                 height_scale=3.0, octaves=6,
                 persistence=0.5, lacunarity=2.0,
                 seed=42):
        """
        width, depth  : grid resolution (number of vertices)
        scale         : how much of the noise domain to sample
                        (larger = more zoomed-out terrain)
        height_scale  : vertical exaggeration factor
        octaves       : fBm layers (more = more detail)
        persistence   : amplitude decay per octave
        lacunarity    : frequency growth per octave
        seed          : random seed for reproducibility
        """
        self.width        = width
        self.depth        = depth
        self.scale        = scale
        self.height_scale = height_scale
        self.octaves      = octaves
        self.persistence  = persistence
        self.lacunarity   = lacunarity
        self.seed         = seed

        self.noise_gen = PerlinNoise(seed=seed)
        self.vertices  = []    # list of Vector3 positions
        self.normals   = []    # list of Vector3 surface normals
        self.heights   = []    # raw height values for colour mapping

        self._generate()

    def _generate(self):
        """
        Generate the terrain height map.
        For each grid point (i,j) sample octave noise to get height.
        Store as Vector3(x, height, z) — Y is up axis.
        """
        self.vertices = []
        self.heights  = []

        for j in range(self.depth):
            row_verts   = []
            row_heights = []
            for i in range(self.width):
                # Map grid coordinates to noise domain
                nx = i / self.width  * self.scale
                nz = j / self.depth  * self.scale

                # Sample fractal noise for height
                h = self.noise_gen.octave_noise(
                    nx, nz,
                    octaves     = self.octaves,
                    persistence = self.persistence,
                    lacunarity  = self.lacunarity
                )

                # Scale height and store as 3D position
                x =  (i / (self.width  - 1)) * 2 - 1   # normalise to [-1, 1]
                z =  (j / (self.depth  - 1)) * 2 - 1
                y =  h * self.height_scale

                row_verts.append(Vector3(x, y, z))
                row_heights.append(h)

            self.vertices.append(row_verts)
            self.heights.append(row_heights)

        self._compute_normals()

    def _compute_normals(self):
        """
        Compute surface normals using central differences.

        For each interior vertex, the normal is:
            n = (right - left) × (up - down)
            n̂ = n / |n|

        This is derived from the cross product of two tangent
        vectors along the surface. The cross product produces a
        vector perpendicular to both — the surface normal.
        Normals are essential for lighting computation.
        """
        self.normals = []
        W, D = self.width, self.depth

        for j in range(D):
            row_normals = []
            for i in range(W):
                # Central difference — use neighbours where available
                left  = self.vertices[j][max(i-1, 0)]
                right = self.vertices[j][min(i+1, W-1)]
                down  = self.vertices[max(j-1, 0)][i]
                up    = self.vertices[min(j+1, D-1)][i]

                # Tangent vectors along surface
                tangent_x = right - left
                tangent_z = up    - down

                # Normal = cross product of tangents
                normal = tangent_z.cross(tangent_x).normalise()
                row_normals.append(normal)

            self.normals.append(row_normals)

    def get_height_at(self, i, j):
        """Return height value at grid position (i,j)."""
        return self.vertices[j][i].y

    def get_vertex(self, i, j):
        """Return Vector3 position at grid position (i,j)."""
        return self.vertices[j][i]

    def get_normal(self, i, j):
        """Return surface normal Vector3 at grid position (i,j)."""
        return self.normals[j][i]

    def height_grid(self):
        """Return 2D list of height values (for visualisation)."""
        return [[self.vertices[j][i].y
                 for i in range(self.width)]
                for j in range(self.depth)]

    def stats(self):
        """Return terrain statistics."""
        all_h = [self.vertices[j][i].y
                 for j in range(self.depth)
                 for i in range(self.width)]
        return {
            'min_height' : min(all_h),
            'max_height' : max(all_h),
            'mean_height': sum(all_h) / len(all_h),
            'vertex_count': self.width * self.depth,
            'total_vertices': self.width * self.depth,
        }

    def classify_biome(self, height):
        """
        Simple biome classification by height.
        This is where the procedural system begins to gain
        semantic meaning — the same noise drives both shape and colour.
        """
        h = height / self.height_scale   # normalise back to [-1,1]
        if h < -0.3:  return 'deep_water'
        if h < -0.05: return 'shallow_water'
        if h <  0.02: return 'sand'
        if h <  0.25: return 'grass'
        if h <  0.45: return 'forest'
        if h <  0.65: return 'rock'
        return 'snow'


# =============================================================
# DEMO: Run this file directly to test the system
# =============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BIT 2325 — Milestone 1: Procedural Terrain System")
    print("=" * 55)

    # ── Vector3 demonstration ─────────────────────────────────
    print("\n--- Vector3 Operations ---")
    a = Vector3(1, 2, 3)
    b = Vector3(4, 5, 6)
    print(f"a              = {a}")
    print(f"b              = {b}")
    print(f"a + b          = {a + b}")
    print(f"a - b          = {a - b}")
    print(f"a * 2          = {a * 2}")
    print(f"a · b          = {a.dot(b)}")
    print(f"a × b          = {a.cross(b)}")
    print(f"|a|            = {a.length():.4f}")
    print(f"â (normalised) = {a.normalise()}")
    print(f"|â|            = {a.normalise().length():.6f}  (should be 1.0)")
    print(f"lerp(a,b,0.5)  = {a.lerp(b, 0.5)}")

    # ── Matrix4 demonstration ─────────────────────────────────
    print("\n--- Matrix4 Operations ---")
    T = Matrix4.translation(1, 2, 3)
    print(f"Translation(1,2,3):\n{T}")
    p = Vector3(0, 0, 0)
    print(f"\nApply T to origin {p} → {T * p}")

    S = Matrix4.scale(2, 2, 2)
    v = Vector3(1, 1, 1)
    print(f"\nScale(2,2,2) applied to {v} → {S * v}")

    R = Matrix4.rotation_y(45)
    print(f"\nRotation_Y(45°) applied to (1,0,0) → {R * Vector3(1,0,0)}")

    combined = T * S * R
    print(f"\nCombined T*S*R applied to origin → {combined * Vector3(0,0,0)}")

    # ── Perlin Noise demonstration ────────────────────────────
    print("\n--- Perlin Noise ---")
    noise = PerlinNoise(seed=42)
    print("noise(0.0, 0.0) =", round(noise.noise(0.0, 0.0), 6))
    print("noise(0.5, 0.5) =", round(noise.noise(0.5, 0.5), 6))
    print("noise(1.0, 1.0) =", round(noise.noise(1.0, 1.0), 6))
    print("\nOctave noise (6 octaves) at same points:")
    for x, y in [(0.0,0.0),(0.25,0.1),(0.5,0.5),(0.75,0.9)]:
        v = noise.octave_noise(x, y)
        print(f"  octave_noise({x}, {y}) = {v:.6f}")

    # ── Terrain generation ────────────────────────────────────
    print("\n--- Terrain Generation ---")
    terrain = Terrain(width=64, depth=64, scale=5.0,
                      height_scale=3.0, octaves=6, seed=42)
    stats = terrain.stats()
    print(f"Resolution    : {terrain.width} × {terrain.depth} vertices")
    print(f"Total vertices: {stats['vertex_count']}")
    print(f"Min height    : {stats['min_height']:.4f}")
    print(f"Max height    : {stats['max_height']:.4f}")
    print(f"Mean height   : {stats['mean_height']:.4f}")

    # Sample biome classification
    print("\nSample biome classification:")
    sample_pts = [(0,0),(16,16),(32,32),(48,48),(63,63)]
    for i,j in sample_pts:
        h  = terrain.get_height_at(i, j)
        n  = terrain.get_normal(i, j)
        bm = terrain.classify_biome(h)
        print(f"  ({i:2d},{j:2d}) h={h:6.3f}  normal={n}  biome={bm}")

    print("\nMilestone 1 system verified successfully.")
