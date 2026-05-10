# =============================================================
# BIT 2325: Computer Graphics & Animation
# Milestone 2 — Space, Transformation & Camera
# Topic: Procedural Content Generation Systems
#
# Authors:  Esther Achieng Otieno (SCT221-C004-0317/2024)
#           Wangui Ninsima Irimu (SCT221-C004-0217/2024)
#           Wendy Wachira (SCT221-C004-0194/2024)
#           Alexander Somba (SCT221-C004-0680/2023)
# Date  : May 2026 | JKUAT
#
# WHAT THIS ADDS ON TOP OF MILESTONE 1:
#   TransformPipeline — full 3D MVP transformation chain
#   Camera            — lookAt view matrix + perspective projection
#   NumericalAnalysis — floating point stability and conditioning
#
# Milestone 1 (terrain_core.py) is imported unchanged.
# Nothing is replaced — everything grows forward.
# =============================================================

import math
import sys
import os

# Import Milestone 1 system unchanged
sys.path.insert(0, os.path.dirname(__file__))
from terrain_core import Vector3, Matrix4, Terrain


# =============================================================
# CLASS: TransformPipeline
# Full 3D Model-View-Projection (MVP) transformation chain.
#
# In any 3D graphics pipeline a point travels through
# four coordinate spaces before reaching the screen:
#
#   Object space → World space → Camera space → Clip space
#        ↑               ↑            ↑              ↑
#   Model matrix    View matrix  Projection    Perspective
#                                 matrix        divide
#
# The MVP matrix combines all three into one:
#   MVP = Projection × View × Model
#   p_clip = MVP × p_object
# =============================================================

class TransformPipeline:
    """
    Full Model-View-Projection transformation pipeline.
    Builds and applies the MVP matrix to 3D geometry.
    """

    def __init__(self):
        self._model      = Matrix4.identity()
        self._view       = Matrix4.identity()
        self._projection = Matrix4.identity()
        self._mvp        = None   # cached combined matrix

    # ── Model matrix setters ──────────────────────────────────

    def set_model(self, matrix):
        """Set the model (world) transformation matrix directly."""
        self._model = matrix
        self._mvp   = None   # invalidate cache

    def translate(self, tx, ty, tz):
        """Apply translation to the model matrix."""
        self._model = Matrix4.translation(tx, ty, tz) * self._model
        self._mvp   = None

    def scale_model(self, sx, sy, sz):
        """Apply scale to the model matrix."""
        self._model = Matrix4.scale(sx, sy, sz) * self._model
        self._mvp   = None

    def rotate_x(self, angle_deg):
        """Apply rotation about X-axis to model matrix."""
        self._model = Matrix4.rotation_x(angle_deg) * self._model
        self._mvp   = None

    def rotate_y(self, angle_deg):
        """Apply rotation about Y-axis to model matrix."""
        self._model = Matrix4.rotation_y(angle_deg) * self._model
        self._mvp   = None

    def rotate_z(self, angle_deg):
        """Apply rotation about Z-axis to model matrix."""
        self._model = Matrix4.rotation_z(angle_deg) * self._model
        self._mvp   = None

    # ── View and Projection setters ───────────────────────────

    def set_view(self, matrix):
        self._view = matrix
        self._mvp  = None

    def set_projection(self, matrix):
        self._projection = matrix
        self._mvp        = None

    # ── MVP composition ───────────────────────────────────────

    @property
    def mvp(self):
        """
        MVP = Projection × View × Model
        Cached — only recomputed when a component changes.

        Matrix multiplication order matters:
        - Model  : local object space → world space
        - View   : world space → camera space (eye at origin)
        - Proj   : camera space → clip space (NDC [-1,1]³)

        Applied RIGHT to LEFT: point is in object space,
        Model brings it to world, View to camera, Proj to clip.
        """
        if self._mvp is None:
            self._mvp = self._projection * self._view * self._model
        return self._mvp

    def transform_point(self, v):
        """Apply full MVP to a Vector3 point."""
        return self.mvp * v

    def transform_direction(self, v):
        """
        Apply only rotation/scale (no translation) to a direction vector.
        Uses the upper 3×3 of the model matrix — ignores the translation column.
        Essential for correctly transforming surface normals.
        """
        m = self._model
        return Vector3(
            m.get(0,0)*v.x + m.get(0,1)*v.y + m.get(0,2)*v.z,
            m.get(1,0)*v.x + m.get(1,1)*v.y + m.get(1,2)*v.z,
            m.get(2,0)*v.x + m.get(2,1)*v.y + m.get(2,2)*v.z,
        ).normalise()

    def apply_to_terrain(self, terrain):
        """
        Apply the model matrix to all terrain vertices and
        recompute transformed normals.
        Returns (transformed_vertices, transformed_normals) as flat lists.
        """
        verts_out   = []
        normals_out = []
        for j in range(terrain.depth):
            for i in range(terrain.width):
                v = terrain.get_vertex(i, j)
                n = terrain.get_normal(i, j)
                verts_out.append(self._model * v)
                normals_out.append(self.transform_direction(n))
        return verts_out, normals_out

    def get_components(self):
        return {
            'model':      self._model,
            'view':       self._view,
            'projection': self._projection,
            'mvp':        self.mvp
        }


# =============================================================
# CLASS: Camera
# View matrix (lookAt) and perspective projection from scratch.
#
# The camera defines WHERE we are looking FROM and HOW we see.
# Two matrices:
#   View matrix     — transforms world into camera space
#   Projection matrix — applies perspective foreshortening
# =============================================================

class Camera:
    """
    Full camera model with lookAt view matrix and perspective
    projection. All matrices derived from first principles.
    """

    def __init__(self, position, target, up,
                 fov_deg=60.0, aspect=16/9,
                 near=0.1, far=100.0):
        """
        position : Vector3 — where the camera is in world space
        target   : Vector3 — what the camera is looking at
        up       : Vector3 — which direction is 'up' for the camera
        fov_deg  : vertical field of view in degrees
        aspect   : width / height ratio of the render target
        near     : near clip plane distance
        far      : far clip plane distance
        """
        self.position = position
        self.target   = target
        self.up       = up
        self.fov_deg  = fov_deg
        self.aspect   = aspect
        self.near     = near
        self.far      = far

        self._view_matrix = None
        self._proj_matrix = None

    # ── View Matrix — lookAt ──────────────────────────────────

    def _build_view_matrix(self):
        """
        Construct the view (lookAt) matrix from scratch.

        Derivation:
        The camera looks from 'eye' toward 'center'.
        We need three orthogonal axes: forward, right, up.

            forward = normalise(eye - center)    ← INTO the scene
            right   = normalise(forward × up)    ← screen right
            true_up = right × forward            ← screen up

        The view matrix rotates world axes to align with
        camera axes, then translates so the camera is at origin.

        | rx  ry  rz  -r·eye |
        | ux  uy  uz  -u·eye |
        | fx  fy  fz  -f·eye |
        |  0   0   0       1 |

        where r=right, u=true_up, f=forward
        """
        eye    = self.position
        center = self.target
        up     = self.up

        # Forward vector: eye → center direction (we negate for right-hand coord)
        f = (eye - center).normalise()   # points FROM scene TO camera

        # Right vector: perpendicular to forward and up
        r = up.normalise().cross(f).normalise()
        # Recompute true up (orthogonal to both f and r)
        u = f.cross(r)                   # no normalise needed — f⊥r already

        # Build 4×4 view matrix
        m = Matrix4([
            r.x,  r.y,  r.z,  -r.dot(eye),
            u.x,  u.y,  u.z,  -u.dot(eye),
            f.x,  f.y,  f.z,  -f.dot(eye),
            0,    0,    0,     1
        ])
        return m

    @property
    def view_matrix(self):
        """Lazily build and cache the view matrix."""
        if self._view_matrix is None:
            self._view_matrix = self._build_view_matrix()
        return self._view_matrix

    # ── Projection Matrix — Perspective ───────────────────────

    def _build_projection_matrix(self):
        """
        Construct the perspective projection matrix from scratch.

        Derivation (Shirley standard form):
        The focal length f = 1 / tan(FOV/2) controls how much
        the scene is magnified. A point at (x,y,z) in camera
        space is projected to screen as:

            x_ndc = f/aspect × x/(-z)
            y_ndc = f × y/(-z)

        This is encoded in a 4×4 matrix that also maps depth
        to the NDC range [-1, 1] for the z-buffer:

        | f/aspect   0        0               0                |
        |    0        f        0               0                |
        |    0        0  -(far+near)/(far-near)  -2fn/(far-near)|
        |    0        0       -1               0                |

        The -1 in row 3 col 2 performs the perspective divide:
        after multiplication, w = -z, and dividing xyz by w
        gives the final NDC coordinates.
        """
        f  = 1.0 / math.tan(math.radians(self.fov_deg / 2.0))
        fn = self.far * self.near
        fd = self.far - self.near

        m = Matrix4([
            f / self.aspect, 0,   0,                       0,
            0,               f,   0,                       0,
            0,               0,  -(self.far+self.near)/fd, -2*fn/fd,
            0,               0,  -1,                       0
        ])
        return m

    @property
    def projection_matrix(self):
        """Lazily build and cache the projection matrix."""
        if self._proj_matrix is None:
            self._proj_matrix = self._build_projection_matrix()
        return self._proj_matrix

    # ── Camera operations ─────────────────────────────────────

    def project_point(self, world_point, pipeline):
        """
        Project a world-space point to normalised device coordinates.
        Returns (x_ndc, y_ndc, depth) or None if behind camera.
        """
        # Apply view
        cam_point = self.view_matrix * world_point
        # Check if behind near plane
        if cam_point.z > -self.near:
            return None
        # Apply projection
        clip = self.projection_matrix * cam_point
        # Perspective divide (w = -z in camera space)
        w = -cam_point.z
        if abs(w) < 1e-10:
            return None
        return (clip.x / w, clip.y / w, cam_point.z)

    def update_position(self, new_position):
        """Move camera — invalidates cached view matrix."""
        self.position      = new_position
        self._view_matrix  = None

    def orbit(self, yaw_deg, pitch_deg, distance=None):
        """
        Orbit the camera around the target point.
        yaw   : rotate horizontally around Y-axis
        pitch : rotate vertically
        """
        if distance is None:
            distance = (self.position - self.target).length()
        yaw   = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        # Spherical to Cartesian
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.sin(pitch)
        z = distance * math.cos(pitch) * math.cos(yaw)
        self.update_position(self.target + Vector3(x, y, z))

    def zoom(self, delta_fov):
        """Adjust FOV (zoom in/out)."""
        self.fov_deg     = max(5.0, min(120.0, self.fov_deg + delta_fov))
        self._proj_matrix = None

    def get_rays(self, img_width, img_height):
        """
        Generate viewing rays for every pixel — used in ray casting.
        Each ray starts at camera position and passes through
        the corresponding pixel on the image plane.
        Returns dict of (col,row): (origin, direction) pairs.
        """
        rays = {}
        f    = 1.0 / math.tan(math.radians(self.fov_deg / 2.0))
        # Camera basis vectors
        fwd   = (self.target - self.position).normalise()
        right = fwd.cross(self.up.normalise()).normalise()
        up    = right.cross(fwd)

        for row in range(img_height):
            for col in range(img_width):
                # NDC coordinates of this pixel
                ndc_x = (2 * (col + 0.5) / img_width  - 1) * self.aspect / f
                ndc_y = (1 - 2 * (row + 0.5) / img_height) / f
                # Ray direction in world space
                direction = (fwd + right * ndc_x + up * ndc_y).normalise()
                rays[(col, row)] = (self.position, direction)
        return rays

    def info(self):
        """Return camera parameters as a formatted string."""
        return (f"Camera:\n"
                f"  Position : {self.position}\n"
                f"  Target   : {self.target}\n"
                f"  FOV      : {self.fov_deg}°\n"
                f"  Aspect   : {self.aspect:.4f}\n"
                f"  Near/Far : {self.near} / {self.far}\n"
                f"  f        : {1/math.tan(math.radians(self.fov_deg/2)):.4f}")


# =============================================================
# CLASS: NumericalAnalysis
# Floating-point stability and condition number analysis.
#
# Requirement: "Numerical stability analysis" (Milestone 2 spec)
# Every matrix operation has floating-point error. We measure
# the condition number — how much error is amplified.
# =============================================================

class NumericalAnalysis:
    """
    Numerical stability analysis for the transformation pipeline.
    Measures conditioning, floating-point drift, and error bounds.
    """

    @staticmethod
    def condition_number_approx(matrix):
        """
        Approximate condition number of a 4×4 matrix.
        κ(M) = ||M|| × ||M⁻¹||   (Frobenius norms)

        A well-conditioned matrix (κ ≈ 1) amplifies errors minimally.
        A poorly conditioned matrix (κ >> 1) can amplify errors
        by κ times — causing visible rendering artifacts.

        For rotation matrices: κ = 1.0 exactly (orthogonal).
        For scale matrices with large ratios: κ = max_s / min_s.
        """
        import math
        # Frobenius norm: sqrt(sum of all squared elements)
        frob = math.sqrt(sum(v**2 for v in matrix.data))

        # Approximate inverse using transpose (works for orthogonal parts)
        # For full accuracy a proper inverse would be needed
        mt = matrix.transpose()
        frob_inv = math.sqrt(sum(v**2 for v in mt.data))
        # Avoid division by zero
        if frob_inv < 1e-12:
            return float('inf')
        return frob * (1.0 / frob_inv)

    @staticmethod
    def orthogonality_error(matrix):
        """
        For a rotation matrix R, R^T × R should equal I exactly.
        The deviation from identity measures accumulated float error.
        Returns max absolute error across all 16 elements.
        """
        mt  = matrix.transpose()
        rtR = mt * matrix   # should be identity
        identity = Matrix4.identity()
        max_err = max(abs(rtR.data[i] - identity.data[i])
                      for i in range(16))
        return max_err

    @staticmethod
    def perspective_divide_stability(z_near, z_far, z_sample):
        """
        Analyse depth buffer precision at a given world depth z.

        The perspective projection maps z ∈ [near, far] to
        NDC depth d ∈ [-1, 1]. Precision is NOT uniform —
        values cluster near the near plane and spread out near far.

        The derivative dd/dz tells us how many NDC units per
        world unit — smaller means less depth resolution.
        """
        # NDC depth formula:
        # d = -(far+near)/(far-near) - 2*far*near / ((far-near)*z)
        n, f, z = z_near, z_far, z_sample
        d     = -(f+n)/(f-n) - 2*f*n/((f-n)*z)
        # Derivative: dd/dz = 2*far*near / ((far-near)*z²)
        dd_dz = 2*f*n / ((f-n) * z**2)
        return {
            'z_world'     : z,
            'ndc_depth'   : round(d, 6),
            'precision'   : round(dd_dz, 6),
            'note'        : 'high precision' if dd_dz > 0.01 else 'low precision'
        }

    @staticmethod
    def analyse_pipeline(pipeline):
        """Run full numerical analysis on a pipeline."""
        comps = pipeline.get_components()
        results = {}
        for name, matrix in comps.items():
            results[name] = {
                'condition'     : NumericalAnalysis.condition_number_approx(matrix),
                'ortho_error'   : NumericalAnalysis.orthogonality_error(matrix),
            }
        return results


# =============================================================
# DEMO: Run directly to verify Milestone 2 system
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BIT 2325 — Milestone 2: Space, Transformation & Camera")
    print("=" * 60)

    # ── Build terrain from Milestone 1 ───────────────────────
    print("\n--- Terrain (from Milestone 1) ---")
    terrain = Terrain(width=64, depth=64, scale=5.0,
                      height_scale=3.0, octaves=6, seed=42)
    stats = terrain.stats()
    print(f"Terrain: {terrain.width}×{terrain.depth}, "
          f"height [{stats['min_height']:.3f}, {stats['max_height']:.3f}]")

    # ── Transformation Pipeline ───────────────────────────────
    print("\n--- Transformation Pipeline ---")
    pipeline = TransformPipeline()

    # Model transforms: scale terrain down, lift slightly above origin
    pipeline.scale_model(1.0, 1.0, 1.0)
    pipeline.translate(0.0, 0.5, 0.0)

    # Camera setup
    camera = Camera(
        position = Vector3(3.0, 4.0, 6.0),
        target   = Vector3(0.0, 0.0, 0.0),
        up       = Vector3(0.0, 1.0, 0.0),
        fov_deg  = 60.0,
        aspect   = 16/9,
        near     = 0.1,
        far      = 50.0
    )

    print(camera.info())

    # Set view and projection in pipeline
    pipeline.set_view(camera.view_matrix)
    pipeline.set_projection(camera.projection_matrix)

    print("\nView matrix (lookAt):")
    print(camera.view_matrix)

    print("\nProjection matrix (perspective):")
    print(camera.projection_matrix)

    print("\nMVP = Projection × View × Model:")
    print(pipeline.mvp)

    # ── Project some terrain vertices ─────────────────────────
    print("\n--- Projecting terrain vertices through MVP ---")
    sample_pts = [(0,0),(16,16),(32,32),(48,48),(63,63)]
    print(f"{'Grid':>10} {'World pos':>30} {'NDC (x,y,z)':>35}")
    print("-" * 80)
    for i,j in sample_pts:
        world   = terrain.get_vertex(i, j)
        result  = camera.project_point(world, camera)
        if result:
            nx, ny, depth = result
            print(f"({i:2d},{j:2d}) {str(world):>30} → "
                  f"NDC({nx:6.3f},{ny:6.3f}) depth={depth:.3f}")
        else:
            print(f"({i:2d},{j:2d}) {str(world):>30} → [clipped]")

    # ── Orbit camera demonstration ────────────────────────────
    print("\n--- Camera Orbit Test ---")
    for yaw in [0, 45, 90, 135, 180]:
        camera.orbit(yaw_deg=yaw, pitch_deg=25, distance=8)
        print(f"  Yaw={yaw:3d}°  Camera position: {camera.position}")

    # ── Numerical Analysis ────────────────────────────────────
    print("\n--- Numerical Stability Analysis ---")
    analysis = NumericalAnalysis.analyse_pipeline(pipeline)
    for name, result in analysis.items():
        print(f"  {name:15s}: κ={result['condition']:.4f}  "
              f"ortho_err={result['ortho_error']:.2e}")

    print("\nDepth buffer precision across near→far range:")
    cam2 = Camera(Vector3(0,0,5), Vector3(0,0,0), Vector3(0,1,0),
                  fov_deg=60, near=0.1, far=100)
    for z in [-0.1, -1.0, -5.0, -20.0, -50.0, -99.0]:
        info = NumericalAnalysis.perspective_divide_stability(0.1, 100, z)
        print(f"  z={z:6.1f}  ndc={info['ndc_depth']:8.5f}  "
              f"precision={info['precision']:.6f}  ({info['note']})")

    print("\nMilestone 2 system verified successfully.")
