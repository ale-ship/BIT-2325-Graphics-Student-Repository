import math, sys, os, time
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from terrain_core import Vector3, Terrain

# =========================================================
# PARTICLE TYPE CONSTANTS (for M6 compatibility)
# =========================================================
PTYPE_LEAF = 0
PTYPE_DUST = 1
PTYPE_EMBER = 2


# =========================================================
# CLASS: KeyframeAnimator (Person A — M5)
# Builds smooth terrain morphing through C² spline interpolation
# =========================================================
class KeyframeAnimator:
    """Smooth morphing between terrain keyframes using cubic spline interpolation."""
    
    def __init__(self, width=128, depth=128, scale=6.0, height_scale=2.8, 
                 octaves=7, persistence=0.5, lacunarity=2.0):
        self.W = width
        self.D = depth
        self.scale = scale
        self.height_scale = height_scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.keyframes = []
        self._built = False
        self.splines = None
        self.t_min = 0
        self.t_max = 0
    
    def add_keyframe(self, t, seed):
        """Add a terrain keyframe at time t using given seed."""
        terrain = Terrain(
            width=self.W, depth=self.D, scale=self.scale,
            height_scale=self.height_scale, octaves=self.octaves,
            persistence=self.persistence, lacunarity=self.lacunarity, seed=seed
        )
        hg = np.array(terrain.height_grid())
        self.keyframes.append((float(t), hg))
        self._built = False
        print(f"    KF t={t} seed={seed} h=[{hg.min():.3f},{hg.max():.3f}]")
    
    def build_splines(self):
        """Build cubic spline interpolation across all keyframes."""
        times = np.array([k[0] for k in self.keyframes])
        grids = np.array([k[1] for k in self.keyframes])
        flat = grids.reshape(len(self.keyframes), self.D * self.W)
        self.splines = CubicSpline(times, flat, axis=0, bc_type='not-a-knot')
        self._built = True
        self.t_min = times[0]
        self.t_max = times[-1]
    
    def get_frame(self, t):
        """Get interpolated height grid at time t."""
        if not self._built:
            self.build_splines()
        t_clipped = np.clip(t, self.t_min, self.t_max)
        return self.splines(t_clipped).reshape(self.D, self.W)
    
    def get_velocity(self, t):
        """Get velocity (first derivative) at time t."""
        if not self._built:
            self.build_splines()
        t_clipped = np.clip(t, self.t_min, self.t_max)
        return self.splines(t_clipped, 1).reshape(self.D, self.W)
    
    def duration(self):
        """Total animation duration."""
        return self.t_max - self.t_min if self._built else 0.0
    
    def motion_analysis(self, n_steps=50):
        """Analyze motion metrics across animation."""
        if not self._built:
            self.build_splines()
        ts = np.linspace(self.t_min, self.t_max, n_steps)
        results = []
        prev = self.get_frame(ts[0])
        
        for t in ts[1:]:
            curr = self.get_frame(t)
            vel = self.get_velocity(t)
            delta = curr - prev
            results.append({
                't': round(float(t), 3),
                'mean_height': round(float(curr.mean()), 4),
                'max_velocity': round(float(np.abs(vel).max()), 4),
                'rms_change': round(float(np.sqrt((delta**2).mean())), 4)
            })
            prev = curr
        
        return results


# =========================================================
# CLASS: ParticleSystem (Person B — M5)
# Multi-type particle emitter with physics and collision
# =========================================================
class ParticleSystem:
    """Particle system with leaves, dust, and embers."""
    
    # Particle type definitions
    TYPES = {
        'leaves': {
            'gravity': -1.5, 'drag': 0.92, 'wind_scale': 1.0,
            'life_range': (4.0, 8.0), 'size_range': (0.015, 0.035),
            'colour': lambda r: np.array([r.uniform(0.55, 0.75), r.uniform(0.35, 0.55), r.uniform(0.05, 0.20)])
        },
        'dust': {
            'gravity': -0.3, 'drag': 0.97, 'wind_scale': 1.5,
            'life_range': (2.0, 5.0), 'size_range': (0.008, 0.020),
            'colour': lambda r: np.array([r.uniform(0.72, 0.88), r.uniform(0.62, 0.78), r.uniform(0.40, 0.55)])
        },
        'embers': {
            'gravity': -0.8, 'drag': 0.94, 'wind_scale': 0.8,
            'life_range': (1.5, 3.5), 'size_range': (0.005, 0.015),
            'colour': lambda r: np.array([r.uniform(0.9, 1.0), r.uniform(0.3, 0.6), 0.0])
        }
    }
    
    def __init__(self, terrain_hgrid=None, world_extent=4.0, max_particles=2000, seed=42):
        """Initialize particle system."""
        self.N = max_particles  # Store max particles as N for scene compatibility
        self.max_n = max_particles
        self.rng = np.random.default_rng(seed)
        self.n = 0  # Current number of live particles
        self.current_type = 'leaves'
        self.world_extent = world_extent
        self.terrain_hgrid = terrain_hgrid
        
        # Particle data (fixed-size arrays)
        self.pos = np.zeros((max_particles, 3), dtype=np.float32)
        self.vel = np.zeros((max_particles, 3), dtype=np.float32)
        self.life = np.zeros(max_particles, dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)
        self.size = np.zeros(max_particles, dtype=np.float32)
        self.colour = np.zeros((max_particles, 3), dtype=np.float32)
        
        # Particle type array (for filtering in scene)
        self.ptype = np.zeros(max_particles, dtype=np.int32)
        self.ptype.fill(PTYPE_LEAF)
        
        # Alive array for scene filtering
        self.alive = np.zeros(max_particles, dtype=bool)
        
        # Wind parameters
        self.wind_dir = np.array([1.0, 0.0, 0.3])
        self.wind_speed = 1.5
        self.time = 0.0
    
    def set_wind(self, direction, speed):
        """Update wind parameters."""
        if len(direction) == 3:
            self.wind_dir = np.array(direction, dtype=np.float32)
        self.wind_speed = float(speed)
    
    def set_type(self, type_name):
        """Set active particle type for emission."""
        if type_name in self.TYPES:
            self.current_type = type_name
    
    def emit(self, n_new, emitter_pos, emitter_radius=0.5, ptype=PTYPE_LEAF):
        """Emit new particles from position."""
        props = self.TYPES[self.current_type]
        
        for _ in range(min(n_new, self.max_n - self.n)):
            i = self.n
            angle = self.rng.uniform(0, 2 * math.pi)
            r = self.rng.uniform(0, emitter_radius)
            
            # Spawn position
            self.pos[i, 0] = emitter_pos[0] + r * math.cos(angle)
            self.pos[i, 1] = emitter_pos[1] + self.rng.uniform(0.05, 0.2)
            self.pos[i, 2] = emitter_pos[2] + r * math.sin(angle)
            
            # Initial velocity
            self.vel[i, 0] = self.rng.uniform(-0.3, 0.3) + self.wind_dir[0] * self.wind_speed * 0.3
            self.vel[i, 1] = self.rng.uniform(0.2, 0.8)
            self.vel[i, 2] = self.rng.uniform(-0.3, 0.3) + self.wind_dir[2] * self.wind_speed * 0.3
            
            # Life and size
            lmin, lmax = props['life_range']
            smin, smax = props['size_range']
            self.life[i] = self.rng.uniform(lmin, lmax)
            self.age[i] = 0.0
            self.size[i] = self.rng.uniform(smin, smax)
            self.colour[i] = props['colour'](self.rng)
            
            # Type and alive state
            self.ptype[i] = ptype
            self.alive[i] = True
            
            self.n += 1
    
    def step(self, dt=0.016):
        """Update all particles for one time step."""
        if self.n == 0:
            return
        
        self.time += dt
        props = self.TYPES[self.current_type]
        
        # Wind gusting
        gust = math.sin(self.time * 0.7) * 0.5 + math.sin(self.time * 1.3) * 0.3
        wind_now = self.wind_dir * (self.wind_speed + gust)
        
        # Physics for active particles
        accel = np.array([0, props['gravity'], 0]) + wind_now * props['wind_scale'] * 0.8
        
        for i in range(self.n):
            if not self.alive[i]:
                continue
            
            # Velocity update
            self.vel[i] = self.vel[i] * props['drag'] + accel * dt
            
            # Position update
            self.pos[i] += self.vel[i] * dt
            
            # Terrain collision (if terrain provided)
            if self.terrain_hgrid is not None:
                D, W = self.terrain_hgrid.shape
                x, y, z = self.pos[i]
                
                # Map world position to grid
                gx = int((x + self.world_extent/2.0) / self.world_extent * (W - 1))
                gz = int((z + self.world_extent/2.0) / self.world_extent * (D - 1))
                
                if 0 <= gx < W and 0 <= gz < D:
                    gh = self.terrain_hgrid[gz, gx]
                    if y < gh + 0.01:
                        # Particle hits terrain
                        self.pos[i, 1] = gh + 0.01
                        self.vel[i, 1] = abs(self.vel[i, 1]) * 0.15  # Bounce
                        self.vel[i, 0] *= 0.6  # Friction
                        self.vel[i, 2] *= 0.6
            
            # Age update
            self.age[i] += dt
            if self.age[i] >= self.life[i]:
                self.alive[i] = False
    
    def get_colours(self):
        """Get colours for active particles (with fade)."""
        if self.n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        active_mask = self.alive[:self.n]
        if not np.any(active_mask):
            return np.zeros((0, 3), dtype=np.float32)
        
        # Fade based on life
        life_frac = self.age[:self.n] / self.life[:self.n]
        fade = (1.0 - life_frac) ** 1.5
        
        colours = np.clip(self.colour[:self.n] * fade[:, None], 0, 1)
        return colours[active_mask].astype(np.float32)
    
    def stats(self):
        """Return particle system statistics."""
        return {
            'active': np.sum(self.alive[:self.n]),
            'wind_speed': round(self.wind_speed, 3),
            'time': round(self.time, 2)
        }


# =========================================================
# CLASS: WaveSimulator (Person C — M5)
# 2D wave equation solver for water surface
# =========================================================
class WaveSimulator:
    """Wave equation solver on regular grid."""
    
    def __init__(self, size=64, wave_speed=1.2, dt=0.016, damping=0.996, seed=77):
        """Initialize wave grid."""
        self.N = size
        self.c = wave_speed
        self.damp = damping
        self.dt = dt
        self.dx = 1.0  # Grid spacing
        
        # CFL stability number
        self.r = (self.c * self.dt / self.dx)
        self.r2 = self.r ** 2
        
        # Wave height arrays
        self.h_cur = np.zeros((size, size), dtype=np.float32)
        self.h_prev = np.zeros((size, size), dtype=np.float32)
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
        self._wind_phase = 0.0
        
        # Randomization
        self.rng = np.random.default_rng(seed)
    
    def add_drop(self, x, z, amplitude=0.08, radius=3):
        """Add impulse disturbance (drop) to wave field."""
        # x, z are in grid coordinates [0, N)
        i0 = int(np.clip(z, 0, self.N - 1))
        j0 = int(np.clip(x, 0, self.N - 1))
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                i = (i0 + di) % self.N
                j = (j0 + dj) % self.N
                d = math.sqrt(di**2 + dj**2)
                self.h_cur[i, j] += amplitude * math.exp(-d**2 / 4.0)
    
    def add_wind_ripple(self, strength=0.004):
        """Add wind-driven ripples to wave surface."""
        self._wind_phase += 0.05
        XI, ZI = np.meshgrid(np.arange(self.N), np.arange(self.N))
        ripple = np.sin((XI * 1.0 + ZI * 0.3) * 0.3 + self._wind_phase) * strength
        self.h_cur += ripple
    
    def get_displacement(self):
        """Get current wave displacement field."""
        return self.h_cur.copy().astype(np.float32)
    
    def get_surface(self, base=-0.12):
        """Get surface height (displacement + base)."""
        return (self.h_cur + base).astype(np.float32)
    
    def step(self, n_steps=1):
        """Advance wave simulation by n time steps."""
        for _ in range(n_steps):
            # Laplacian
            lap = (np.roll(self.h_cur, 1, 0) + np.roll(self.h_cur, -1, 0) +
                   np.roll(self.h_cur, 1, 1) + np.roll(self.h_cur, -1, 1) -
                   4 * self.h_cur)
            
            # Wave equation: h_next = 2h - h_prev + c²∇²h
            h_next = 2 * self.h_cur - self.h_prev + self.r2 * lap
            
            # Damping
            h_next *= self.damp
            
            # Update
            self.h_prev = self.h_cur.copy()
            self.h_cur = h_next
            
            self.time += self.dt
            self.step_count += 1
    
    def get_energy(self):
        """Compute total wave energy (kinetic + potential)."""
        vel = (self.h_cur - self.h_prev) / self.dt
        k = np.sum(vel ** 2) / 2.0
        p = np.sum(self.h_cur ** 2) / 2.0
        return float(k + p)
    
    def stability_check(self):
        """Check CFL stability condition."""
        return {
            'r': round(self.r, 4),
            'stable': self.r <= 1.0 / math.sqrt(2),
            'dt': round(self.dt, 5),
            'dx': self.dx,
            'c': self.c
        }


# =========================================================
# CLASS: TerrainDeformer (Person D — M5)
# Interactive terrain deformation with ripples
# =========================================================
class TerrainDeformer:
    """Deform terrain with ripple effects."""
    
    def __init__(self, width=200, depth=200, world_extent=4.0):
        """Initialize deformation system."""
        self.W = width
        self.D = depth
        self.world_extent = world_extent
        self.base = None
        self.deform = np.zeros((depth, width), dtype=np.float32)
        self.active_ripples = []
        self.time = 0.0
    
    def set_base_terrain(self, base_hgrid):
        """Set the base terrain to deform."""
        self.base = base_hgrid.astype(np.float32).copy()
        self.D, self.W = base_hgrid.shape
        self.deform = np.zeros_like(self.base)
    
    def deform(self, world_x, world_z, amplitude=0.3, sigma=0.3):
        """Apply localized deformation at world coordinates."""
        # Convert world coordinates to grid
        gx = int((world_x + self.world_extent/2.0) / self.world_extent * (self.W - 1))
        gz = int((world_z + self.world_extent/2.0) / self.world_extent * (self.D - 1))
        
        if 0 <= gx < self.W and 0 <= gz < self.D:
            radius = int(sigma * self.W / self.world_extent)
            self.apply_deformation(gx, gz, amplitude, radius)
    
    def apply_deformation(self, gx, gz, strength=0.3, radius=8):
        """Apply radial deformation at grid coordinates."""
        for j in range(max(0, gz - radius * 2), min(self.D, gz + radius * 2)):
            for i in range(max(0, gx - radius * 2), min(self.W, gx + radius * 2)):
                d = math.sqrt((i - gx)**2 + (j - gz)**2)
                if d < radius * 2:
                    influence = math.exp(-d**2 / (2 * (radius * 0.6)**2))
                    self.deform[j, i] += strength * influence
        
        # Add outward-expanding ripple
        self.active_ripples.append({
            'cx': gx, 'cz': gz, 'time': 0.0,
            'speed': 8.0, 'amp': abs(strength) * 0.3,
            'decay': 0.85, 'wave_k': 0.8
        })
    
    def get_deformation(self):
        """Get current deformation grid."""
        return self.deform.copy().astype(np.float32)
    
    def step(self, dt=0.033):
        """Update deformation ripples."""
        self.time += dt
        ripple_grid = np.zeros((self.D, self.W), dtype=np.float32)
        still = []
        
        for rip in self.active_ripples:
            rip['time'] += dt
            t = rip['time']
            cx, cz = rip['cx'], rip['cz']
            radius = rip['speed'] * t
            amp = rip['amp'] * (rip['decay'] ** t)
            
            if amp < 0.001:
                continue
            
            band = int(radius) + 5
            for j in range(max(0, cz - band), min(self.D, cz + band)):
                for i in range(max(0, cx - band), min(self.W, cx + band)):
                    d = math.sqrt((i - cx)**2 + (j - cz)**2)
                    df = d - radius
                    if abs(df) < 8:
                        ripple_grid[j, i] += amp * math.cos(rip['wave_k'] * df) * math.exp(-abs(df) * 0.3)
            
            still.append(rip)
        
        self.active_ripples = still
    
    def get_height_grid(self):
        """Get total deformed height grid (base + deformation)."""
        if self.base is None:
            return self.deform.copy()
        return self.base + self.deform
    
    def reset_deformations(self):
        """Reset deformation to zero."""
        self.deform.fill(0.0)
        self.active_ripples.clear()
    
    def deformation_stats(self):
        """Get deformation statistics."""
        d = self.deform
        return {
            'max_raise': round(float(d.max()), 4),
            'max_lower': round(float(d.min()), 4),
            'rms_deform': round(float(np.sqrt((d**2).mean())), 4),
            'active_ripples': len(self.active_ripples),
            'affected_area': int((np.abs(d) > 0.001).sum())
        }


# =========================================================
# TESTING
# =========================================================
if __name__ == "__main__":
    print("Milestone 5 core — Testing...")
    
    # Test KeyframeAnimator
    t = Terrain(width=16, depth=16, scale=5.0, height_scale=2.5, octaves=4, seed=42)
    hg = np.array(t.height_grid())
    a = KeyframeAnimator(width=16, depth=16, octaves=4)
    a.add_keyframe(0, 42)
    a.add_keyframe(5, 123)
    print(f"Keyframe: {a.get_frame(2.5).shape}")
    print(f"Duration: {a.duration():.2f}s")
    
    # Test ParticleSystem
    ps = ParticleSystem(max_particles=100, seed=42)
    ps.emit(20, [0, 1, 0])
    ps.step(0.016)
    print(f"Particles: {ps.n} active")
    
    # Test WaveSimulator
    ws = WaveSimulator(size=16, wave_speed=1.2, dt=0.016)
    ws.add_drop(8, 8, amplitude=0.1)
    ws.step(5)
    print(f"Waves: {ws.get_displacement().shape}")
    
    # Test TerrainDeformer
    d = TerrainDeformer(width=16, depth=16)
    d.set_base_terrain(hg)
    d.apply_deformation(8, 8, 0.5, 3)
    d.step(0.033)
    print(f"Deform: {d.deformation_stats()}")
    
    print("All tests OK!")
