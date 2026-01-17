import numpy as np
from typing import Dict, List, Tuple,Optional
from pathlib import Path
import yaml


class UAVConstraints:
    """
    Constraint handler for fixed-wing UAV path planning and flight dynamics.
    Loads and validates constraints from the vehicleparams.yaml
    """
    
    def __init__(self, config_path: str = 'vehicleparams.yaml'):
        """
        Initialize constraints from YAML file.
        
        Args:
            config_path: Path to YAML configuration file with my vehicleparams.yaml to be found and set automatically incase the path returns None
        """
        if config_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / 'vehicleparams.yaml'

        else:
            config_path = Path(config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)


        # Extract constraint categories
        self.uav = self.config['uav']
        self.path = self.config['path']
        self.environment = self.config['environment']
        self.mission = self.config['mission']
        self.propulsion = self.config['propulsion']
        
        
        
        # Physical parameters
        self.mass = self.config['mass']
        self.inertia = np.array(self.config['inertia'])
        self.wing_area = self.config['wing_area']
        self.wing_span = self.config['wing_span']


    def _compute_derived_constraints(self):
        """Pre-compute derived constraint values for efficiency."""
        # Convert angles to radians for internal use
        self.max_bank_angle_rad = np.deg2rad(self.uav['max_bankangle'])
        self.max_turn_rate_rad = np.deg2rad(self.uav['max_turn_rate'])
        
        # Calculate minimum turn radius from G-load
        g = 9.81
        max_centripetal_accel = self.uav.get('max_g_load', 3.5) * g
        self.min_turn_radius_from_g = self.uav['min_V']**2 / max_centripetal_accel
        
        # Effective minimum turn radius (max of aerodynamic and G-load constraints)
        self.effective_min_turn_radius = max(
            self.get_min_turn_radius(self.uav['min_V'], self.uav['max_bankangle']),
            self.min_turn_radius_from_g
        )
        
        # Pre-computed bounds array for faster checking
        bounds = self.environment['bounds']
        self.bounds_array = np.array([
            bounds['x_min'], bounds['x_max'],
            bounds['y_min'], bounds['y_max'],
            bounds['z_min'], bounds['z_max']
        ])
        
    def check_velocity(self, V: float) -> bool:
        """Check if velocity is within operational limits."""
        return self.uav['min_V'] <= V <= self.uav['max_V']
    
    def clamp_velocity(self, V: float) -> float:
        """Clamp velocity to valid range."""
        return np.clip(V, self.uav['min_V'], self.uav['max_V'])
    
    def check_acceleration(self, a: float) -> bool:
        """Check if acceleration is within limits."""
        return self.uav['min_a'] <= a <= self.uav['max_a']
    
    def clamp_acceleration(self, a: float) -> float:
        """Clamp acceleration to valid range."""
        return np.clip(a, self.uav['min_a'], self.uav['max_a'])
    
    def check_bank_angle(self, phi: float) -> bool:
        return abs(phi) <= self.uav['max_bankangle']
    
    def get_required_bank_angle(self, velocity: float, turn_radius: float) -> float:
        #Calculate the required bank angle for a given turn radius and velocity
        """Args:
            velocity: Current velocity (m/s)
            turn_radius: Desired turn radius (m)
        
        Returns:
            Required bank angle in degrees"""
        if turn_radius < 1e-6:
            return self.uav['max_bankangle']
        
        g = 9.81
        phi_rad = np.arctan(velocity**2 / (g * turn_radius))
        return np.rad2deg(phi_rad)
    
    def check_climb_rate(self, climb_rate: float) -> bool:
        """Check if climb rate is within limits (m/s)."""
        return self.uav['max_descentrate'] <= climb_rate <= self.uav['max_climbrate']
    
    def check_turn_rate(self, turn_rate: float) -> bool:
        """Check if turn rate is within limits (deg/s)."""
        return abs(turn_rate) <= self.uav['max_turn_rate']
    
    def get_min_turn_radius(self, V: float, phi: float = None) -> float:
        """
        Calculate minimum turn radius based on velocity and bank angle.
        
        Args:
            V: Velocity (m/s)
            phi: Bank angle (degrees). If None, uses max bank angle.
        
        Returns:
            Minimum turn radius (m)
        """
        if phi is None:
            phi = self.uav['max_bankangle']
        
        g = 9.81  # m/s^2
        phi_rad = np.deg2rad(phi)
        
        # R = V^2 / (g * tan(phi))
        if np.abs(phi_rad) < 1e-6:
            return float('inf')
        
        R = V**2 / (g * np.tan(phi_rad))
        return max(R, self.uav['max_turn_radius'])
    
    def check_path_curvature(self, path_points: np.ndarray) -> Tuple[bool, float]:
        """
        Check curvature along entire path segment.
        
        Args:
            path_points: Nx3 array of path points
            
        Returns:
            (is_valid, max_curvature_found)
        """
        if len(path_points) < 3:
            return True, 0.0
        
        max_curvature = 0.0
        
        for i in range(1, len(path_points) - 1):
            p1 = path_points[i-1]
            p2 = path_points[i]
            p3 = path_points[i+1]
            
            # Calculate curvature using three-point circle method
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a < 1e-6 or b < 1e-6:
                continue
            
            # Semi-perimeter
            s = (a + b + c) / 2
            
            # Triangle area (Heron's formula)
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                continue
            
            area = np.sqrt(area_sq)
            
            # Radius of circumcircle
            if area < 1e-6:
                continue
            
            R = (a * b * c) / (4 * area)
            
            if R < 1e-6:
                curvature = float('inf')
            else:
                curvature = 1.0 / R
            
            max_curvature = max(max_curvature, curvature)
        
        return max_curvature <= self.path['max_curvature'], max_curvature
    
    def check_path_climb_rates(self, path_points: np.ndarray, 
                              dt: float = 0.1) -> Tuple[bool, float, float]:
        """
        Check climb/descent rates along path.
        
        Args:
            path_points: Nx3 array of path points
            dt: Time step between points (seconds)
            
        Returns:
            (is_valid, max_climb_rate, min_climb_rate)
        """
        if len(path_points) < 2:
            return True, 0.0, 0.0
        
        # Calculate climb rates
        z_coords = path_points[:, 2]
        dz = np.diff(z_coords)
        
        # Assume average velocity for time estimation
        avg_velocity = (self.uav['min_V'] + self.uav['max_V']) / 2
        horizontal_dist = np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1)
        times = horizontal_dist / avg_velocity
        
        # Avoid division by zero
        times = np.maximum(times, 1e-6) #protects the simulation from exploding.
        climb_rates = dz / times
        
        max_climb = np.max(climb_rates) if len(climb_rates) > 0 else 0.0
        min_climb = np.min(climb_rates) if len(climb_rates) > 0 else 0.0
        
        valid = (min_climb >= self.uav['max_descentrate'] and 
                max_climb <= self.uav['max_climbrate'])
        
        return valid, max_climb, min_climb
    
    def check_service_ceiling(self, z_amsl: float) -> bool:
        """Check if altitude is below service ceiling."""
        return z_amsl <= self.uav['service_ceiling']
    
    def check_altitude(self, z_agl: float) -> bool:
        """Check if altitude AGL is within operational limits."""
        return (self.uav['minimum_operational_altitude'] <= z_agl <= 
                self.uav['operational_ceiling'])
    
    def get_min_safe_altitude(self, terrain_height: float = 0.0) -> float:
        """
        Get minimum safe altitude considering terrain and clearances.
        
        Args:
            terrain_height: Current terrain elevation (m AMSL)
        
        Returns:
            Minimum safe altitude (m AMSL)
        """
        return terrain_height + max(
            self.uav['ground_clearance'],
            self.uav['terrain_clearance'],
            self.uav['minimum_operational_altitude']
        )
    
    def check_waypoint_deviation(self, position: np.ndarray, 
                                 waypoint: np.ndarray) -> bool:
        """
        Check if position is within acceptable deviation from waypoint.
        
        Args:
            position: Current position [x, y, z]
            waypoint: Target waypoint [x, y, z]
        """
        lateral_deviation = np.linalg.norm(position[:2] - waypoint[:2])
        return lateral_deviation <= self.mission['max_deviation']
    
    def get_control_limits(self) -> Dict[str, Tuple[float, float]]:
        surfaces = self.uav['control_surfaces']
        limits = {}
        for surface_name, surface_config in surfaces.items():
          limits[surface_name] = {
            'min': surface_config.get('min_deflection', -25.0),
            'max': surface_config.get('max_deflection', 25.0),
            'max_rate': surface_config.get('max_rate', 60.0)
        }
        return limits
    
    
    def validate_path(self, path_points: np.ndarray, 
                     velocities: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive path validation against all constraints.
        
        Args:
            path_points: Nx3 array of path points
            velocities: Optional array of velocities at each point
            
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'violations': [],
            'metrics': {}
        }
        
        # 1. Check path curvature
        curvature_valid, max_curvature = self.check_path_curvature(path_points)
        results['metrics']['max_curvature'] = max_curvature
        if not curvature_valid:
            results['valid'] = False
            results['violations'].append(f"Curvature {max_curvature:.3f} > max {self.path['max_curvature']}")
        
        # 2. Check airspace bounds for all points
        for i, point in enumerate(path_points):
            if not self.check_bounds(point):
                results['valid'] = False
                results['violations'].append(f"Point {i} out of bounds")
                break
        
        # 3. Check no-fly zones
        for i, point in enumerate(path_points):
            if not self.check_no_fly_zones(point):
                results['valid'] = False
                results['violations'].append(f"Point {i} in no-fly zone")
                break
        
        # 4. Check climb rates
        if path_points.shape[1] >= 3:
            climb_valid, max_climb, min_climb = self.check_path_climb_rates(path_points)
            results['metrics']['max_climb_rate'] = max_climb
            results['metrics']['min_climb_rate'] = min_climb
            if not climb_valid:
                results['valid'] = False
                results['violations'].append(f"Climb rate violation: [{min_climb:.1f}, {max_climb:.1f}] m/s")
        
        # 5. Check path length
        if len(path_points) >= 2:
            distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
            total_length = np.sum(distances)
            results['metrics']['total_length'] = total_length
            
            if 'max_path_length' in self.uav and total_length > self.uav['max_path_length']:
                results['valid'] = False
                results['violations'].append(f"Path length {total_length:.1f}m > max {self.uav['max_path_length']}m")
        
        # 6. Turn feasibility check
        if velocities is not None and len(velocities) >= 2:
            for i in range(1, len(path_points) - 1):

                # Estimate turn radius
                p1 = path_points[i-1]
                p2 = path_points[i]
                p3 = path_points[i+1]
                
                # Calculate turn radius from these three points
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                
                if a < 1e-6 or b < 1e-6 or c < 1e-6:
                    continue #Points are too close

                # Heron's formula for triangle area
                s = (a + b + c) / 2
                area_sq = s * (s - a) * (s - b) * (s - c)

                if area_sq <=0:
                    continue #Points are collinear
            
                area = np.sqrt(area_sq)
                turn_radius = (a * b * c) / (4 * area)
            
                if turn_radius < 1e-6:
                 continue  # Infinite curvature

                # Simple curvature-based bank angle check
                velocity = velocities[i] if i < len(velocities) else velocities[-1]
                min_R_possible = self.get_min_turn_radius(velocity)
                if turn_radius < min_R_possible:
                 results['valid'] = False
                 results['violations'].append(
                    f"Turn radius too small at point {i}: required {turn_radius:.1f} < possible {min_R_possible:.1f}"
                 )
                
        
        # 7. Check altitude constraints
        if path_points.shape[1] >= 3:
            for i, point in enumerate(path_points):
                z_agl = point[2]  # Assuming Z is altitude AGL
                if not self.check_altitude(z_agl):
                    results['valid'] = False
                    results['violations'].append(f"Altitude {z_agl:.1f}m out of range at point {i}")
                    break
        
        return results
    
    def smooth_path_for_constraints(self, path_points: np.ndarray, 
                                   iterations: int = 5) -> np.ndarray:
        """
        Simple path smoothing to help meet constraints.
        
        Args:
            path_points: Nx3 array of path points
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed path points
        """
        smoothed = path_points.copy()
        alpha = 0.3  # Smoothing factor
        beta = 0.4   # Attraction to original
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Original point attraction
                original_attraction = path_points[i] - smoothed[i]             
                # Smoothing term (move toward average of neighbors)
                smoothing = (smoothed[i-1] + smoothed[i+1]) / 2 - smoothed[i]
                
                # Apply update
                smoothed[i] += alpha * smoothing + beta * original_attraction
        
        return smoothed
    
    def enforce_min_turn_radius(self, path_points: np.ndarray) -> np.ndarray:
        """
        Adjust path to respect minimum turn radius.
        
        Args:
            path_points: Nx3 array of path points
            
        Returns:
            Adjusted path points
        """
        adjusted = path_points.copy()
        min_radius = self.effective_min_turn_radius
        
        for i in range(1, len(adjusted) - 1):
            p1 = adjusted[i-1]
            p2 = adjusted[i]
            p3 = adjusted[i+1]
            
            # Calculate approximate turn radius
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            # Heron's formula for triangle area
            s = (a + b + c) / 2
            area_sq = s * (s - a) * (s - b) * (s - c)
            
            if area_sq <= 0:
                continue
            
            area = np.sqrt(area_sq)
            
            if area < 1e-6 or a * b * c < 1e-6:
                continue
            
            radius = a * b * c / (4 * area)
            
            if radius < min_radius:
                # Move the middle point outward to increase radius
                center_dir = (p1 + p3) / 2 - p2
                scale = (min_radius - radius) / min_radius
                adjusted[i] += center_dir * scale * 0.5
        
        return adjusted
    
    def validate_state(self, state: Dict) -> Tuple[bool, List[str]]:
        """
        Validate complete UAV state against all constraints.
        
        Args:
            state: Dict containing UAV state variables
                   (position, velocity, attitude, etc.)
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Position constraints
        if 'position' in state:
            if not self.check_bounds(state['position']):
                violations.append("Position out of bounds")
            if not self.check_no_fly_zones(state['position']):
                violations.append("Position in no-fly zone")
        
        # Velocity constraints
        if 'velocity' in state:
            V = np.linalg.norm(state['velocity'])
            if not self.check_velocity(V):
                violations.append(f"Velocity {V:.1f} m/s out of range")
        
        # Attitude constraints
        if 'bank_angle' in state:
            if not self.check_bank_angle(state['bank_angle']):
                violations.append(f"Bank angle {state['bank_angle']:.1f}° exceeds limit")
        
        # Altitude constraints
        if 'altitude_agl' in state:
            if not self.check_altitude(state['altitude_agl']):
                violations.append(f"Altitude {state['altitude_agl']:.1f} m out of operational range")
        
        return (len(violations) == 0, violations)
    
    def summary(self) -> str:
        """Generate human-readable constraint summary."""
        bounds = self.environment['bounds']
        return f"""
UAV Constraint Summary:
======================
Velocity: {self.uav['min_V']:.1f} - {self.uav['max_V']:.1f} m/s
Altitude: {self.uav['minimum_operational_altitude']} - {self.uav['operational_ceiling']} m AGL
Bank Angle: ±{self.uav['max_bankangle']:.1f}°
Turn Rate: ±{self.uav['max_turn_rate']:.1f}°/s
Max Turn Radius: {self.uav['max_turn_radius']:.1f} m
Effective Min Turn Radius: {self.effective_min_turn_radius:.1f} m
Climb Rate: {self.uav['max_descentrate']:.1f} to {self.uav['max_climbrate']:.1f} m/s
Max Climb Angle: {self.path['max_climbangle']:.1f}°
Path Curvature Limit: {self.path['max_curvature']:.3f} 1/m
Endurance: {self.uav['max_endurance']/60:.0f} min (Mission: {self.uav['mission_endurance']/60:.0f} min)
Environment Bounds: 
  X: [{bounds['x_min']}, {bounds['x_max']}] 
  Y: [{bounds['y_min']}, {bounds['y_max']}]
  Z: [{bounds['z_min']}, {bounds['z_max']}]
No-Fly Zones: {len(self.environment['no_fly_zones'])}
Mass: {self.mass:.1f} kg, Wingspan: {self.wing_span:.1f} m
"""
