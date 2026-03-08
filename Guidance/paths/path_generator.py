import numpy as np
import yaml
from pathlib import Path
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.special import comb


"""Decided on path segment as a superclass in which every other class must ensure 
     they enforce the conditions there in. """    

class UAVConstraints:
    def __init__(self, config_path: str = 'Vehicleparams.yaml'):
        # Handle config path
        if config_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / 'Vehicleparams.yaml'
        else:
            config_path = Path(config_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self._params = yaml.safe_load(f)
        
        # Extract sections
        self.uav = self._params['uav']

    
    @property
    def min_turn_radius(self):
        """Read-only minimum turn radius from config to avoid it being changed in the future """
        return self.uav['min_turn_radius'] 
    
    @property
    def effective_min_turn_radius(self):
        """Computed value (also read-only)"""
        return self._compute_min_turn_radius()
    
    def _compute_min_turn_radius(self):
        """Internal computation method"""
        g = 9.81
        max_g = self.uav.get('max_g_load', 3.5)
        radius_from_g = self.uav['min_V']**2 / (max_g * g)
        radius_from_bank = self.get_min_turn_radius(
            self.uav['min_V'], 
            self.uav['max_bankangle']
        )
        return max(radius_from_bank, radius_from_g)
    
    def get_min_turn_radius(self, V, phi):
        """Your existing method"""
        g = 9.81
        phi_rad = np.deg2rad(phi)
        if np.abs(phi_rad) < 1e-6:
            return float('inf')
        return V**2 / (g * np.tan(phi_rad))

constraints = UAVConstraints()
radius = constraints.min_turn_radius  # Returns value from config
computed = constraints.effective_min_turn_radius  # Returns computed value

class Waypoint():
      def __init__(self, x: float, y: float, yaw: float = None):
        self.x   = float(x)
        self.y   = float(y)
        self.yaw = float(yaw) if yaw is not None else None

      def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

      def yaw_rad(self) -> float:
        """Return yaw in radians, or 0.0 if not set."""
        return np.deg2rad(self.yaw) if self.yaw is not None else 0.0

      def distance_to(self, other: "Waypoint") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

      def __repr__(self):
        return f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw}°)"


class Pathsegment ():
        def __init__(
                        self,
                        step:   float,
                        turn_radius: float,
                        enforce_direction: bool = True
        ) -> None:
                if step <=0:
                        raise ValueError(f"{step!r} is less than 0" )
                if turn_radius < radius:
                        raise ValueError(f"Minimum turn radius must be greater than{radius}")
                
                self.step =step
                self.turn_radius = turn_radius
                self.enforce_direction = enforce_direction
                
        def sample(self, num_points: int = 100) -> np.ndarray:
              """Return (num_points, 2) array of (x, y) positions along segment."""
              raise NotImplementedError
        def length(self) -> float:
               """Total arc length of segment in metres."""
               raise NotImplementedError
        def heading_at_start(self) -> float:  
              """Heading in radians at the start of this segment.""" 
              raise NotImplementedError
        def heading_at_end(self) -> float:
              """Heading in radians at the end of this segment."""
              raise NotImplementedError
        def yaw_array(self, num_points: int = 100) -> np.ndarray:
              
              pts  = self.sample(num_points)
              d    = np.diff(pts, axis=0)
              yaw  = np.arctan2(d[:, 1], d[:, 0])
              return np.append(yaw, yaw[-1])   # repeat last so shape == (N,)
        
class line_segment(Pathsegment): #Path segment defines interface for class to follow
      def __init__(self,start,end):
           self.start=np.array([start.x,start.y])
           self.end=np.array([end.x,end.y])
           """Straight line between two poiints.
           The heading is fixed at arctan2(dy, dx) with waypoints objects ignored"""


      def sample(self, num_points: int = 100) -> np.ndarray:
           x=np.linspace(self.start[0],self.end[0],num_points)
           y=np.linspace(self.start[1],self.end[1],num_points)
           return np.vstack((x,y)).T
      
      def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start)) #Calculates the Euclidian distance between two points
      
      def heading_at_start(self) -> float:
           """Calculates the angle of line arctan2(dy/dx) from start to end in radians """
           return float (np.arctan2(
                self.end[1]-self.start[1],
                self.end[0]-self.start[0]
           ))
      
      def heading_at_end(self) -> float:
           return self.heading_at_start() #Returns same direction angle as start given it's a straight line
      
      def yaw_array(self,num_points: int=100) -> np.ndarray:
           return np.full(num_points,self.heading_at_start())
           
class circular_arc(Pathsegment):
     """For turns,corners and loiter circles.
     Parameters considered:
     1.Center which will be array (x,y)
     2.Radius in metres(float for consistency)
     3.Start-angle in radians measured from positive x-axis(also a float)
     4.Direction CCW will stand for counter clockwise then CW will stand for clockwise
     5.End-angle in radians measured also from the positive x axis"""
     def __init__(
               self,
               center: list,
               radius: float,
               start_angle: float,
               direction: str,
               end_angle: float,
     ):
          self.center = np.array(center,dtype=float)
          self.radius = float(radius)
          self.start_angle = float(start_angle)
          self.end_angle = float(end_angle)
          self.direction = direction.upper()

          if self.direction not in ("CCW" or "CW"):
               raise ValueError(f"Direction must be in 'CW' or 'CCW',{direction!r} fails to meet that")
          
     def _angle_range(self,num_points: int) -> np.ndarray:
          """Compute the correct angle array respecting wrap around and angle direction to handle those 
              angles that were crossing the 0/2π boundary.
              
              Angle = 0: Points to the right (+x axis)
              Angle = π/2 (90°): Points up (+y axis)
              Angle = π (180°): Points left (-x axis)
              
              CCW: angles increase from start → end  (add 2π if end < start)
              CW:  angles decrease from start → end  (subtract 2π if end > start)"""
          s,e = self.start_angle,self.end_angle
          if self.direction == "CCW":
               if e < s:
                    e += 2 * np.pi
          else:#If it's clockwise instead
               if e > s:
                    e-= 2 *np.pi
          return np.linspace(s, e, num_points)
     
     def sample(self, num_points: int = 100) -> np.ndarray:
        angles = self._angle_range(num_points)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        return np.vstack((x, y)).T
     
     def length(self):
          s,e =self.start_angle,self.end_angle
          if self.direction=="CCW":
               sweep=(e-s) % (2* np.pi)
               if sweep < 1e-9 : sweep = 2* np.pi
          else:
               sweep=(s-e) % (2* np.pi)
               if sweep < 1e-9 : sweep=2* np.pi
          return float(self.radius * sweep)
     
     def heading_at_start(self):
          return self._tangent_heading(self.start_angle)
     
     def heading_at_end(self):
          return self._tangent_heading(self.end_angle)
     
     def _tangent_heading(self,angle:float) -> float:
          """
          Tangent direction is at a point on the circle.
          For CCW,tangent at angle θ points at θ + π/2.
          For CW,tangent at angle θ points at θ - π/2.
          This deliberately eliminates the need for an angle argument that would break PathSegment Interface"""
          if self.direction =="CCW":
               return float((angle + np.pi/2) % (2* np.pi))
          else:
               return float((angle-np.pi / 2) % (2 * np.pi))
    
     def yaw_array(self, num_points = 100):
            """Analytic tangent heading at every sampled angle."""
            angles = self._angle_range(num_points)
            if self.direction =="CCW":
                 return (angles + np.pi / 2)% (2 * np.pi)
            else:
                 return (angles - np.pi / 2) % (2* np.pi)
            
class BezierSegment(Pathsegment):
     """
     Cubic Bezier curve between two heading-stamped waypoints.
     Parameters:
        start  (Waypoint): Start — x, y, yaw (degrees) required
        goal   (Waypoint): Goal  — x, y, yaw (degrees) required
        offset (float):    chord / offset = projection distance for P1/P2.
                           Larger = softer curve. Typical: 2–5.
"""
     def __init__(self, start: Waypoint, goal: Waypoint, offset: float = 3.0):
        if start.yaw is None or goal.yaw is None:
            raise ValueError("BezierSegment requires yaw on both start and goal.")
        if offset <= 0:
            raise ValueError(f"offset must be > 0, got {offset}")

        self.start  = start
        self.goal   = goal
        self.offset = offset
        self._ctrl  = np.array(self._compute_control_points(), dtype=float)

     def _compute_control_points(self):
          sx ,sy ,syaw = self.start.x ,self.start.y ,self.start.yaw_rad()
          gx ,gy,gyaw = self.goal.x ,self.goal.y ,self.goal.yaw_rad()
          dist = self.start.distance_to(self.goal) / self.offset
          return[
               (sx,sy),
               (sx +dist * np.cos(syaw) ,sy +dist * np.sin(syaw)),
               (gx-dist * np.cos (gyaw),gy - dist * np.sin(gyaw)),
               (gx,gy),
          ]
     
     def _eval(self, t: float) -> np.ndarray:
        t      = float(np.clip(t, 0, 1))
        n      = len(self._ctrl) - 1
        i_vec  = np.arange(n + 1)
        coeffs = comb(n, i_vec) * (t ** i_vec) * ((1 - t) ** (n - i_vec))
        return coeffs @ self._ctrl

     def _d1(self, t: float) -> np.ndarray:
        t       = float(np.clip(t, 0, 1))
        n       = len(self._ctrl) - 1
        d1_ctrl = n * np.diff(self._ctrl, axis=0)
        m       = len(d1_ctrl) - 1
        i_vec   = np.arange(m + 1)
        coeffs  = comb(m, i_vec) * (t ** i_vec) * ((1 - t) ** (m - i_vec))
        return coeffs @ d1_ctrl

     def sample(self, num_points: int = 100) -> np.ndarray:
        return np.array([self._eval(t) for t in np.linspace(0, 1, num_points)])

     def length(self) -> float:
        pts = self.sample(200)
        d   = np.diff(pts, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

     def heading_at_start(self) -> float:
        d = self._d1(0.0)
        return float(np.arctan2(d[1], d[0]))

     def heading_at_end(self) -> float:
        d = self._d1(1.0)
        return float(np.arctan2(d[1], d[0]))

     def yaw_array(self, num_points: int = 100) -> np.ndarray:
        t_vals = np.linspace(0, 1, num_points)
        return np.array([
            np.arctan2(*self._d1(t)[::-1]) for t in t_vals
        ])
     
     
class SplineSegment(Pathsegment):
    """
    Natural cubic spline through an ordered list of waypoints.

    Fits two independent CubicSplines x(s) and y(s) where s is the
    cumulative chord distance — giving arc-length parametrisation.

    Best for: smooth photogrammetry passes through many waypoints.
    """

    def __init__(self, waypoints: list, bc_type: str = "natural"):
        if len(waypoints) < 2:
            raise ValueError("SplineSegment needs at least 2 waypoints.")
        self.waypoints = waypoints
        self.bc_type   = bc_type
        self._fit()

    def _fit(self):
        pts   = np.array([[w.x, w.y] for w in self.waypoints])
        diffs = np.diff(pts, axis=0)
        ds    = np.hypot(diffs[:, 0], diffs[:, 1])
        self._s = np.concatenate([[0.0], np.cumsum(ds)])
        self._cs_x = CubicSpline(self._s, pts[:, 0], bc_type=self.bc_type)
        self._cs_y = CubicSpline(self._s, pts[:, 1], bc_type=self.bc_type)

    def _eval_s(self, s: float) -> np.ndarray:
        return np.array([float(self._cs_x(s)), float(self._cs_y(s))])

    def sample(self, num_points: int = 100) -> np.ndarray:
        s_vals = np.linspace(0, self._s[-1], num_points)
        return np.array([self._eval_s(s) for s in s_vals])

    def length(self) -> float:
        return float(self._s[-1])

    def heading_at_start(self) -> float:
        dx = float(self._cs_x(self._s[0], 1))
        dy = float(self._cs_y(self._s[0], 1))
        return float(np.arctan2(dy, dx))

    def heading_at_end(self) -> float:
        dx = float(self._cs_x(self._s[-1], 1))
        dy = float(self._cs_y(self._s[-1], 1))
        return float(np.arctan2(dy, dx))

    def yaw_array(self, num_points: int = 100) -> np.ndarray:
        s_vals = np.linspace(0, self._s[-1], num_points)
        dx = self._cs_x(s_vals, 1)
        dy = self._cs_y(s_vals, 1)
        return np.arctan2(dy, dx)
    
class BSplineSegment(Pathsegment):
    """
    B-Spline through an ordered list of control waypoints.

    Unlike SplineSegment, the curve does NOT necessarily pass through
    every waypoint — only the endpoints (when clamped).  This gives
    local shape control: editing one point reshapes only a small region.

    Parameters:
        waypoints (list): Control waypoints. Need at least degree+1.
        degree    (int):  Spline degree — 3 (cubic) recommended.
        smoothing (float):0 = interpolating, >0 = approximating/smoothed.
    """

    def __init__(self, waypoints: list, degree: int = 3, smoothing: float = 0.0):
        if len(waypoints) < degree + 1:
            raise ValueError(
                f"BSplineSegment degree={degree} needs ≥ {degree+1} waypoints, "
                f"got {len(waypoints)}"
            )
        self.waypoints = waypoints
        self.degree    = degree
        self.smoothing = smoothing
        self._fit()

    def _fit(self):
        pts = np.array([[w.x, w.y] for w in self.waypoints])
        self._tck, _ = splprep(
            [pts[:, 0], pts[:, 1]],
            k=self.degree,
            s=self.smoothing,
        )

    def _eval_t(self, t: float) -> np.ndarray:
        x, y = splev(np.clip(t, 0, 1), self._tck, der=0)
        return np.array([float(x), float(y)])

    def _d1_t(self, t: float) -> np.ndarray:
        dx, dy = splev(np.clip(t, 0, 1), self._tck, der=1)
        return np.array([float(dx), float(dy)])

    def sample(self, num_points: int = 100) -> np.ndarray:
        t_vals = np.linspace(0, 1, num_points)
        pts = np.array([self._eval_t(t) for t in t_vals])
        return pts

    def length(self) -> float:
        pts = self.sample(200)
        d   = np.diff(pts, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    def heading_at_start(self) -> float:
        d = self._d1_t(0.0)
        return float(np.arctan2(d[1], d[0]))

    def heading_at_end(self) -> float:
        d = self._d1_t(1.0)
        return float(np.arctan2(d[1], d[0]))

    def yaw_array(self, num_points: int = 100) -> np.ndarray:
        t_vals = np.linspace(0, 1, num_points)
        dx, dy = splev(t_vals, self._tck, der=1)
        return np.arctan2(dy, dx)

"""I deliberately did not input polyspline as part of the run code  given
 for a UAV the paths are chained by arcs and line segments though this maybe appropriate in the case of
  a dense cloud of GPS waypoints or terrain following where the shape is data driven or one wants a fairing curve for visualisation
over a rough set points"""
# class PolySpline(Pathsegment):
#     """
#     Smooth spline path through an ordered list of Waypoint objects.
#     Uses a parametric cubic B-spline (splprep/splev) internally.
    
#     Fits a smooth curve through all waypoints and exposes the same
#     interface as line_segment and circular_arc.
#     """

#     def __init__(
#         self,
#         waypoint_list: list,
#         step:          float = 1.0,
#         turn_radius:   float = None,
#         smoothing:     float = 0.0,   # 0 = interpolates exactly through points
#     ):
#         _radius = turn_radius if turn_radius is not None else radius
#         super().__init__(step=step, turn_radius=_radius)

#         if len(waypoint_list) < 2:
#             raise ValueError("PolySpline requires at least 2 waypoints.")

#         self._waypoints = waypoint_list
#         self.smoothing  = smoothing

#         # Fit once at construction — sample/length/heading are cheap after
#         pts         = np.array([[wp.x, wp.y] for wp in self._waypoints])
#         self._tck, self._u = splprep(
#             [pts[:, 0], pts[:, 1]],
#             s=self.smoothing,
#             k=min(3, len(waypoint_list) - 1)  # degree can't exceed n_points - 1
#         )

#     def sample(self, num_points: int = 100) -> np.ndarray:
#         u_vals   = np.linspace(0, 1, num_points)
#         x, y     = splev(u_vals, self._tck)
#         return np.vstack((x, y)).T

#     def length(self) -> float:
#         pts   = self.sample(1000)
#         diffs = np.diff(pts, axis=0)
#         return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())

#     def heading_at_start(self) -> float:
#         dx, dy = splev(0.0, self._tck, der=1)
#         return float(np.arctan2(dy, dx))

#     def heading_at_end(self) -> float:
#         dx, dy = splev(1.0, self._tck, der=1)
#         return float(np.arctan2(dy, dx))

#     def yaw_array(self, num_points: int = 100) -> np.ndarray:
#         """Analytic tangent heading — overrides base class finite-difference fallback."""
#         u_vals = np.linspace(0, 1, num_points)
#         dx, dy = splev(u_vals, self._tck, der=1)
#         return np.arctan2(dy, dx)

     

