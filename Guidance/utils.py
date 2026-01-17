import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

EPS = 1e-9  # Global numerical stability constant


@dataclass
class LineSegment:
    """Straight line segment (supports 2D or 3D)."""
    start: np.ndarray
    end: np.ndarray


@dataclass  
class Circle:
    """2D turn circle (used for Dubins paths)."""
    center: np.ndarray     # (x, y)
    radius: float


class GeometryCore:
    """
    Pure geometry and math utilities.
    Timeless, reusable, no mission logic.
    Works fully in 3D unless explicitly marked 2D-only.
    """

    @staticmethod
    def safe_norm(v: np.ndarray) -> float:
        """Numerically stable norm that avoids divide-by-zero."""
        n = np.linalg.norm(v)
        return n if n > EPS else 0.0

    @staticmethod
    def safe_normalize(v: np.ndarray) -> np.ndarray:
        """Return unit vector, or zero vector if magnitude is tiny."""
        n = GeometryCore.safe_norm(v)
        return v / n if n > EPS else np.zeros_like(v)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Deprecated alias for backward compat."""
        return GeometryCore.safe_normalize(v)


    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle between vectors (3D-capable)."""
        v1_u = GeometryCore.safe_normalize(v1)
        v2_u = GeometryCore.safe_normalize(v2)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return GeometryCore.safe_norm(p2 - p1)

    @staticmethod
    def curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Curvature from 3 points in 3D.
        Uses 3D circumcircle formulation.
        """
        a = GeometryCore.distance(p2, p1)
        b = GeometryCore.distance(p3, p2)
        c = GeometryCore.distance(p3, p1)

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= EPS:
            return 0.0

        area = np.sqrt(area_sq)
        radius = (a * b * c) / (4 * area)

        if radius < EPS:
            return 0.0

        return 1.0 / radius


    @staticmethod
    def interpolate(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        return (1 - t) * p1 + t * p2

    @staticmethod
    def smooth_path(points: np.ndarray, iterations: int = 3) -> np.ndarray:
        smoothed = points.copy()
        alpha = 0.3

        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                neighbor_avg = (smoothed[i-1] + smoothed[i+1]) / 2
                smoothed[i] += alpha * (neighbor_avg - smoothed[i])

        return smoothed

    @staticmethod
    def arc_length_parametrize(points: np.ndarray, 
                               num_samples: int = 100) -> np.ndarray:
        if len(points) < 2:
            return points.copy()

        d = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.zeros(len(points))
        cumulative[1:] = np.cumsum(d)
        L = cumulative[-1]

        if L < EPS:
            return points.copy()

        samples = np.linspace(0, L, num_samples)
        result = np.zeros((num_samples, points.shape[1]))

        j = 0
        for i, s in enumerate(samples):
            while j < len(cumulative) - 1 and cumulative[j+1] < s:
                j += 1

            if j == len(cumulative) - 1:
                result[i] = points[-1]
            else:
                t = (s - cumulative[j]) / (cumulative[j+1] - cumulative[j])
                result[i] = GeometryCore.interpolate(points[j], points[j+1], t)

        return result



    @staticmethod
    def circle_tangents(circle: Circle, point: np.ndarray):
        """
        Returns tangent points from external point to circle.
        2D ONLY. Ignores Z.
        """
        cp = point[:2] - circle.center[:2]
        dist_cp = GeometryCore.safe_norm(cp)

        if dist_cp < circle.radius:
            return None, None

        if abs(dist_cp - circle.radius) < EPS:
            tdir = np.array([-cp[1], cp[0]])
            tdir = GeometryCore.safe_normalize(tdir)
            return point.copy(), None

        angle = np.arcsin(circle.radius / dist_cp)
        cp_norm = cp / dist_cp

        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ])
        rot2 = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)],
        ])

        dir1 = rot @ cp_norm
        dir2 = rot2 @ cp_norm

        t1 = circle.center[:2] + circle.radius * dir1
        t2 = circle.center[:2] + circle.radius * dir2

        return t1, t2

    @staticmethod
    def circle_intersection(circle: Circle, line: LineSegment):
        """
        Circle-line intersection.
        2D ONLY. Uses XY plane.
        """
        p1 = line.start[:2] - circle.center[:2]
        p2 = line.end[:2] - circle.center[:2]

        d = p2 - p1
        d_len = GeometryCore.safe_norm(d)

        if d_len < EPS:
            return []

        d_norm = d / d_len
        t = -np.dot(p1, d_norm)
        closest = p1 + t * d_norm
        dist = GeometryCore.safe_norm(closest)

        if dist > circle.radius + EPS:
            return []

        points = []
        if abs(dist - circle.radius) < EPS:
            points.append(closest)
        else:
            dt = np.sqrt(circle.radius**2 - dist**2)
            points.append(closest - dt * d_norm)
            points.append(closest + dt * d_norm)

        results = []
        for pt in points:
            t_param = np.dot(pt - p1, d_norm) / d_len
            if 0 <= t_param <= 1:
                results.append(pt + circle.center[:2])

        return results



    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """2D ONLY polygon test (no-fly zones)."""
        if len(polygon) < 3:
            return False

        x, y = point[:2]
        inside = False

        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            intersects = ((yi > y) != (yj > y)) and \
                         (x < (xj - xi) * (y - yi) / (yj - yi + EPS) + xi)

            if intersects:
                inside = not inside

        return inside

    @staticmethod
    def distance_to_segment(point: np.ndarray, segment: LineSegment) -> float:
        seg = segment.end - segment.start
        L = GeometryCore.safe_norm(seg)

        if L < EPS:
            return GeometryCore.distance(point, segment.start)

        d = seg / L
        w = point - segment.start
        proj = np.dot(w, d)

        if proj <= 0:
            return GeometryCore.distance(point, segment.start)
        elif proj >= L:
            return GeometryCore.distance(point, segment.end)

        closest = segment.start + proj * d
        return GeometryCore.distance(point, closest)

    @staticmethod
    def bounding_box(points: np.ndarray):
        if len(points) == 0:
            return None, None
        return np.min(points, axis=0), np.max(points, axis=0)


    @staticmethod
    def turn_radius_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        a = GeometryCore.distance(p1, p2)
        b = GeometryCore.distance(p2, p3)
        c = GeometryCore.distance(p1, p3)

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= EPS:
            return float("inf")

        area = np.sqrt(area_sq)
        radius = (a * b * c) / (4 * area)

        return radius if radius > EPS else float("inf")

    @staticmethod
    def required_bank_angle(velocity: float, turn_radius: float) -> float:
        if turn_radius < EPS:
            return 90.0
        g = 9.81
        phi = np.arctan(velocity**2 / (g * turn_radius))
        return np.rad2deg(phi)

    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2*np.pi) - np.pi

    @staticmethod
    def heading_from_vector(v: np.ndarray) -> float:
        """Yaw angle (rad) from 3D vector."""
        return np.arctan2(v[1], v[0])

    @staticmethod
    def pitch_from_vector(v: np.ndarray) -> float:
        """Pitch angle (rad) from 3D vector."""
        xy = np.sqrt(v[0]**2 + v[1]**2)
        return np.arctan2(v[2], xy)

    @staticmethod
    def vector_from_heading_pitch(yaw: float, pitch: float) -> np.ndarray:
        """3D unit vector from yaw/pitch."""
        return np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])

    @staticmethod
    def bearing(p1: np.ndarray, p2: np.ndarray) -> float:
        """Yaw from p1 → p2 in 3D."""
        return GeometryCore.heading_from_vector(p2 - p1)
