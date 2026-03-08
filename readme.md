# AUTONOMOUS UAV NAVIGATION

### PATHS GENERATION
This module implements the geometry layer of a UAV navigation stack.

It focuses only on constructing mathematically valid paths, leaving decision making and planning to higher-level modules such as:
1.path planners
2.Dubins planners
3.guidance algorithms
4.trajectory optimizers

**The core design principles include;**
1.Geometry-first design.Paths are constructed as geometric segments that can be chained together.
     i.Each segment implements the same interface:
    ii.Sampling along the curve
   iii.Computing path length
    iv.Heading along the path

2.Vehicle-aware constraints which are found in vehicleparams.yaml file.

3.Extensible architecture.All path primitives inherit from a common base class: Pathsegment with new paths able to be added without modifying the rest of the system.

**Path segment interface;**
Required methods:
Method	                             Purpose
sample()	                      Returns sampled (x,y) coordinates along the segment
length()	                      Total path length
heading_at_start()	              Starting heading
heading_at_end()	              Ending heading
yaw_array()	                      Heading profile along the path
This common interface allows planners to treat all path segments uniformly.

**Implemented Path Types**
1. Line Segment.
   Straight path between two waypoints with a constant heading.
2. Circular Arc.
   Represents constant-radius turns.
   Used for:
           UAV turning maneuvers
           Loiter circles
           Waypoint filleting
3. Bezier Segment.
   Cubic Bézier curve connecting two waypoints with heading constraints.
   Control points are automatically generated from waypoint headings.
   Used for:
            Smooth transitions
            Camera paths
            Gentle maneuver shaping
4. Cubic Spline Segment.
   Interpolating cubic spline through a sequence of waypoints.
   Uses arc-length parameterization for stable sampling.
   Best suited for:
                    Photogrammetry missions
                    Smooth waypoint traversal
                    Terrain following paths
5. B-Spline Segment.
    Approximate spline curve using control points and unlike cubic splines the curve does not pass through every waypoint as well as it provides local shape control.
    Advantages:
               Smoother global paths
               Robust editing
               Noise tolerant

6.Optional polynomial spline segment.
It is appropriate in the case of a dense cloud of GPS waypoints or terrain following where the shape is data driven or one wants a fairing curve for visualisation over a rough set points.

### LIMITS(UAV Constraint Management System)
A Python module for handling physical, environmental, and mission constraints for a fixed-wing UAV.
This system loads UAV limits from a configuration file and provides utilities to:
- validate UAV states
- verify path feasibility
- enforce flight constraints
- smooth trajectories to respect vehicle limits
It acts as the constraint authority for the UAV autonomy stack.

**The purpose of this part is to ensure that the autonomous UAV systems operate within strict aerodynamic, structural, and mission limits**.

**The constraints are loaded from a configuration file: vehicleparams.yaml to ensure that vehicle parameters are decoupled from navigation algorithms.This design allows the same codebase to be used for different UAV platforms simply by changing the configuration.**

Key Features.
1. Vehicle constraint validation
This ensures the UAV operates within safe limits:
- velocity
- bank angle
- climb rate
- turn rate
- altitude limits

2. Path feasibility checking
Validates entire paths against:
- curvature limits
- turn radius constraints
- climb/descent limits
- airspace boundaries
- no-fly zones

3. Automatic constraint enforcement
The system can automatically modify paths to satisfy constraints:
- smoothing sharp turns
- enforcing minimum turn radius
- adjusting trajectory curvature

4. State validation
Checks real-time UAV state variables against safety constraints.

### UTILITIES (GEOMETRY CORE)
This module provides core geometric utilities used throughout the UAV navigation system. It contains reusable mathematical tools for working with vectors, paths, and geometric relationships required for path generation and planning.

The functions in this module are independent of UAV dynamics and mission logic, allowing them to serve as a foundational geometry layer for the rest of the project.

**Capabilities**
The utilities support several key geometric operations:

* Vector operations
normalization
dot and cross products
distance calculations

* Path geometry
curvature estimation
arc-length parametrization
path smoothing

* Navigation calculations
heading and pitch extraction from vectors
vector construction from heading/pitch

* 2D geometric analysis
circle intersections
circle tangents
point-in-polygon tests
distance to line segments

* Turn and flight constraints
turn radius estimation
required bank angle for turns

**Design Principles**
Pure geometry layer (no UAV-specific logic)
Reusable across planning modules
Numerically stable computations




