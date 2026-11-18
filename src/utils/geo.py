import math
import numpy as np

# Earth radius in meters
EARTH_RADIUS = 6371000.0

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computes great-circle distance between two points in meters.
    Supports scalars or numpy arrays.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS * c

def to_local_meters(lon, lat, origin_lon, origin_lat):
    """
    Converts spherical coordinates (lon, lat) to local Euclidean (x, y) meters
    using Equirectangular projection centered at (origin_lon, origin_lat).
    """
    phi0 = math.radians(origin_lat)
    x = EARTH_RADIUS * np.radians(lon - origin_lon) * math.cos(phi0)
    y = EARTH_RADIUS * np.radians(lat - origin_lat)
    return x, y

def to_lon_lat(x, y, origin_lon, origin_lat):
    """
    Inverse projection: local Euclidean (x, y) in meters back to (lon, lat).
    """
    phi0 = math.radians(origin_lat)
    lon = origin_lon + np.degrees(x / (EARTH_RADIUS * math.cos(phi0)))
    lat = origin_lat + np.degrees(y / EARTH_RADIUS)
    return lon, lat

def compute_bearing(lon1, lat1, lon2, lat2):
    """
    Calculates initial bearing from point 1 to point 2 in degrees [0, 360).
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return bearing

def angular_difference(bearing1, bearing2):
    """
    Calculates the minimum absolute angle between two bearings in degrees [0, 180].
    """
    diff = np.abs(bearing1 - bearing2) % 360.0
    return np.minimum(diff, 360.0 - diff)

def project_point_to_segment(px, py, u_x, u_y, v_x, v_y):
    """
    Projects point p (px, py) onto directed line segment u->v.
    Returns:
        perp_dist: Distance from p to closest point on segment (in meters)
        t: Normalized scalar position along segment [0.0, 1.0]
        proj_x, proj_y: Coordinates of the projected point
    """
    dx = v_x - u_x
    dy = v_y - u_y
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq <= 1e-12:
        dist = math.sqrt((px - u_x)**2 + (py - u_y)**2)
        return dist, 0.0, u_x, u_y

    ap_x = px - u_x
    ap_y = py - u_y
    t = (ap_x * dx + ap_y * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    proj_x = u_x + t * dx
    proj_y = u_y + t * dy
    perp_dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    return perp_dist, t, proj_x, proj_y
