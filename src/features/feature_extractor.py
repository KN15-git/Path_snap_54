import math
import numpy as np
import pandas as pd
from src.utils.geo import to_local_meters, angular_difference
from src.features.candidate_generator import RoadNetworkSpatialIndex

FEATURE_NAMES = [
    "perp_dist",        # Orthogonal distance from GPS to road segment (m)
    "t_proj",           # Normalized position along road segment [0, 1]
    "dist_u",           # Distance to start node (m)
    "dist_v",           # Distance to end node (m)
    "arc_len",          # Length of candidate road segment (m)
    "bearing_diff",     # Angle discrepancy between adaptive trajectory heading & road [0, 180 deg]
    "gps_speed",        # Vehicle observed speed (m/s)
    "rel_dir_cost",     # Relative directional-distance advantage vs closest candidate at step t
    "spatial_prior"     # Directional-Gaussian prior exp(-d^2 / (2*sigma^2)) * cos_factor
]

def compute_gps_kinematics(track_df, origin_lon, origin_lat):
    """
    Computes local (x, y) meters, speeds, and adaptive trajectory bearings for all GPS observations.
    Adaptive window expands until displacement >= 20m so stationary GPS jitter never corrupts heading.
    """
    lons = track_df["lon"].values
    lats = track_df["lat"].values
    times = track_df["time"].values
    n = len(lons)

    xs, ys = to_local_meters(lons, lats, origin_lon, origin_lat)

    speeds = np.zeros(n, dtype=np.float32)
    bearings = np.zeros(n, dtype=np.float32)
    dts = np.ones(n, dtype=np.float32)

    for i in range(n):
        if i == 0:
            if n > 1:
                dx = xs[1] - xs[0]
                dy = ys[1] - ys[0]
                dt = max(1e-3, times[1] - times[0])
            else:
                dx, dy, dt = 0.0, 0.0, 1.0
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            dt = max(1e-3, times[i] - times[i - 1])

        dist = math.sqrt(dx * dx + dy * dy)
        speeds[i] = dist / dt
        dts[i] = dt

        # Adaptive trajectory window (>= 20m displacement) to eliminate low-speed GPS bearing noise
        best_dx, best_dy = 0.0, 0.0
        for w in [2, 4, 8, 15, 30, 60, 120]:
            i0 = max(0, i - w)
            i1 = min(n - 1, i + w)
            wx = xs[i1] - xs[i0]
            wy = ys[i1] - ys[i0]
            if wx * wx + wy * wy >= 400.0:
                best_dx, best_dy = wx, wy
                break
        if best_dx == 0.0 and best_dy == 0.0 and n > 1:
            best_dx = xs[-1] - xs[0]
            best_dy = ys[-1] - ys[0]

        bearings[i] = (math.degrees(math.atan2(best_dx, best_dy)) + 360.0) % 360.0

    return xs, ys, speeds, bearings, dts

def match_ground_truth_route(track_xs, track_ys, bearings, route_arcs, index):
    """
    Associates each GPS observation with its ground-truth arc from the chronological route sequence
    using vectorized monotonic dynamic programming alignment.
    """
    n_pts = len(track_xs)
    r_arr = np.array(route_arcs, dtype=int)
    n_r = len(r_arr)

    ux, uy = index.u_x[r_arr], index.u_y[r_arr]
    vx, vy = index.v_x[r_arr], index.v_y[r_arr]
    dx, dy = vx - ux, vy - uy
    len_sq = np.maximum(1e-12, dx * dx + dy * dy)
    arc_b = index.arc_bearings[r_arr]

    px, py = track_xs[:, None], track_ys[:, None]
    t = np.clip(((px - ux) * dx + (py - uy) * dy) / len_sq, 0.0, 1.0)
    d = np.sqrt((px - (ux + t * dx))**2 + (py - (uy + t * dy))**2)
    b_diff = angular_difference(bearings[:, None], arc_b[None, :])
    cost = d + 0.08 * b_diff

    dp = np.full((n_pts, n_r), 1e12, dtype=np.float64)
    bp = np.zeros((n_pts, n_r), dtype=np.int32)
    dp[0, :min(15, n_r)] = cost[0, :min(15, n_r)]

    max_jump = 12
    for i in range(1, n_pts):
        prev = dp[i - 1]
        best_prev = prev.copy()
        best_src = np.arange(n_r, dtype=np.int32)
        for shift in range(1, min(max_jump, n_r)):
            pen = 0.0 if shift <= 1 else (shift - 1) * 1.5
            cand_val = prev[:-shift] + pen
            mask = cand_val < best_prev[shift:]
            best_prev[shift:][mask] = cand_val[mask]
            best_src[shift:][mask] = np.arange(n_r - shift, dtype=np.int32)[mask]
        dp[i] = best_prev + cost[i]
        bp[i] = best_src

    curr = int(np.argmin(dp[-1]))
    path = [curr]
    for i in range(n_pts - 1, 0, -1):
        curr = int(bp[i, curr])
        path.append(curr)
    path.reverse()
    return r_arr[path]

def extract_features_for_trip(trip_dict, search_radius=45.0, max_candidates=25, sigma_prior=15.0):
    """
    Extracts 9D candidate feature matrix X, ground-truth labels y, candidate metadata,
    and candidate lists for all GPS observations in a trip.
    """
    track_df = trip_dict["track"]
    nodes = trip_dict["nodes"]
    arcs = trip_dict["arcs"]
    route = trip_dict["route"]

    index = RoadNetworkSpatialIndex(nodes, arcs)
    xs, ys, speeds, bearings, dts = compute_gps_kinematics(track_df, index.origin_lon, index.origin_lat)

    n_pts = len(xs)
    route_set = set(int(a) for a in route) if (route is not None and len(route) > 0) else None

    if route is not None and len(route) > 0:
        true_arcs_per_point = match_ground_truth_route(xs, ys, bearings, route, index)
    else:
        true_arcs_per_point = None

    rows = []
    labels = []
    groups = []
    metadata = []
    candidates_per_point = []
    miss_count = 0

    for i in range(n_pts):
        px = xs[i]
        py = ys[i]
        speed = speeds[i]
        bearing = bearings[i]
        true_arc = true_arcs_per_point[i] if true_arcs_per_point is not None else None

        candidates = index.get_candidates_for_point(px, py, radius=search_radius, max_candidates=max_candidates)
        candidates_per_point.append(candidates)

        cand_arc_ids = [c["arc_id"] for c in candidates]
        if true_arc is not None:
            if true_arc not in cand_arc_ids and not (route_set and any(cid in route_set for cid in cand_arc_ids)):
                miss_count += 1

        # Compute directional-spatial cost across candidates at timestamp i for relative context
        dir_costs = [
            c["perp_dist"] + 0.15 * float(angular_difference(bearing, c["arc_bearing"]))
            for c in candidates
        ]
        min_dir_cost = min(dir_costs) if dir_costs else 0.0

        for c, d_cost in zip(candidates, dir_costs):
            perp_dist = c["perp_dist"]
            t_proj = c["t"]
            dist_u = c["dist_u"]
            dist_v = c["dist_v"]
            arc_len = c["arc_len"]
            arc_bearing = c["arc_bearing"]

            bearing_diff = float(angular_difference(bearing, arc_bearing))
            cos_factor = max(0.05, 0.5 * (1.0 + math.cos(math.radians(bearing_diff))))
            spatial_prior = math.exp(-(perp_dist**2) / (2.0 * sigma_prior**2)) * cos_factor
            rel_dir_cost = d_cost - min_dir_cost

            # Store bearing_diff inside candidate dict as well for HMM emission refinement
            c["bearing_diff"] = bearing_diff
            c["dir_cost"] = d_cost

            feat_vec = [
                perp_dist,
                t_proj,
                dist_u,
                dist_v,
                arc_len,
                bearing_diff,
                speed,
                rel_dir_cost,
                spatial_prior
            ]

            rows.append(feat_vec)
            groups.append(trip_dict["trip_id"])
            metadata.append({
                "point_idx": i,
                "arc_id": c["arc_id"],
                "true_arc": true_arc,
                "perp_dist": perp_dist,
                "time": float(track_df["time"].iloc[i]),
                "u_id": c["u_id"],
                "v_id": c["v_id"]
            })

            if true_arc is not None:
                labels.append(1 if c["arc_id"] == true_arc else 0)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if len(labels) > 0 else None
    groups = np.array(groups)
    miss_rate = miss_count / max(1, n_pts)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "metadata": metadata,
        "candidates_per_point": candidates_per_point,
        "spatial_index": index,
        "miss_rate": miss_rate,
        "true_arcs_per_point": true_arcs_per_point,
        "route_set": route_set
    }
