import math
import numpy as np
from scipy.spatial import cKDTree
from src.utils.geo import to_local_meters, project_point_to_segment, angular_difference

class RoadNetworkSpatialIndex:
    """
    Fast 2D spatial index over road network arcs for candidate generation.
    """
    def __init__(self, nodes, arcs, origin_lon=None, origin_lat=None, sample_step_meters=25.0):
        self.raw_nodes = nodes
        self.arcs = arcs
        self.sample_step = sample_step_meters

        # Determine origin for projection if not given
        if origin_lon is None or origin_lat is None:
            self.origin_lon = float(np.mean(nodes[:, 0]))
            self.origin_lat = float(np.mean(nodes[:, 1]))
        else:
            self.origin_lon = origin_lon
            self.origin_lat = origin_lat

        # Project all nodes to local Euclidean coordinates (meters)
        self.node_x, self.node_y = to_local_meters(nodes[:, 0], nodes[:, 1], self.origin_lon, self.origin_lat)

        # Precompute arc geometries
        self.num_arcs = len(arcs)
        u_indices = arcs[:, 0]
        v_indices = arcs[:, 1]

        self.u_x = self.node_x[u_indices]
        self.u_y = self.node_y[u_indices]
        self.v_x = self.node_x[v_indices]
        self.v_y = self.node_y[v_indices]

        dx = self.v_x - self.u_x
        dy = self.v_y - self.u_y
        self.arc_lengths = np.sqrt(dx * dx + dy * dy)

        # Compute arc bearing in degrees [0, 360)
        # In Cartesian: atan2(dx, dy) where y is North, x is East
        self.arc_bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

        # Build KD-tree with sample points along arcs
        sample_pts = []
        sample_arc_ids = []

        for arc_id in range(self.num_arcs):
            length = self.arc_lengths[arc_id]
            ux, uy = self.u_x[arc_id], self.u_y[arc_id]
            vx, vy = self.v_x[arc_id], self.v_y[arc_id]

            if length <= self.sample_step:
                # Add midpoint
                sample_pts.append([(ux + vx) * 0.5, (uy + vy) * 0.5])
                sample_arc_ids.append(arc_id)
            else:
                # Discretize along segment
                n_samples = max(2, int(math.ceil(length / self.sample_step)))
                ts = np.linspace(0.0, 1.0, n_samples)
                for t in ts:
                    sample_pts.append([ux + t * (vx - ux), uy + t * (vy - uy)])
                    sample_arc_ids.append(arc_id)

        self.sample_pts = np.array(sample_pts, dtype=np.float32)
        self.sample_arc_ids = np.array(sample_arc_ids, dtype=np.int32)
        self.tree = cKDTree(self.sample_pts)

    def get_candidates_for_point(self, px, py, radius=40.0, max_candidates=25, min_candidates=3):
        """
        Finds candidate arc IDs for a local metric point (px, py).
        Dynamically widens search radius if fewer than min_candidates are found.
        """
        # Query radius
        indices = self.tree.query_ball_point([px, py], r=radius)
        if len(indices) < min_candidates:
            # Expand radius to 80m
            indices = self.tree.query_ball_point([px, py], r=radius * 2.0)
            if len(indices) < min_candidates:
                # Top k nearest neighbors fallback
                _, indices = self.tree.query([px, py], k=min_candidates)
                if not hasattr(indices, "__iter__"):
                    indices = [indices]

        # Extract unique arc IDs
        candidate_arc_ids = np.unique(self.sample_arc_ids[indices])

        # Exact distance filtering and sorting
        candidates_data = []
        for arc_id in candidate_arc_ids:
            ux, uy = self.u_x[arc_id], self.u_y[arc_id]
            vx, vy = self.v_x[arc_id], self.v_y[arc_id]
            perp_dist, t, proj_x, proj_y = project_point_to_segment(px, py, ux, uy, vx, vy)
            candidates_data.append({
                "arc_id": int(arc_id),
                "perp_dist": perp_dist,
                "t": t,
                "arc_len": float(self.arc_lengths[arc_id]),
                "arc_bearing": float(self.arc_bearings[arc_id]),
                "dist_u": float(math.sqrt((px - ux)**2 + (py - uy)**2)),
                "dist_v": float(math.sqrt((px - vx)**2 + (py - vy)**2)),
                "u_id": int(self.arcs[arc_id, 0]),
                "v_id": int(self.arcs[arc_id, 1])
            })

        # Sort by perpendicular distance ascending
        candidates_data.sort(key=lambda item: item["perp_dist"])
        return candidates_data[:max_candidates]
