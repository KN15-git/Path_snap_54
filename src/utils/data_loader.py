import os
import glob
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.environ.get(
    "PATHSNAP_DATA_DIR",
    os.path.join(PROJECT_ROOT, "Student folder")
)

def find_trip_folder(trip_id: str, base_dir: str = DEFAULT_DATA_DIR):
    """
    Locates the directory containing trip_id, checking 'evaluator_test', 'student_test', and 'train'.
    """
    for partition in ["evaluator_test", "student_test", "train"]:
        candidate = os.path.join(base_dir, partition, str(trip_id))
        if os.path.exists(candidate):
            return candidate, partition
    raise FileNotFoundError(f"Trip {trip_id} not found in {base_dir}")

def list_trip_ids(partition: str = "train", base_dir: str = DEFAULT_DATA_DIR):
    """
    Returns sorted list of trip IDs in the specified partition.
    """
    partition_dir = os.path.join(base_dir, partition)
    if not os.path.exists(partition_dir):
        return []
    trips = [d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))]
    return sorted(trips)

def get_evaluation_trip_ids(partition: str = "student_test", base_dir: str = DEFAULT_DATA_DIR):
    """
    Returns the evaluation trips for the given partition (default: student_test).
    """
    return list_trip_ids(partition=partition, base_dir=base_dir)

def load_trip(trip_id: str, base_dir: str = DEFAULT_DATA_DIR):
    """
    Loads all 4 raw files for a trip:
    Returns dict:
        - 'trip_id': str
        - 'partition': 'train' | 'student_test'
        - 'track': DataFrame with columns ['lon', 'lat', 'time']
        - 'nodes': float numpy array of shape (num_nodes, 2) [lon, lat]
        - 'arcs': int numpy array of shape (num_arcs, 2) [from_node, to_node]
        - 'route': int numpy array of ground truth arc sequence (or None if not present)
    """
    folder, partition = find_trip_folder(trip_id, base_dir)
    trip_str = str(trip_id)

    # 1. Load .track
    track_file = os.path.join(folder, f"{trip_str}.track")
    track_data = np.loadtxt(track_file, dtype=float)
    track_df = pd.DataFrame(track_data, columns=["lon", "lat", "time"])

    # 2. Load .nodes
    nodes_file = os.path.join(folder, f"{trip_str}.nodes")
    nodes_data = np.loadtxt(nodes_file, dtype=float)

    # 3. Load .arcs
    arcs_file = os.path.join(folder, f"{trip_str}.arcs")
    arcs_data = np.loadtxt(arcs_file, dtype=int)

    # 4. Load .route (ground truth reference route if present)
    route_file = os.path.join(folder, f"{trip_str}.route")
    if os.path.exists(route_file):
        route_data = np.loadtxt(route_file, dtype=int)
        if route_data.ndim == 0:
            route_data = np.array([route_data], dtype=int)
    else:
        route_data = None

    return {
        "trip_id": trip_str,
        "partition": partition,
        "track": track_df,
        "nodes": nodes_data,
        "arcs": arcs_data,
        "route": route_data,
        "folder": folder
    }
