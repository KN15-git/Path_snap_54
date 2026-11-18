import math
import numpy as np
import networkx as nx
from src.utils.geo import haversine_distance

class RoadNetworkGraph:
    """
    Directed graph representation of the road network with fast bounded Dijkstra caching.
    """
    def __init__(self, nodes, arcs, arc_lengths=None):
        self.G = nx.DiGraph()
        self.arcs = arcs
        self.nodes = nodes
        self._dist_cache = {}

        num_arcs = len(arcs)
        for arc_id in range(num_arcs):
            u = int(arcs[arc_id, 0])
            v = int(arcs[arc_id, 1])
            if arc_lengths is not None:
                length = float(arc_lengths[arc_id])
            else:
                length = float(haversine_distance(nodes[u, 0], nodes[u, 1], nodes[v, 0], nodes[v, 1]))
            self.G.add_edge(u, v, arc_id=arc_id, weight=max(0.1, length))

    def get_shortest_distance(self, from_node, to_node, cutoff=300.0):
        """
        Calculates network shortest driving distance between from_node and to_node
        using bounded single-source Dijkstra with memoization.
        """
        if from_node == to_node:
            return 0.0
        if from_node not in self._dist_cache:
            if from_node not in self.G:
                self._dist_cache[from_node] = {}
            else:
                self._dist_cache[from_node] = nx.single_source_dijkstra_path_length(
                    self.G, source=from_node, cutoff=cutoff, weight="weight"
                )
        return self._dist_cache[from_node].get(to_node, None)

class HMMViterbiDecoder:
    """
    Hidden Markov Model route reconstruction using Viterbi dynamic programming
    with directed segment progress, U-turn twin-arc suppression, and ML emissions.
    """
    def __init__(self, road_graph, beta=12.0, sigma=10.0):
        self.graph = road_graph
        self.beta = beta
        self.sigma = sigma

    def decode(self, candidates_per_point, model_probs_per_point, track_df):
        """
        Executes Viterbi log-trellis decoding over candidate arcs for each GPS point.
        """
        n_steps = len(candidates_per_point)
        if n_steps == 0:
            return [], []

        lons = track_df["lon"].values
        lats = track_df["lat"].values
        times = track_df["time"].values

        V = []
        backpointer = []

        # 1. Initialize t = 0
        cands_0 = candidates_per_point[0]
        probs_0 = model_probs_per_point[0]
        v_0 = np.zeros(len(cands_0), dtype=np.float64)

        for i, cand in enumerate(cands_0):
            ml_prob = max(1e-4, float(probs_0[i]))
            dir_cost = cand.get("dir_cost", cand["perp_dist"])
            v_0[i] = 0.65 * math.log(ml_prob) - (dir_cost / self.sigma)

        V.append(v_0)
        backpointer.append([0] * len(cands_0))

        # 2. Forward Recursion for t = 1 ... n_steps - 1
        for t in range(1, n_steps):
            cands_prev = candidates_per_point[t - 1]
            cands_curr = candidates_per_point[t]
            probs_curr = model_probs_per_point[t]

            dt = max(1e-3, times[t] - times[t - 1])
            gps_dist = float(haversine_distance(lons[t - 1], lats[t - 1], lons[t], lats[t]))

            v_t = np.full(len(cands_curr), -1e9, dtype=np.float64)
            bp_t = [0] * len(cands_curr)

            curr_emissions = np.zeros(len(cands_curr), dtype=np.float64)
            for j, cand_j in enumerate(cands_curr):
                ml_prob = max(1e-4, float(probs_curr[j]))
                dir_cost = cand_j.get("dir_cost", cand_j["perp_dist"])
                curr_emissions[j] = 0.65 * math.log(ml_prob) - (dir_cost / self.sigma)

            for j, cand_j in enumerate(cands_curr):
                best_val = -1e9
                best_i = 0

                for i, cand_i in enumerate(cands_prev):
                    if cand_i["arc_id"] == cand_j["arc_id"]:
                        # Same directed road segment: check forward progress along segment
                        ds = (cand_j["t"] - cand_i["t"]) * cand_i["arc_len"]
                        if ds < -4.0:
                            log_p_trans = -6.0
                        else:
                            log_p_trans = -abs(max(0.0, ds) - gps_dist) / self.beta + 0.85
                    elif cand_i["u_id"] == cand_j["v_id"] and cand_i["v_id"] == cand_j["u_id"]:
                        # Immediate U-turn onto opposite-direction twin arc on a two-way street
                        log_p_trans = -18.0
                    elif cand_i["v_id"] == cand_j["u_id"]:
                        # Directly connected consecutive segments (v_i == u_j)
                        ds = (1.0 - cand_i["t"]) * cand_i["arc_len"] + cand_j["t"] * cand_j["arc_len"]
                        log_p_trans = -abs(ds - gps_dist) / self.beta + 0.65
                    else:
                        # Graph shortest-path transition
                        net_dist = self.graph.get_shortest_distance(cand_i["v_id"], cand_j["u_id"], cutoff=300.0)
                        if net_dist is None:
                            log_p_trans = -4.5 if dt > 5.0 else -20.0
                        else:
                            ds = (1.0 - cand_i["t"]) * cand_i["arc_len"] + net_dist + cand_j["t"] * cand_j["arc_len"]
                            log_p_trans = -abs(ds - gps_dist) / 10.0 - 0.8

                    candidate_total = V[t - 1][i] + log_p_trans + curr_emissions[j]
                    if candidate_total > best_val:
                        best_val = candidate_total
                        best_i = i

                v_t[j] = best_val
                bp_t[j] = best_i

            V.append(v_t)
            backpointer.append(bp_t)

        # 3. Backtrack optimal sequence
        best_last_idx = int(np.argmax(V[-1]))
        best_path_indices = [best_last_idx]

        for t in range(n_steps - 1, 0, -1):
            prev_idx = backpointer[t][best_path_indices[-1]]
            best_path_indices.append(prev_idx)

        best_path_indices.reverse()

        predicted_arcs = [int(candidates_per_point[t][idx]["arc_id"]) for t, idx in enumerate(best_path_indices)]

        reconstructed_route = []
        for arc in predicted_arcs:
            if len(reconstructed_route) == 0 or reconstructed_route[-1] != arc:
                reconstructed_route.append(arc)

        return predicted_arcs, reconstructed_route

def evaluate_route_predictions(predicted_arcs, true_arcs_per_point, road_arcs):
    """
    Evaluates both observation-level accuracy and topological route continuity
    directly against the chronological reference sequence true_arcs_per_point.
    Allows a +/-1 observation tolerance at segment transition boundaries where
    1 Hz sampling straddles an intersection node.
    """
    n = len(predicted_arcs)
    if n == 0:
        return {"top1_accuracy": 0.0, "strict_accuracy": 0.0, "route_continuity": 0.0, "matched_points": 0, "total_points": 0}

    correct = 0
    strict_correct = 0
    for i, p in enumerate(predicted_arcs):
        if p == true_arcs_per_point[i]:
            strict_correct += 1
            correct += 1
        elif p == true_arcs_per_point[max(0, i - 1)] or p == true_arcs_per_point[min(n - 1, i + 1)]:
            correct += 1

    top1_acc = correct / n
    strict_acc = strict_correct / n

    out_neighbors = {}
    for idx_a in range(len(road_arcs)):
        u = int(road_arcs[idx_a, 0])
        v = int(road_arcs[idx_a, 1])
        if u not in out_neighbors:
            out_neighbors[u] = set()
        out_neighbors[u].add(v)

    continuous_transitions = 0
    total_transitions = max(1, n - 1)

    for i in range(n - 1):
        a1 = predicted_arcs[i]
        a2 = predicted_arcs[i + 1]
        if a1 == a2:
            continuous_transitions += 1
        else:
            v1 = int(road_arcs[a1, 1])
            u2 = int(road_arcs[a2, 0])
            if v1 == u2 or (v1 in out_neighbors and u2 in out_neighbors[v1]):
                continuous_transitions += 1

    continuity = continuous_transitions / total_transitions

    return {
        "top1_accuracy": round(float(top1_acc) * 100.0, 2),
        "strict_accuracy": round(float(strict_acc) * 100.0, 2),
        "route_continuity": round(float(continuity) * 100.0, 2),
        "matched_points": int(correct),
        "total_points": int(n)
    }
