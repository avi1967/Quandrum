import numpy as np
from typing import List, Dict, Any, Tuple
from qubo import compute_qubo_energy

def solve_static_dijkstra(num_demands: int) -> List[int]:
    """
    Static Dijkstra baseline:
    Selects candidate path 0 (shortest base latency path) for all demands.
    """
    return [0] * num_demands

def solve_greedy_congestion(
    G: Any,
    demands: List[Dict[str, Any]],
    candidate_paths: List[List[List[Tuple[int, int]]]],
    k: int = 4,
    load_mult: float = 1.0
) -> List[int]:
    """
    Greedy congestion-aware routing baseline:
    1. Sorts demands by bandwidth in descending order (largest flow first).
    2. Sequentially places each flow on the candidate path p that minimizes incremental congestion-aware cost.
    """
    num_demands = len(demands)
    selection = [0] * num_demands
    
    # Track current cumulative load per link
    current_loads = { (min(u, v), max(u, v)): 0.0 for u, v in G.edges() }
    
    # Sort demand indices by bandwidth descending
    sorted_demand_indices = sorted(range(num_demands), key=lambda idx: demands[idx]['bandwidth'], reverse=True)
    
    for d_idx in sorted_demand_indices:
        bw = demands[d_idx]['bandwidth'] * load_mult
        best_p = 0
        best_cost = float('inf')
        
        for p_idx in range(k):
            path_edges = candidate_paths[d_idx][p_idx]
            path_cost = 0.0
            
            for u, v in path_edges:
                edge = (min(u, v), max(u, v))
                base_lat = G[u][v]['base_latency']
                cap = G[u][v]['capacity']
                
                new_load = current_loads[edge] + bw
                util = new_load / cap if cap > 0 else 0.0
                
                # Congestion-aware incremental latency for this link
                link_lat = base_lat * (1.0 + util ** 2)
                path_cost += link_lat
                
            if path_cost < best_cost:
                best_cost = path_cost
                best_p = p_idx
                
        selection[d_idx] = best_p
        
        # Update link loads with chosen path
        chosen_edges = candidate_paths[d_idx][best_p]
        for u, v in chosen_edges:
            edge = (min(u, v), max(u, v))
            current_loads[edge] += bw

    return selection

def solve_simulated_annealing(
    Q: np.ndarray,
    num_demands: int,
    k: int = 4,
    steps: int = 1000,
    t_start: float = 100.0,
    t_end: float = 0.1,
    seed: int = 42
) -> List[int]:
    """
    Simulated Annealing baseline operating directly on the QUBO objective within the feasible space (1 path per demand).
    """
    rng = np.random.RandomState(seed)
    
    # Initial state: random selection
    current_sel = [int(rng.randint(0, k)) for _ in range(num_demands)]
    
    # Helper to construct binary vector x
    def sel_to_x(sel: List[int]) -> np.ndarray:
        x = np.zeros(num_demands * k, dtype=np.float64)
        for td, tp in enumerate(sel):
            x[td * k + tp] = 1.0
        return x

    current_x = sel_to_x(current_sel)
    current_energy = compute_qubo_energy(current_x, Q)
    
    best_sel = list(current_sel)
    best_energy = current_energy
    
    # Exponential cooling factor
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, steps))
    T = t_start
    
    for step in range(steps):
        # Pick random demand and alternative path
        d_idx = rng.randint(0, num_demands)
        current_p = current_sel[d_idx]
        other_paths = [p for p in range(k) if p != current_p]
        if not other_paths:
            continue
        cand_p = int(rng.choice(other_paths))
        
        # Test neighbor state
        test_x = current_x.copy()
        test_x[d_idx * k + current_p] = 0.0
        test_x[d_idx * k + cand_p] = 1.0
        
        test_energy = compute_qubo_energy(test_x, Q)
        delta = test_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta < 0 or rng.uniform() < np.exp(-delta / max(1e-6, T)):
            current_sel[d_idx] = cand_p
            current_x = test_x
            current_energy = test_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_sel = list(current_sel)
                
        T *= cooling_rate

    return best_sel
