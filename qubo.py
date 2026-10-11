import numpy as np
from typing import List, Dict, Any, Tuple

def build_qubo_matrix(
    G: Any,
    demands: List[Dict[str, Any]],
    candidate_paths: List[List[List[Tuple[int, int]]]],
    A: float = 20.0,
    B: float = 20.0,
    C: float = 1.0,
    load_mult: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Constructs the QUBO matrix Q of dimension N x N, where N = num_demands * k.
    Variable index i = d * k + p represents picking path p for demand d.
    
    Q = Latency (diag) + A * OneHot + B * Congestion + C * Failure
    """
    num_demands = len(demands)
    k = len(candidate_paths[0])
    N = num_demands * k
    
    Q = np.zeros((N, N), dtype=np.float64)
    
    # 1. Base Latency (Diagonal) & 4. Failure Probability (Diagonal)
    for d_idx in range(num_demands):
        for p_idx in range(k):
            i = d_idx * k + p_idx
            path_edges = candidate_paths[d_idx][p_idx]
            
            # Base Latency
            base_lat = sum(G[u][v]['base_latency'] for u, v in path_edges)
            Q[i, i] += base_lat
            
            # Failure Probability
            succ_p = 1.0
            for u, v in path_edges:
                succ_p *= (1.0 - G[u][v]['failure_prob'])
            fail_p = 1.0 - succ_p
            Q[i, i] += C * fail_p

    # 2. One-Hot Penalty: A * (sum_p x_{d,p} - 1)^2 = A * (sum x^2 + 2 sum x_p x_p' - 2 sum x)
    # Since x^2 = x for binary, x^2 - 2x = -x on diagonal.
    for d_idx in range(num_demands):
        for p1 in range(k):
            i1 = d_idx * k + p1
            Q[i1, i1] += A * (-1.0) # -A on diagonal
            for p2 in range(p1 + 1, k):
                i2 = d_idx * k + p2
                # Off-diagonal pair (i1, i2) has term +2*A*x_i1*x_i2.
                # In symmetric Q, Q[i1,i2] + Q[i2,i1] = 2*A, so each gets A.
                Q[i1, i2] += A
                Q[i2, i1] += A

    # 3. Congestion Penalty: B * sum_e ( (sum_{d,p: e in p} bw_d * x_{d,p}) / cap_e )^2 * sqrt(base_lat_e)
    # = B * sum_e (sqrt(base_lat_e) / cap_e^2) * sum_{i} sum_{j} bw_{d(i)} * bw_{d(j)} * x_i * x_j
    for u, v, data in G.edges(data=True):
        edge = (min(u, v), max(u, v))
        cap = data['capacity']
        base_lat = data['base_latency']
        if cap <= 0:
            continue
            
        weight_e = (np.sqrt(base_lat) / (cap ** 2)) * (load_mult ** 2)
        
        # Find all (d, p) variables whose path contains this edge
        active_vars = []
        for d_idx in range(num_demands):
            bw = demands[d_idx]['bandwidth']
            for p_idx in range(k):
                if edge in candidate_paths[d_idx][p_idx]:
                    var_idx = d_idx * k + p_idx
                    active_vars.append((var_idx, bw))
                    
        # Add pairwise congestion terms to Q
        for i_v, (var_i, bw_i) in enumerate(active_vars):
            for j_v, (var_j, bw_j) in enumerate(active_vars):
                contrib = B * weight_e * bw_i * bw_j
                Q[var_i, var_j] += contrib

    # Ensure exact symmetry
    Q = 0.5 * (Q + Q.T)
    
    meta = {
        'num_demands': num_demands,
        'k': k,
        'N': N,
        'A': A,
        'B': B,
        'C': C
    }
    
    return Q, meta

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Converts symmetric QUBO matrix Q to Ising coupling matrix J, magnetic field h, and constant offset.
    x in {0, 1}^N, s in {-1, 1}^N with x = (s + 1) / 2.
    
    E(s) = -0.5 * s^T J s - h^T s + offset = x^T Q x
    """
    N = Q.shape[0]
    Q_diag = np.diag(Q)
    Q_off = Q - np.diag(Q_diag)
    
    # J = -0.5 * offdiag(Q)
    J = -0.5 * Q_off
    
    # h = - (0.5 * Q_off.sum(axis=1) + 0.5 * Q_diag)
    h = - (0.5 * np.sum(Q_off, axis=1) + 0.5 * Q_diag)
    
    # offset such that x^T Q x == E_ising(s) + offset
    offset = 0.25 * np.sum(Q_off) + 0.5 * np.sum(Q_diag)
    
    return J, h, offset

def compute_qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """Computes binary x^T Q x energy."""
    return float(x.T @ Q @ x)

def compute_ising_energy(s: np.ndarray, J: np.ndarray, h: np.ndarray, offset: float = 0.0) -> float:
    """Computes Ising energy -0.5 * s^T J s - h^T s + offset."""
    return float(-0.5 * s.T @ J @ s - h.T @ s + offset)

def decode_with_repair(
    raw_spins: np.ndarray,
    Q: np.ndarray,
    num_demands: int,
    k: int
) -> Tuple[List[int], float]:
    """
    Decodes raw spin vector s in {-1, 1}^N (or binary x) into a valid selection vector (1 path per demand).
    Calculates and returns raw feasibility rate BEFORE repair.
    Uses marginal QUBO energy to repair broken demands.
    """
    # Map spins to binary (if input is spins in {-1, 1})
    if np.any(raw_spins < 0):
        raw_x = (raw_spins > 0).astype(int)
    else:
        raw_x = (raw_spins > 0.5).astype(int)

    # Calculate raw feasibility rate BEFORE repair
    feasible_demands = 0
    for d_idx in range(num_demands):
        paths_active = np.sum(raw_x[d_idx * k : (d_idx + 1) * k])
        if paths_active == 1:
            feasible_demands += 1
            
    raw_feasibility_rate = float(feasible_demands / num_demands) if num_demands > 0 else 0.0

    # Initial selection from raw output where valid, default to path 0 otherwise
    selection = [0] * num_demands
    for d_idx in range(num_demands):
        active_indices = np.where(raw_x[d_idx * k : (d_idx + 1) * k] == 1)[0]
        if len(active_indices) == 1:
            selection[d_idx] = int(active_indices[0])
        else:
            # Candidate path with lowest diagonal / base cost for broken demand
            selection[d_idx] = 0

    # Repair step using marginal energy minimization for demands that were infeasible
    for d_idx in range(num_demands):
        active_indices = np.where(raw_x[d_idx * k : (d_idx + 1) * k] == 1)[0]
        if len(active_indices) != 1:
            # Test all k candidate paths for this demand, evaluate full QUBO energy
            best_p = 0
            best_energy = float('inf')
            
            for p_cand in range(k):
                test_sel = list(selection)
                test_sel[d_idx] = p_cand
                
                # Convert test_sel to binary vector
                test_x = np.zeros(num_demands * k, dtype=np.float64)
                for td, tp in enumerate(test_sel):
                    test_x[td * k + tp] = 1.0
                    
                energy = compute_qubo_energy(test_x, Q)
                if energy < best_energy:
                    best_energy = energy
                    best_p = p_cand
                    
            selection[d_idx] = best_p

    return selection, raw_feasibility_rate

def feasible_coordinate_descent_polish(
    selection: List[int],
    Q: np.ndarray,
    num_demands: int,
    k: int,
    max_passes: int = 10
) -> List[int]:
    """
    Feasible-space coordinate-descent polish:
    Given a valid selection vector (1 path per demand), tests swapping path of demand d
    to another candidate path p' in [0, k-1].
    If energy improves, updates selection. Repeats until convergence.
    """
    polished_sel = list(selection)
    
    # Construct binary vector x
    x = np.zeros(num_demands * k, dtype=np.float64)
    for td, tp in enumerate(polished_sel):
        x[td * k + tp] = 1.0
        
    current_energy = compute_qubo_energy(x, Q)
    
    for _ in range(max_passes):
        improved = False
        for d_idx in range(num_demands):
            current_p = polished_sel[d_idx]
            best_p = current_p
            best_e = current_energy
            
            for p_cand in range(k):
                if p_cand == current_p:
                    continue
                    
                # Create test binary vector
                test_x = x.copy()
                test_x[d_idx * k + current_p] = 0.0
                test_x[d_idx * k + p_cand] = 1.0
                
                test_e = compute_qubo_energy(test_x, Q)
                if test_e < best_e - 1e-6:
                    best_e = test_e
                    best_p = p_cand
                    
            if best_p != current_p:
                x[d_idx * k + current_p] = 0.0
                x[d_idx * k + best_p] = 1.0
                polished_sel[d_idx] = best_p
                current_energy = best_e
                improved = True
                
        if not improved:
            break
            
    return polished_sel
