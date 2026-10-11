import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def create_random_network(num_nodes: int = 20, radius: float = 0.35, seed: int = 42) -> nx.Graph:
    """
    Generates a Random Geometric Graph (20 nodes by default) ensuring full connectivity.
    Each edge is assigned base_latency, capacity, and failure_prob.
    """
    rng = np.random.RandomState(seed)
    
    # Try generating connected graph, increment radius slightly if needed
    current_radius = radius
    for _ in range(50):
        G = nx.random_geometric_graph(num_nodes, current_radius, seed=seed)
        if nx.is_connected(G):
            break
        current_radius += 0.05
    else:
        # If still disconnected, connect components
        G = nx.random_geometric_graph(num_nodes, current_radius, seed=seed)
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)

    # Assign edge attributes
    for u, v in G.edges():
        G[u][v]['base_latency'] = float(rng.uniform(5.0, 25.0))      # ms
        G[u][v]['capacity'] = float(rng.uniform(10.0, 50.0))          # Mbps
        G[u][v]['failure_prob'] = float(rng.uniform(0.01, 0.08))      # probability [0.01, 0.08]

    return G

def generate_demands(G: nx.Graph, num_demands: int = 15, seed: int = 42, k: int = 4) -> Tuple[List[Dict[str, Any]], List[List[List[Tuple[int, int]]]]]:
    """
    Generates routing demands (src, dst, bandwidth) and finds k shortest simple paths per demand based on base latency.
    """
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    demands = []
    candidate_paths = []
    
    pairs = []
    attempts = 0
    while len(pairs) < num_demands and attempts < 1000:
        attempts += 1
        src, dst = rng.choice(nodes, size=2, replace=False)
        pair = (int(src), int(dst))
        if pair not in pairs:
            pairs.append(pair)
            
    for src, dst in pairs:
        bw = float(rng.uniform(2.0, 10.0)) # Mbps
        demands.append({
            'src': src,
            'dst': dst,
            'bandwidth': bw
        })
        
        # Generator for simple paths ordered by base_latency weight
        def path_weight(u, v, d):
            return d['base_latency']
        
        path_gen = nx.shortest_simple_paths(G, src, dst, weight=path_weight)
        k_paths_nodes = []
        try:
            for _ in range(k):
                k_paths_nodes.append(next(path_gen))
        except StopIteration:
            # If fewer than k paths exist, pad with the last path
            while len(k_paths_nodes) < k:
                k_paths_nodes.append(k_paths_nodes[-1] if k_paths_nodes else [src, dst])
                
        # Convert node lists to edge lists (u, v) where u < v for undirected indexing
        k_paths_edges = []
        for p_nodes in k_paths_nodes:
            edges = []
            for i in range(len(p_nodes) - 1):
                u, v = p_nodes[i], p_nodes[i+1]
                edge = (min(u, v), max(u, v))
                edges.append(edge)
            k_paths_edges.append(edges)
            
        candidate_paths.append(k_paths_edges)

    return demands, candidate_paths

def compute_link_loads(G: nx.Graph, demands: List[Dict[str, Any]], candidate_paths: List[List[List[Tuple[int, int]]]], selection: List[int], load_mult: float = 1.0) -> Dict[Tuple[int, int], float]:
    """
    Computes cumulative bandwidth load for each undirected link (u, v) given flow choices.
    """
    link_loads = { (min(u, v), max(u, v)): 0.0 for u, v in G.edges() }
    
    for d_idx, p_idx in enumerate(selection):
        bw = demands[d_idx]['bandwidth'] * load_mult
        path_edges = candidate_paths[d_idx][p_idx]
        for edge in path_edges:
            if edge in link_loads:
                link_loads[edge] += bw
                
    return link_loads

def evaluate(G: nx.Graph, demands: List[Dict[str, Any]], candidate_paths: List[List[List[Tuple[int, int]]]], selection: List[int], load_mult: float = 1.0, disabled_links: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """
    Evaluates network metrics for a given selection vector (one path choice per demand).
    Congestion formula per link: base_latency * (1 + (load / capacity)^2)
    """
    if disabled_links is None:
        disabled_links_set = set()
    else:
        disabled_links_set = { (min(u, v), max(u, v)) for u, v in disabled_links }

    link_loads = compute_link_loads(G, demands, candidate_paths, selection, load_mult)
    
    # Calculate effective congestion latency per link
    link_latencies = {}
    link_utils = {}
    overloaded_links_count = 0
    
    for u, v, d in G.edges(data=True):
        edge = (min(u, v), max(u, v))
        cap = d['capacity']
        base_lat = d['base_latency']
        load = link_loads[edge]
        
        util = load / cap if cap > 0 else 0.0
        link_utils[edge] = util
        if util > 1.0:
            overloaded_links_count += 1
            
        # Congestion latency formula
        link_latencies[edge] = base_lat * (1.0 + util ** 2)
        
    path_latencies = []
    path_fail_probs = []
    flows_on_failed_links_count = 0
    flows_on_overloaded_links_count = 0
    latency_violations_count = 0
    
    for d_idx, p_idx in enumerate(selection):
        path_edges = candidate_paths[d_idx][p_idx]
        
        # Path base latency for budget check
        base_path_lat = sum(G[u][v]['base_latency'] for u, v in path_edges if G.has_edge(u, v))
        
        # Path congestion latency
        path_lat = sum(link_latencies[edge] for edge in path_edges if edge in link_latencies)
        path_latencies.append(path_lat)
        
        # Latency budget check (e.g., > 2.0x base path latency)
        if base_path_lat > 0 and path_lat > 2.0 * base_path_lat:
            latency_violations_count += 1
            
        # Path failure probability: 1 - prod(1 - fail_prob_e)
        succ_prob = 1.0
        has_disabled_link = False
        for edge in path_edges:
            u, v = edge
            if G.has_edge(u, v):
                fail_p = G[u][v]['failure_prob']
                succ_prob *= (1.0 - fail_p)
            if edge in disabled_links_set:
                has_disabled_link = True
                
        fail_prob = 1.0 - succ_prob
        path_fail_probs.append(fail_prob)
        
        # Separate flow tracking
        has_overloaded_link = any(link_utils.get(edge, 0.0) > 1.0 for edge in path_edges)
        
        if has_disabled_link:
            flows_on_failed_links_count += 1
        if has_overloaded_link:
            flows_on_overloaded_links_count += 1

    avg_latency = float(np.mean(path_latencies)) if path_latencies else 0.0
    max_latency = float(np.max(path_latencies)) if path_latencies else 0.0
    max_utilisation = float(np.max(list(link_utils.values()))) if link_utils else 0.0
    avg_path_failure_prob = float(np.mean(path_fail_probs)) if path_fail_probs else 0.0

    return {
        'avg_latency': avg_latency,
        'max_latency': max_latency,
        'max_utilisation': max_utilisation,
        'overloaded_links': overloaded_links_count,
        'avg_path_failure_prob': avg_path_failure_prob,
        'flows_on_failed_links': flows_on_failed_links_count,
        'flows_on_overloaded_links': flows_on_overloaded_links_count,
        'latency_budget_violations': latency_violations_count,
        'link_loads': link_loads,
        'link_latencies': link_latencies,
        'link_utils': link_utils,
        'path_latencies': path_latencies
    }
