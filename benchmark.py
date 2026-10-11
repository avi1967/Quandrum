import time
import json
import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional

from network import create_random_network, generate_demands, evaluate
from qubo import build_qubo_matrix, qubo_to_ising, decode_with_repair, feasible_coordinate_descent_polish
from baselines import solve_static_dijkstra, solve_greedy_congestion, solve_simulated_annealing
from sb_module import solve_simulated_bifurcation

def compute_stats(vals: List[float]) -> Dict[str, Any]:
    """
    Computes mean, sample standard deviation (ddof=1), and 95% Confidence Interval.
    Uses t-distribution critical value t=2.262157 for n=10 (df=9).
    """
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    t_crit = 2.262157 if n == 10 else 2.0
    margin = float(t_crit * (std_val / np.sqrt(n))) if n > 0 else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "ci95": [float(mean_val - margin), float(mean_val + margin)]
    }

def compute_paired_comparison(qubo_vals: List[float], greedy_vals: List[float], lower_is_better: bool = True) -> Dict[str, Any]:
    """
    Computes paired per-seed difference (qubo - greedy), W/T/L count, 95% CI of difference,
    whether CI excludes 0, and verbal conclusion ("better" if CI excludes 0 and improves metric; else "comparable").
    """
    diffs = [q - g for q, g in zip(qubo_vals, greedy_vals)]
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    t_crit = 2.262157 if n == 10 else 2.0
    margin = float(t_crit * (std_diff / np.sqrt(n)))
    ci95 = [float(mean_diff - margin), float(mean_diff + margin)]
    
    wins = sum(1 for d in diffs if (d < -1e-5 if lower_is_better else d > 1e-5))
    ties = sum(1 for d in diffs if abs(d) <= 1e-5)
    losses = sum(1 for d in diffs if (d > 1e-5 if lower_is_better else d < -1e-5))
    
    # Check if 95% CI of difference strictly excludes 0
    excludes_zero = bool(ci95[0] > 0 or ci95[1] < 0)
    
    if excludes_zero:
        if lower_is_better:
            conclusion = "better" if mean_diff < 0 else "worse"
        else:
            conclusion = "better" if mean_diff > 0 else "worse"
    else:
        conclusion = "comparable"
        
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_diff": ci95,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "excludes_zero": excludes_zero,
        "conclusion": conclusion
    }

def run_benchmark():
    print("=" * 80)
    print(" QUANTUM-INSPIRED NETWORK ROUTE OPTIMIZER - STATISTICAL BENCHMARK SUITE")
    print("=" * 80)

    # Scenarios setup
    scenarios = {
        "Normal (1.0x Load)": {"load_mult": 1.0, "num_failed_links": 0},
        "High Congestion (1.5x)": {"load_mult": 1.5, "num_failed_links": 0},
        "High Congestion (2.0x)": {"load_mult": 2.0, "num_failed_links": 0},
        "Link Failures (2 Links)": {"load_mult": 1.0, "num_failed_links": 2},
        "Combined Stress (1.8x + 2 Links)": {"load_mult": 1.8, "num_failed_links": 2},
    }

    # Methods list
    methods = [
        "Static (stale paths)",
        "Static Dijkstra",
        "Greedy Congestion",
        "Simulated Annealing",
        "SB FP32",
        "SB FP16",
        "SB FP16 + Polish"
    ]

    # Held-out evaluation seeds for benchmark
    eval_seeds = list(range(200, 210))

    benchmark_results = {
        "_meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": eval_seeds,
            "solver_settings": {
                "num_nodes": 20,
                "num_demands": 15,
                "k_paths": 4,
                "A": 20.0,
                "B": 20.0,
                "C": 1.0
            }
        }
    }

    total_start_time = time.time()

    for sc_name, sc_params in scenarios.items():
        print(f"\n Running Scenario: {sc_name}...")
        load_mult = sc_params["load_mult"]
        num_failed_links = sc_params["num_failed_links"]

        sc_method_raw = {m: {
            "avg_latency": [],
            "max_latency": [],
            "max_utilisation": [],
            "overloaded_links": [],
            "flows_on_failed_links": [],
            "flows_on_overloaded_links": [],
            "raw_feasibility": [],
            "runtime_ms": []
        } for m in methods}

        # Run benchmark on held-out evaluation seeds
        for seed in eval_seeds:
            # 1. Full network graph
            G_full = create_random_network(num_nodes=20, radius=0.35, seed=seed)
            stale_demands, stale_candidate_paths = generate_demands(G_full, num_demands=15, seed=seed, k=4)

            # 2. Select failed links maintaining graph connectivity
            G_surviving = G_full.copy()
            disabled_links = []
            if num_failed_links > 0:
                rng = np.random.RandomState(seed)
                edges_list = list(G_full.edges())
                chosen_idx = rng.choice(len(edges_list), size=len(edges_list), replace=False)
                for idx in chosen_idx:
                    if len(disabled_links) >= num_failed_links:
                        break
                    u, v = edges_list[idx]
                    edge = (min(u, v), max(u, v))
                    G_temp = G_surviving.copy()
                    G_temp.remove_edge(u, v)
                    if nx.is_connected(G_temp):
                        G_surviving.remove_edge(u, v)
                        disabled_links.append(edge)

            # 3. Generate reactive demands & candidate paths ONCE per seed for all methods
            demands, candidate_paths = generate_demands(G_surviving, num_demands=15, seed=seed, k=4)
            num_demands = len(demands)
            k = 4

            # Build QUBO and Ising matrices for reactive graph
            Q, meta = build_qubo_matrix(G_surviving, demands, candidate_paths, A=20.0, B=20.0, C=1.0, load_mult=load_mult)
            J, h, offset = qubo_to_ising(Q)

            # --- Method 0: Static (stale paths) ---
            t0 = time.time()
            sel_stale = solve_static_dijkstra(len(stale_demands))
            t_stale = (time.time() - t0) * 1000.0
            res_stale = evaluate(G_full, stale_demands, stale_candidate_paths, sel_stale, load_mult, disabled_links)

            sc_method_raw["Static (stale paths)"]["avg_latency"].append(res_stale["avg_latency"])
            sc_method_raw["Static (stale paths)"]["max_latency"].append(res_stale["max_latency"])
            sc_method_raw["Static (stale paths)"]["max_utilisation"].append(res_stale["max_utilisation"])
            sc_method_raw["Static (stale paths)"]["overloaded_links"].append(res_stale["overloaded_links"])
            sc_method_raw["Static (stale paths)"]["flows_on_failed_links"].append(res_stale["flows_on_failed_links"])
            sc_method_raw["Static (stale paths)"]["flows_on_overloaded_links"].append(res_stale["flows_on_overloaded_links"])
            sc_method_raw["Static (stale paths)"]["runtime_ms"].append(t_stale)

            # --- Method 1: Static Dijkstra (Reactive on G_surviving) ---
            t0 = time.time()
            sel_dijkstra = solve_static_dijkstra(num_demands)
            t_dijkstra = (time.time() - t0) * 1000.0
            res_dijkstra = evaluate(G_surviving, demands, candidate_paths, sel_dijkstra, load_mult, disabled_links)

            sc_method_raw["Static Dijkstra"]["avg_latency"].append(res_dijkstra["avg_latency"])
            sc_method_raw["Static Dijkstra"]["max_latency"].append(res_dijkstra["max_latency"])
            sc_method_raw["Static Dijkstra"]["max_utilisation"].append(res_dijkstra["max_utilisation"])
            sc_method_raw["Static Dijkstra"]["overloaded_links"].append(res_dijkstra["overloaded_links"])
            sc_method_raw["Static Dijkstra"]["flows_on_failed_links"].append(res_dijkstra["flows_on_failed_links"])
            sc_method_raw["Static Dijkstra"]["flows_on_overloaded_links"].append(res_dijkstra["flows_on_overloaded_links"])
            sc_method_raw["Static Dijkstra"]["runtime_ms"].append(t_dijkstra)

            # --- Method 2: Greedy Congestion ---
            t0 = time.time()
            sel_greedy = solve_greedy_congestion(G_surviving, demands, candidate_paths, k=k, load_mult=load_mult)
            t_greedy = (time.time() - t0) * 1000.0
            res_greedy = evaluate(G_surviving, demands, candidate_paths, sel_greedy, load_mult, disabled_links)

            sc_method_raw["Greedy Congestion"]["avg_latency"].append(res_greedy["avg_latency"])
            sc_method_raw["Greedy Congestion"]["max_latency"].append(res_greedy["max_latency"])
            sc_method_raw["Greedy Congestion"]["max_utilisation"].append(res_greedy["max_utilisation"])
            sc_method_raw["Greedy Congestion"]["overloaded_links"].append(res_greedy["overloaded_links"])
            sc_method_raw["Greedy Congestion"]["flows_on_failed_links"].append(res_greedy["flows_on_failed_links"])
            sc_method_raw["Greedy Congestion"]["flows_on_overloaded_links"].append(res_greedy["flows_on_overloaded_links"])
            sc_method_raw["Greedy Congestion"]["runtime_ms"].append(t_greedy)

            # --- Method 3: Simulated Annealing ---
            t0 = time.time()
            sel_sa = solve_simulated_annealing(Q, num_demands, k=k, steps=500, seed=seed)
            t_sa = (time.time() - t0) * 1000.0
            res_sa = evaluate(G_surviving, demands, candidate_paths, sel_sa, load_mult, disabled_links)

            sc_method_raw["Simulated Annealing"]["avg_latency"].append(res_sa["avg_latency"])
            sc_method_raw["Simulated Annealing"]["max_latency"].append(res_sa["max_latency"])
            sc_method_raw["Simulated Annealing"]["max_utilisation"].append(res_sa["max_utilisation"])
            sc_method_raw["Simulated Annealing"]["overloaded_links"].append(res_sa["overloaded_links"])
            sc_method_raw["Simulated Annealing"]["flows_on_failed_links"].append(res_sa["flows_on_failed_links"])
            sc_method_raw["Simulated Annealing"]["flows_on_overloaded_links"].append(res_sa["flows_on_overloaded_links"])
            sc_method_raw["Simulated Annealing"]["runtime_ms"].append(t_sa)

            # --- Method 4: SB FP32 ---
            t0 = time.time()
            best_s_fp32, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float32, seed=seed)
            sel_fp32, raw_feas_fp32 = decode_with_repair(best_s_fp32, Q, num_demands, k)
            t_fp32 = (time.time() - t0) * 1000.0
            res_fp32 = evaluate(G_surviving, demands, candidate_paths, sel_fp32, load_mult, disabled_links)

            sc_method_raw["SB FP32"]["avg_latency"].append(res_fp32["avg_latency"])
            sc_method_raw["SB FP32"]["max_latency"].append(res_fp32["max_latency"])
            sc_method_raw["SB FP32"]["max_utilisation"].append(res_fp32["max_utilisation"])
            sc_method_raw["SB FP32"]["overloaded_links"].append(res_fp32["overloaded_links"])
            sc_method_raw["SB FP32"]["flows_on_failed_links"].append(res_fp32["flows_on_failed_links"])
            sc_method_raw["SB FP32"]["flows_on_overloaded_links"].append(res_fp32["flows_on_overloaded_links"])
            sc_method_raw["SB FP32"]["raw_feasibility"].append(raw_feas_fp32)
            sc_method_raw["SB FP32"]["runtime_ms"].append(t_fp32)

            # --- Method 5: SB FP16 ---
            t0 = time.time()
            best_s_fp16, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float16, seed=seed)
            sel_fp16, raw_feas_fp16 = decode_with_repair(best_s_fp16, Q, num_demands, k)
            t_fp16 = (time.time() - t0) * 1000.0
            res_fp16 = evaluate(G_surviving, demands, candidate_paths, sel_fp16, load_mult, disabled_links)

            sc_method_raw["SB FP16"]["avg_latency"].append(res_fp16["avg_latency"])
            sc_method_raw["SB FP16"]["max_latency"].append(res_fp16["max_latency"])
            sc_method_raw["SB FP16"]["max_utilisation"].append(res_fp16["max_utilisation"])
            sc_method_raw["SB FP16"]["overloaded_links"].append(res_fp16["overloaded_links"])
            sc_method_raw["SB FP16"]["flows_on_failed_links"].append(res_fp16["flows_on_failed_links"])
            sc_method_raw["SB FP16"]["flows_on_overloaded_links"].append(res_fp16["flows_on_overloaded_links"])
            sc_method_raw["SB FP16"]["raw_feasibility"].append(raw_feas_fp16)
            sc_method_raw["SB FP16"]["runtime_ms"].append(t_fp16)

            # --- Method 6: SB FP16 + Polish ---
            t0 = time.time()
            best_s_fp16_p, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float16, seed=seed)
            sel_fp16_unp, raw_feas_fp16_p = decode_with_repair(best_s_fp16_p, Q, num_demands, k)
            sel_fp16_polished = feasible_coordinate_descent_polish(sel_fp16_unp, Q, num_demands, k)
            t_fp16_polish = (time.time() - t0) * 1000.0
            res_fp16_polish = evaluate(G_surviving, demands, candidate_paths, sel_fp16_polished, load_mult, disabled_links)

            sc_method_raw["SB FP16 + Polish"]["avg_latency"].append(res_fp16_polish["avg_latency"])
            sc_method_raw["SB FP16 + Polish"]["max_latency"].append(res_fp16_polish["max_latency"])
            sc_method_raw["SB FP16 + Polish"]["max_utilisation"].append(res_fp16_polish["max_utilisation"])
            sc_method_raw["SB FP16 + Polish"]["overloaded_links"].append(res_fp16_polish["overloaded_links"])
            sc_method_raw["SB FP16 + Polish"]["flows_on_failed_links"].append(res_fp16_polish["flows_on_failed_links"])
            sc_method_raw["SB FP16 + Polish"]["flows_on_overloaded_links"].append(res_fp16_polish["flows_on_overloaded_links"])
            sc_method_raw["SB FP16 + Polish"]["raw_feasibility"].append(raw_feas_fp16_p)
            sc_method_raw["SB FP16 + Polish"]["runtime_ms"].append(t_fp16_polish)

        # Aggregate averages, std, and 95% CIs per scenario & method
        sc_summary = {}
        greedy_raw = sc_method_raw["Greedy Congestion"]

        for m in methods:
            m_raw = sc_method_raw[m]
            raw_feas_list = m_raw["raw_feasibility"]
            raw_feas_val = float(np.mean(raw_feas_list)) if len(raw_feas_list) > 0 else None

            # Compute stats for key metrics
            stat_avg_lat = compute_stats(m_raw["avg_latency"])
            stat_max_util = compute_stats(m_raw["max_utilisation"])
            stat_overloads = compute_stats(m_raw["flows_on_overloaded_links"])

            # Compute paired comparison against Greedy for QUBO / alternative methods
            paired_lat = compute_paired_comparison(m_raw["avg_latency"], greedy_raw["avg_latency"], lower_is_better=True)
            paired_overloads = compute_paired_comparison(m_raw["flows_on_overloaded_links"], greedy_raw["flows_on_overloaded_links"], lower_is_better=True)

            sc_summary[m] = {
                "per_seed_raw": m_raw,
                "avg_latency": float(stat_avg_lat["mean"]),
                "avg_latency_stats": stat_avg_lat,
                "max_latency": float(np.mean(m_raw["max_latency"])),
                "max_utilisation": float(stat_max_util["mean"]),
                "max_utilisation_stats": stat_max_util,
                "overloaded_links": float(np.mean(m_raw["overloaded_links"])),
                "flows_on_failed_links": float(np.mean(m_raw["flows_on_failed_links"])),
                "flows_on_overloaded_links": float(stat_overloads["mean"]),
                "flows_on_overloaded_links_stats": stat_overloads,
                "raw_feasibility": raw_feas_val,
                "runtime_ms": float(np.mean(m_raw["runtime_ms"])),
                "paired_vs_greedy": {
                    "avg_latency": paired_lat,
                    "flows_on_overloaded_links": paired_overloads
                }
            }

        benchmark_results[sc_name] = sc_summary

        # Print Markdown table for scenario
        print(f"\n--- Results Summary: {sc_name} ---")
        print(f"| {'Method':<22} | {'Avg Lat (Mean±Std)':<24} | {'95% CI Latency':<20} | {'Overloaded Flows':<16} | {'Paired vs Greedy Latency':<28} | {'Conclusion':<12} |")
        print("|" + "-"*24 + "|" + "-"*26 + "|" + "-"*22 + "|" + "-"*18 + "|" + "-"*30 + "|" + "-"*14 + "|")
        for m in methods:
            s = sc_summary[m]
            lat_st = s["avg_latency_stats"]
            lat_str = f"{lat_st['mean']:.2f} ± {lat_st['std']:.2f}"
            ci_str = f"[{lat_st['ci95'][0]:.2f}, {lat_st['ci95'][1]:.2f}]"
            p_comp = s["paired_vs_greedy"]["avg_latency"]
            p_str = f"Diff: {p_comp['mean_diff']:+.2f} (W:{p_comp['wins']}/T:{p_comp['ties']}/L:{p_comp['losses']})"
            conc = p_comp["conclusion"] if m != "Greedy Congestion" else "baseline"
            print(f"| {m:<22} | {lat_str:<24} | {ci_str:<20} | {s['flows_on_overloaded_links']:<16.1f} | {p_str:<28} | {conc:<12} |")

    total_elapsed = time.time() - total_start_time
    print(f"\n [SUCCESS] Statistical benchmark finished in {total_elapsed:.2f} seconds.")

    # Save output to results.json
    with open("results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(" Saved statistical benchmark results with _meta and per-seed metrics to results.json")

    return benchmark_results

if __name__ == "__main__":
    run_benchmark()
