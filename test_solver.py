from dotenv import load_dotenv
load_dotenv()

import os
import pytest
import numpy as np
import torch

from network import create_random_network, generate_demands, compute_link_loads
from qubo import (
    build_qubo_matrix,
    qubo_to_ising,
    compute_qubo_energy,
    compute_ising_energy,
    decode_with_repair
)
from sb_module import solve_simulated_bifurcation, SimulatedBifurcation

def test_qubo_energy_matches_true_cost():
    """
    Unit Test 1: Verifies QUBO matrix energy x^T Q x equals true network penalty cost formula.
    """
    G = create_random_network(num_nodes=15, seed=123)
    demands, candidate_paths = generate_demands(G, num_demands=10, seed=123, k=4)
    num_demands = len(demands)
    k = 4
    
    A, B, C = 20.0, 20.0, 1.0
    Q, meta = build_qubo_matrix(G, demands, candidate_paths, A=A, B=B, C=C)
    
    # Valid selection vector (1 path per demand)
    selection = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    
    # Construct binary vector x
    x = np.zeros(num_demands * k, dtype=np.float64)
    for d_idx, p_idx in enumerate(selection):
        x[d_idx * k + p_idx] = 1.0
        
    qubo_energy = compute_qubo_energy(x, Q)
    
    # Manual true cost calculation:
    # 1. Base Latency
    base_lat_sum = 0.0
    for d_idx, p_idx in enumerate(selection):
        base_lat_sum += sum(G[u][v]['base_latency'] for u, v in candidate_paths[d_idx][p_idx])
        
    # 2. One-hot penalty: for valid selection, sum_p x - 1 = 0, so penalty is 0 (diagonal offset -A * N_demands)
    # Note that in QUBO diagonal -A * x_i + A * x_i x_j sums to 0 when exactly 1 active per demand.
    
    # 3. Congestion penalty
    link_loads = compute_link_loads(G, demands, candidate_paths, selection)
    cong_penalty = 0.0
    for u, v, data in G.edges(data=True):
        edge = (min(u, v), max(u, v))
        cap = data['capacity']
        base_lat = data['base_latency']
        load = link_loads[edge]
        util = load / cap if cap > 0 else 0.0
        cong_penalty += B * (util ** 2) * np.sqrt(base_lat)
        
    # 4. Failure probability penalty
    fail_penalty = 0.0
    for d_idx, p_idx in enumerate(selection):
        succ_p = 1.0
        for u, v in candidate_paths[d_idx][p_idx]:
            succ_p *= (1.0 - G[u][v]['failure_prob'])
        fail_penalty += C * (1.0 - succ_p)
        
    # One-hot penalty constant offset: x^T Q_onehot x = -A for valid selection (differs from 0 by -A * num_demands)
    expected_cost = base_lat_sum + cong_penalty + fail_penalty - (A * num_demands)
    
    assert abs(qubo_energy - expected_cost) < 1e-4, f"QUBO energy {qubo_energy} != expected cost {expected_cost}"

def test_ising_conversion_energy_matches_qubo():
    """
    Unit Test 2: Verifies Ising Hamiltonian energy E(s) matches QUBO energy x^T Q x.
    """
    G = create_random_network(num_nodes=15, seed=456)
    demands, candidate_paths = generate_demands(G, num_demands=8, seed=456, k=4)
    Q, meta = build_qubo_matrix(G, demands, candidate_paths)
    J, h, offset = qubo_to_ising(Q)
    
    # Random binary vector x
    rng = np.random.RandomState(456)
    x = rng.choice([0, 1], size=meta['N']).astype(np.float64)
    s = 2.0 * x - 1.0
    
    qubo_e = compute_qubo_energy(x, Q)
    ising_e = compute_ising_energy(s, J, h, offset)
    
    assert abs(qubo_e - ising_e) < 1e-6, f"Ising energy {ising_e} does not match QUBO energy {qubo_e}"

def test_sb_output_shape_and_values():
    """
    Unit Test 3: Verifies PyTorch SB solver outputs shape (128, 128) with spin values in {-1, 1}.
    """
    steps = 50
    module = SimulatedBifurcation(steps=steps)
    module.eval()
    
    x_init = 0.1 * torch.randn(128, 128)
    y_init = torch.zeros(128, 128)
    J = torch.randn(128, 128)
    J = 0.5 * (J + J.T)
    h = torch.randn(128)
    c0 = torch.tensor(0.5)
    
    with torch.no_grad():
        out_x = module(x_init, y_init, J, h, c0)
        
    assert out_x.shape == (128, 128), f"Expected shape (128, 128), got {out_x.shape}"
    
    spins = torch.sign(out_x)
    unique_vals = torch.unique(spins)
    for val in unique_vals:
        assert val.item() in [-1.0, 1.0], f"Unexpected spin value: {val.item()}"

def test_decode_repair_always_yields_one_path_per_demand():
    """
    Unit Test 4: Verifies decode_with_repair always outputs exactly one valid path index per demand.
    """
    G = create_random_network(num_nodes=15, seed=789)
    demands, candidate_paths = generate_demands(G, num_demands=12, seed=789, k=4)
    num_demands = len(demands)
    k = 4
    Q, meta = build_qubo_matrix(G, demands, candidate_paths)
    
    # Generate 50 completely random/invalid spin vectors
    rng = np.random.RandomState(789)
    for test_i in range(50):
        raw_spins = rng.choice([-1.0, 1.0], size=meta['N'])
        selection, raw_feas = decode_with_repair(raw_spins, Q, num_demands, k)
        
        assert len(selection) == num_demands, f"Selection length {len(selection)} != num_demands {num_demands}"
        for d_idx, p_idx in enumerate(selection):
            assert 0 <= p_idx < k, f"Invalid path index {p_idx} for demand {d_idx}"

def test_null_npu_profile_metrics():
    """
    Unit Test 5: Verifies that when source is 'none' or 'cirrascale_models_endpoint', hardware NPU metrics are null.
    """
    import json
    import os
    if os.path.exists("npu_profile.json"):
        with open("npu_profile.json", "r") as f:
            data = json.load(f)
        if data.get("source") in ["none", "cirrascale_models_endpoint"]:
            assert data.get("latency_ms") is None, "latency_ms must be None when not from real profiler job"
            assert data.get("peak_memory_mb") is None, "peak_memory_mb must be None when not from real profiler job"
            assert data.get("npu_coverage_pct") is None, "npu_coverage_pct must be None when not from real profiler job"

def test_parse_scenario_valid_json(mocker=None):
    """
    Unit Test 6: Verifies parse_scenario with valid JSON output from LLM (Mocked API).
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"load_scale": 2.0, "n_demands": 20, "fail_links": [[0, 1]], "seed": 42}'
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("Double load, 20 demands, break link 0-1", G=G)

    assert sc["load_scale"] == 2.0
    assert sc["n_demands"] == 20
    assert sc["fail_links"] == [[0, 1]]
    assert "Hosted LLM" in sc["parsed_by"]

def test_parse_scenario_code_fences():
    """
    Unit Test 7: Verifies parse_scenario handles JSON enclosed in markdown code fences.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"load_scale": 1.5, "n_demands": 12, "fail_links": []}\n```'
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("1.5x load", G=G)

    assert sc["load_scale"] == 1.5
    assert sc["n_demands"] == 12

def test_parse_scenario_think_blocks():
    """
    Unit Test 8: Verifies parse_scenario strips <think>...</think> reasoning blocks before parsing.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '<think>\nAnalyzing the user prompt...\nLoad multiplier is 2.5.\n</think>```json\n{"load_scale": 2.5, "n_demands": 18, "fail_links": []}\n```'
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("2.5x load", G=G)

    assert sc["load_scale"] == 2.5
    assert sc["n_demands"] == 18

def test_parse_scenario_garbage_output():
    """
    Unit Test 9: Verifies parse_scenario falls back gracefully on unparseable LLM output.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = 'I cannot process this request into JSON format.'
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("double load", G=G)

    assert sc["parsed_by"].startswith("keyword_fallback")
    assert sc["load_scale"] == 2.0

def test_parse_scenario_out_of_range_values():
    """
    Unit Test 10: Verifies parse_scenario rejects out-of-range values instead of silently clamping them.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=10.0; fail_links=none; n_demands=100"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("extreme values", G=G)

    assert "keyword_fallback" in sc["parsed_by"]
    assert sc["load_scale"] != 10.0

def test_parse_scenario_nonexistent_fail_links():
    """
    Unit Test 11: Verifies parse_scenario places links not in graph G into dropped_links.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    # 99-100 is guaranteed not to exist in a 10-node graph
    mock_choice.message.content = '{"load_scale": 1.0, "n_demands": 15, "fail_links": [[99, 100]]}'
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, lat, msg = parse_scenario("fail non-existent link", G=G)

    assert sc["fail_links"] == []
    assert [99, 100] in sc["dropped_links"]

def test_parse_scenario_missing_api_key():
    """
    Unit Test 12: Verifies parse_scenario uses keyword fallback parser when API key is missing.
    """
    from unittest.mock import patch
    from network import create_random_network
    from llm_assistant import parse_scenario

    G = create_random_network(num_nodes=10, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(None, "")):
        sc, lat, msg = parse_scenario("double load", G=G)

    assert sc["parsed_by"].startswith("keyword_fallback")
    assert sc["load_scale"] == 2.0

def test_forbidden_claims_in_readme_and_app():
    """
    Unit Test 13: Scans README.md and app.py to verify no forbidden claims exist.
    Forbidden strings: "100% NPU", "deployment verified", "solver runs on Cloud AI 100", "quantum speedup".
    Fails if any forbidden phrase is found.
    """
    import os

    forbidden_phrases = [
        "100% NPU",
        "deployment verified",
        "solver runs on Cloud AI 100",
        "quantum speedup"
    ]

    files_to_check = ["README.md", "app.py"]

    for file_path in files_to_check:
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        for phrase in forbidden_phrases:
            assert phrase.lower() not in content, (
                f"FORBIDDEN CLAIM DETECTED in {file_path}: '{phrase}'! "
                "Remove or rephrase forbidden claim to maintain technical honesty."
            )

def test_parse_scenario_exact_string_mocked_cases():
    """
    Unit Test: Tests exact user string "Double traffic load and break link between nodes 3 and 7" against:
    (a) clean line, (b) line wrapped in prose, (c) fail_links=3-7,none, (d) empty content plus tool_calls.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    exact_prompt = "Double traffic load and break link between nodes 3 and 7"
    G = create_random_network(num_nodes=20, seed=42)

    # (a) Clean line
    mock_client_a = MagicMock()
    mock_choice_a = MagicMock()
    mock_choice_a.finish_reason = "stop"
    mock_choice_a.message.content = "load_scale=2; fail_links=3-7; n_demands=default"
    mock_choice_a.message.tool_calls = None
    mock_client_a.chat.completions.create.return_value.choices = [mock_choice_a]

    with patch('llm_assistant.get_openai_client', return_value=(mock_client_a, "https://mock-endpoint")):
        sc_a, _, msg_a = parse_scenario(exact_prompt, G=G)
    assert sc_a["load_scale"] == 2.0
    assert [3, 7] in sc_a["dropped_links"] or [3, 7] in sc_a["fail_links"]

    # (b) Wrapped in prose
    mock_client_b = MagicMock()
    mock_choice_b = MagicMock()
    mock_choice_b.finish_reason = "stop"
    mock_choice_b.message.content = "<think>Analyzing...</think>Output line: load_scale=2; fail_links=3-7; n_demands=default"
    mock_choice_b.message.tool_calls = None
    mock_client_b.chat.completions.create.return_value.choices = [mock_choice_b]

    with patch('llm_assistant.get_openai_client', return_value=(mock_client_b, "https://mock-endpoint")):
        sc_b, _, _ = parse_scenario(exact_prompt, G=G)
    assert sc_b["load_scale"] == 2.0

    # (c) Stray 'none' in fail_links=3-7,none
    mock_client_c = MagicMock()
    mock_choice_c = MagicMock()
    mock_choice_c.finish_reason = "stop"
    mock_choice_c.message.content = "load_scale=2; fail_links=3-7,none; n_demands=default"
    mock_choice_c.message.tool_calls = None
    mock_client_c.chat.completions.create.return_value.choices = [mock_choice_c]

    with patch('llm_assistant.get_openai_client', return_value=(mock_client_c, "https://mock-endpoint")):
        sc_c, _, _ = parse_scenario(exact_prompt, G=G)
    assert sc_c["load_scale"] == 2.0

    # (d) Empty content plus tool call (finish_reason='tool_calls')
    mock_client_d = MagicMock()
    mock_choice_d = MagicMock()
    mock_choice_d.finish_reason = "tool_calls"
    mock_choice_d.message.content = ""
    mock_choice_d.message.tool_calls = [{"name": "", "arguments": "{}"}]
    mock_client_d.chat.completions.create.return_value.choices = [mock_choice_d]

    with patch('llm_assistant.get_openai_client', return_value=(mock_client_d, "https://mock-endpoint")):
        sc_d, _, msg_d = parse_scenario(exact_prompt, G=G)
    assert "keyword_fallback" in sc_d["parsed_by"]
    assert "tool_calls" in msg_d or "LLM parse failed" in msg_d

def test_parse_scenario_seed_42_invalid_link_3_7():
    """
    Unit Test: Verifies Seed 42 graph (20 nodes) has no edge (3, 7), node 3's neighbors are [0, 9, 12],
    and parsing fail_links=3-7 reports [3, 7] as an invalid link in dropped_links.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    G = create_random_network(num_nodes=20, radius=0.35, seed=42)
    neighbors_3 = sorted(list(G.neighbors(3)))
    assert neighbors_3 == [0, 9, 12], f"Expected neighbors [0, 9, 12], got {neighbors_3}"
    assert not G.has_edge(3, 7), "Edge (3, 7) must not exist in seed 42 graph"

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=2.0; fail_links=3-7; n_demands=15"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, _, msg = parse_scenario("break link 3-7", G=G)

    assert sc["fail_links"] == [], "Link (3, 7) must not be in valid fail_links"
    assert [3, 7] in sc["dropped_links"], "Link (3, 7) must be reported in dropped_links as invalid"

import pytest

@pytest.mark.skipif(not os.getenv("INFERENCE_CLOUD_API_KEY"), reason="INFERENCE_CLOUD_API_KEY not set")
def test_live_parse_scenario_plain_prompt():
    """
    Live Integration Test: Sends plain-text prompt to hosted Llama-3.1-8B endpoint.
    Skipped if no API key is present.
    """
    from network import create_random_network
    from llm_assistant import parse_scenario

    G = create_random_network(num_nodes=20, seed=42)
    prompt1 = "Double traffic load and break link between nodes 3 and 7"
    sc1, lat1, msg1 = parse_scenario(prompt1, G=G)

    assert sc1["load_scale"] == 2.0
    assert lat1 > 0.0
    assert "Hosted LLM" in sc1["parsed_by"]

    prompt2 = "Increase load by 50% and fail links 0-3 and 9-3"
    sc2, lat2, msg2 = parse_scenario(prompt2, G=G)

    assert sc2["load_scale"] == 1.5
    assert [0, 3] in sc2["fail_links"]
    assert [3, 9] in sc2["fail_links"]
    assert "Hosted LLM" in sc2["parsed_by"]


def test_rerouting_methods_never_use_failed_links():
    """
    Unit Test 14: Verifies that no reactive solver method's chosen path uses a failed link on the surviving graph.
    """
    import networkx as nx
    from network import create_random_network, generate_demands, evaluate
    from qubo import build_qubo_matrix, qubo_to_ising, decode_with_repair
    from baselines import solve_static_dijkstra, solve_greedy_congestion, solve_simulated_annealing
    from sb_module import solve_simulated_bifurcation
    import torch

    seed = 200
    G_full = create_random_network(num_nodes=20, radius=0.35, seed=seed)
    G_surviving = G_full.copy()
    disabled_links = []
    edges_list = list(G_full.edges())

    for u, v in edges_list:
        if len(disabled_links) >= 2:
            break
        edge = (min(u, v), max(u, v))
        G_temp = G_surviving.copy()
        G_temp.remove_edge(u, v)
        if nx.is_connected(G_temp):
            G_surviving.remove_edge(u, v)
            disabled_links.append(edge)

    demands, candidate_paths = generate_demands(G_surviving, num_demands=15, seed=seed, k=4)
    Q, meta = build_qubo_matrix(G_surviving, demands, candidate_paths)
    J, h, offset = qubo_to_ising(Q)

    # 1. Dijkstra on G_surviving
    sel_d = solve_static_dijkstra(len(demands))
    res_d = evaluate(G_surviving, demands, candidate_paths, sel_d, disabled_links=disabled_links)
    assert res_d["flows_on_failed_links"] == 0, f"Dijkstra used failed link: {res_d['flows_on_failed_links']}"

    # 2. Greedy on G_surviving
    sel_g = solve_greedy_congestion(G_surviving, demands, candidate_paths, k=4)
    res_g = evaluate(G_surviving, demands, candidate_paths, sel_g, disabled_links=disabled_links)
    assert res_g["flows_on_failed_links"] == 0, f"Greedy used failed link: {res_g['flows_on_failed_links']}"

    # 3. SA on G_surviving
    sel_sa = solve_simulated_annealing(Q, len(demands), k=4, seed=seed)
    res_sa = evaluate(G_surviving, demands, candidate_paths, sel_sa, disabled_links=disabled_links)
    assert res_sa["flows_on_failed_links"] == 0, f"SA used failed link: {res_sa['flows_on_failed_links']}"

    # 4. SB on G_surviving
    best_s, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float32, seed=seed)
    sel_sb, _ = decode_with_repair(best_s, Q, len(demands), 4)
    res_sb = evaluate(G_surviving, demands, candidate_paths, sel_sb, disabled_links=disabled_links)
    assert res_sb["flows_on_failed_links"] == 0, f"SB used failed link: {res_sb['flows_on_failed_links']}"

def test_stale_path_baseline_detects_failed_links():
    """
    Unit Test 15: Verifies that stale-path baseline has flows_on_failed_links > 0 when a failed link lies on its path.
    """
    import networkx as nx
    from network import create_random_network, generate_demands, evaluate
    from baselines import solve_static_dijkstra

    seed = 200
    G_full = create_random_network(num_nodes=20, radius=0.35, seed=seed)
    demands, candidate_paths = generate_demands(G_full, num_demands=15, seed=seed, k=4)
    sel_stale = solve_static_dijkstra(len(demands))

    # Pick an edge that lies on candidate path 0 of demand 0
    first_path_edges = candidate_paths[0][0]
    assert len(first_path_edges) > 0
    failed_edge = first_path_edges[0]

    res_stale = evaluate(G_full, demands, candidate_paths, sel_stale, disabled_links=[failed_edge])
    assert res_stale["flows_on_failed_links"] > 0, "Stale path baseline failed to detect flow on failed link"

def test_zero_failure_scenarios_have_zero_failed_link_flows():
    """
    Unit Test 16: Verifies zero-failure scenarios have flows_on_failed_links == 0 for all methods.
    """
    from network import create_random_network, generate_demands, evaluate
    from baselines import solve_static_dijkstra, solve_greedy_congestion

    seed = 42
    G = create_random_network(num_nodes=15, seed=seed)
    demands, candidate_paths = generate_demands(G, num_demands=10, seed=seed, k=4)

    sel_d = solve_static_dijkstra(len(demands))
    res_d = evaluate(G, demands, candidate_paths, sel_d, disabled_links=[])
    assert res_d["flows_on_failed_links"] == 0

    sel_g = solve_greedy_congestion(G, demands, candidate_paths, k=4)
    res_g = evaluate(G, demands, candidate_paths, sel_g, disabled_links=[])
    assert res_g["flows_on_failed_links"] == 0

def test_link_failures_scenario_metrics_differ_from_normal():
    """
    Unit Test 17: Verifies Link Failures scenario metrics differ from Normal scenario metrics.
    """
    import json
    import os

    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            bench_data = json.load(f)

        if "Normal (1.0x Load)" in bench_data and "Link Failures (2 Links)" in bench_data:
            norm_stale = bench_data["Normal (1.0x Load)"]["Static (stale paths)"]
            fail_stale = bench_data["Link Failures (2 Links)"]["Static (stale paths)"]

            assert norm_stale["flows_on_failed_links"] == 0.0, "Normal scenario must have 0 flows on failed links"
            assert fail_stale["flows_on_failed_links"] > 0.0, "Link Failures scenario must have > 0 flows on failed links for stale paths"

def test_rejection_bad_load_scale():
    """
    Unit Test: Mocked model line load_scale=50; fail_links=none is rejected,
    parsed_by contains the reason, and result is NOT 4.0.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=50; fail_links=none; n_demands=default"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=20, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, _, msg = parse_scenario("Double load", G=G)

    assert sc["load_scale"] != 4.0, "Must NOT silently clamp 50 to 4.0"
    assert "keyword_fallback" in sc["parsed_by"]
    assert "outside 0.25-4.0" in sc["parsed_by"] or "load_scale=50" in sc["parsed_by"]
    assert "outside 0.25-4.0" in msg or "load_scale=50" in msg

def test_rejection_bad_n_demands():
    """
    Unit Test: Mocked n_demands=2 is rejected.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=1.5; fail_links=none; n_demands=2"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=20, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, _, msg = parse_scenario("1.5x load with 2 demands", G=G)

    assert "keyword_fallback" in sc["parsed_by"]
    assert "n_demands=2" in sc["parsed_by"] or "outside 4-40" in sc["parsed_by"]

def test_double_traffic_load_n_demands_none():
    """
    Unit Test: "Double traffic load" returns n_demands None via mocked-LLM path AND via fallback.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario, parse_scenario_fallback

    # 1. Fallback path
    sc_fb = parse_scenario_fallback("Double traffic load")
    assert sc_fb["n_demands"] is None, "Fallback must return n_demands=None when no count in prompt"
    assert sc_fb["load_scale"] == 2.0

    # 2. Mocked-LLM path returning n_demands=default
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=2.0; fail_links=none; n_demands=default"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=20, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc_llm, _, _ = parse_scenario("Double traffic load", G=G)

    assert sc_llm["n_demands"] is None, "Mocked LLM path must return n_demands=None for default"
    assert sc_llm["load_scale"] == 2.0

def test_triple_load_25_demands():
    """
    Unit Test: "Triple the load with 25 demands" returns 25.
    """
    from llm_assistant import parse_scenario_fallback
    sc = parse_scenario_fallback("Triple the load with 25 demands")
    assert sc["load_scale"] == 3.0
    assert sc["n_demands"] == 25

def test_percentage_50_percent_fallback():
    """
    Unit Test: "Increase load by 50%" returns load_scale 1.5 via fallback.
    """
    from llm_assistant import parse_scenario_fallback
    sc = parse_scenario_fallback("Increase load by 50%")
    assert sc["load_scale"] == 1.5
    assert sc["n_demands"] is None

def test_extract_link_pairs_negative_and_positive():
    """
    Unit Test: extract_link_pairs returns [] for negative phrases and finds pairs for positive phrases.
    """
    from llm_assistant import extract_link_pairs

    negative_texts = [
        "Increase load by 2 to 3 times",
        "Use 5 to 10 demands",
        "Double load, 4 and 5 demands",
        "Triple the load with 25 demands"
    ]
    for text in negative_texts:
        res = extract_link_pairs(text)
        assert res == [], f"Expected [] for '{text}', got {res}"

    assert extract_link_pairs("fail links 3-9 and 0-3") == [[3, 9], [0, 3]]
    assert extract_link_pairs("break link between nodes 3 and 9") == [[3, 9]]
    assert extract_link_pairs("3 to 9") == [[3, 9]]
    assert extract_link_pairs("3/9") == [[3, 9]]
    assert extract_link_pairs("link 3 9") == [[3, 9]]

def test_mocked_tool_calls_failure():
    """
    Unit Test: Mocked empty-content + tool_calls case with finish_reason='tool_calls'
    AND a tool call with name='' and arguments='{}'.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_tool_call = MagicMock()
    mock_tool_call.name = ""
    mock_tool_call.arguments = "{}"

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=20, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, _, msg = parse_scenario("Double load and break link 3-9", G=G)

    assert "keyword_fallback" in sc["parsed_by"]
    assert "tool_calls" in sc["parsed_by"]
    assert "tool_calls" in msg or "LLM parse failed" in msg


def test_phrasings_exact_fail_and_dropped_links_seed_42():
    """
    Unit Test: Tests exact phrasings on seed-42 graph (offline, mocked API/fallback).
    """
    from network import create_random_network
    from llm_assistant import parse_scenario_fallback

    G = create_random_network(num_nodes=20, seed=42)

    phrasing_cases = [
        ("0-3 and 9-3", [[0, 3], [3, 9]], []),
        ("9-3", [[3, 9]], []),
        ("3 to 9", [[3, 9]], []),
        ("between nodes 9 and 3", [[3, 9]], []),
        ("link 3,9", [[3, 9]], []),
        ("nodes 3 and 9", [[3, 9]], []),
        ("links 0-3, 9-3", [[0, 3], [3, 9]], []),
        ("break link 3-9 and link 3-7", [[3, 9]], [[3, 7]]),
    ]

    for text, expected_fail, expected_dropped in phrasing_cases:
        res = parse_scenario_fallback(text, G=G)
        assert res["fail_links"] == expected_fail, f"Failed for '{text}': got fail_links {res['fail_links']}, expected {expected_fail}"
        assert res["dropped_links"] == expected_dropped, f"Failed for '{text}': got dropped_links {res['dropped_links']}, expected {expected_dropped}"


def test_mocked_model_returns_empty_links_but_text_has_links():
    """
    Unit Test: Mocked LLM returns fail_links=none while text contains links.
    Verifies regex cross-check recovers the links and records 'model missed links; used keyword parser' in parsed_by.
    """
    from unittest.mock import patch, MagicMock
    from network import create_random_network
    from llm_assistant import parse_scenario

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "load_scale=1.5; fail_links=none; n_demands=default"
    mock_choice.message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    G = create_random_network(num_nodes=20, seed=42)

    with patch('llm_assistant.get_openai_client', return_value=(mock_client, "https://mock-endpoint")):
        sc, _, status_msg = parse_scenario("Increase load by 50% and fail links 0-3 and 9-3", G=G)

    assert sc["fail_links"] == [[0, 3], [3, 9]]
    assert sc["dropped_links"] == []
    assert "model missed links; used keyword parser" in sc["parsed_by"]


def test_property_requested_links_invariant():
    """
    Property-style Test: For many random sentences built from random node pairs,
    the union of fail_links and dropped_links equals the set of normalized requested links.
    No link is lost!
    """
    import random
    from network import create_random_network
    from llm_assistant import parse_scenario_fallback

    G = create_random_network(num_nodes=20, seed=42)
    rng = random.Random(1337)

    templates = [
        "break link between nodes {u} and {v}",
        "fail links {u1}-{v1} and {u2}-{v2}",
        "link {u},{v}",
        "nodes {u} and {v}",
        "cut link {u} to {v}",
    ]

    for trial in range(100):
        u1, v1 = rng.randint(0, 19), rng.randint(0, 19)
        while u1 == v1:
            v1 = rng.randint(0, 19)

        u2, v2 = rng.randint(0, 19), rng.randint(0, 19)
        while u2 == v2:
            v2 = rng.randint(0, 19)

        tmpl = rng.choice(templates)
        sentence = tmpl.format(u=u1, v=v1, u1=u1, v1=v1, u2=u2, v2=v2)

        res = parse_scenario_fallback(sentence, G=G)

        norm1 = [min(u1, v1), max(u1, v1)]
        norm2 = [min(u2, v2), max(u2, v2)]
        if "{u1}" in tmpl:
            expected_set = {tuple(norm1), tuple(norm2)}
        else:
            expected_set = {tuple(norm1)}

        actual_set = set(tuple(p) for p in res["fail_links"]) | set(tuple(p) for p in res["dropped_links"])

        assert actual_set == expected_set, f"Trial {trial} failed for '{sentence}': actual {actual_set} != expected {expected_set}"
        assert len(res["fail_links"]) + len(res["dropped_links"]) == len(expected_set), "Disjoint invariant failed"




