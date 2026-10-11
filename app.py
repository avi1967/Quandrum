import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import json
import os
import time
import torch

from network import create_random_network, generate_demands, evaluate
from qubo import build_qubo_matrix, qubo_to_ising, decode_with_repair, feasible_coordinate_descent_polish, compute_qubo_energy
from baselines import solve_static_dijkstra, solve_greedy_congestion, solve_simulated_annealing
from sb_module import solve_simulated_bifurcation
from llm_assistant import parse_scenario, explain_results, get_openai_client

# Page Configuration
st.set_page_config(
    page_title="Quantum-Inspired Network Route Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #1A202C;
        border-left: 4px solid #4FD1C5;
        padding: 0.85rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #E2E8F0;
        font-size: 0.93rem;
    }
    .metric-card {
        background: #2D3748;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #63B3ED;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #CBD5E0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Quantum-Inspired Network Route Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Infosys EngNext Prototype | Simulated Bifurcation Routing Solver (runs on CPU) & Hosted LLM Assistant (uses Qualcomm Cloud AI 100 via Cirrascale)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ℹ️ <b>Quantum-INSPIRED Architecture Notice</b>: This system implements <b>Simulated Bifurcation (dSB)</b> as a static PyTorch graph executing on classical CPU hardware. It does <b>NOT</b> use quantum hardware or SDKs and makes no claims of quantum advantage.
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Settings")

seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
num_nodes = st.sidebar.slider("Number of Nodes", min_value=10, max_value=40, value=20, step=5)
num_demands = st.sidebar.slider("Number of Demands", min_value=5, max_value=30, value=15, step=1)
load_mult = st.sidebar.slider("Load Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ QUBO Penalty Weights")
A_weight = st.sidebar.slider("Penalty A (One-Hot Constraint)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
B_weight = st.sidebar.slider("Penalty B (Congestion Weight)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
C_weight = st.sidebar.slider("Penalty C (Failure Risk Weight)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

# Session state for link failures
if 'disabled_links' not in st.session_state or st.session_state.get('last_seed') != seed or st.session_state.get('last_nodes') != num_nodes:
    st.session_state['disabled_links'] = []
    st.session_state['last_seed'] = seed
    st.session_state['last_nodes'] = num_nodes

st.sidebar.markdown("---")
st.sidebar.header("💥 Failure Injection")
col_fail1, col_fail2 = st.sidebar.columns(2)

# Generate Graph and Demands
@st.cache_data(show_spinner=False)
def get_network_data(nodes: int, demands_cnt: int, s: int):
    G = create_random_network(num_nodes=nodes, radius=0.35, seed=s)
    demands, candidate_paths = generate_demands(G, num_demands=demands_cnt, seed=s, k=4)
    return G, demands, candidate_paths

G, demands, candidate_paths = get_network_data(num_nodes, num_demands, seed)

if col_fail1.button("Inject Failures"):
    rng = np.random.RandomState(int(time.time()) % 10000)
    edges_list = list(G.edges())
    num_to_fail = min(2, len(edges_list))
    chosen_edges = rng.choice(len(edges_list), size=num_to_fail, replace=False)
    st.session_state['disabled_links'] = [(min(edges_list[i][0], edges_list[i][1]), max(edges_list[i][0], edges_list[i][1])) for i in chosen_edges]
    st.toast(f"Injected {num_to_fail} random link failures!", icon="⚠️")

if col_fail2.button("Reset Links"):
    st.session_state['disabled_links'] = []
    st.toast("Restored all network links.", icon="🔄")

disabled_links = st.session_state['disabled_links']

if disabled_links:
    st.sidebar.warning(f"Active Link Failures: {len(disabled_links)} link(s) down")

# Helper function to run all solvers for current state
def solve_all(G, demands, candidate_paths, load_m, disabled_l, A, B, C, seed_val):
    num_d = len(demands)
    k = 4
    Q, meta = build_qubo_matrix(G, demands, candidate_paths, A=A, B=B, C=C, load_mult=load_m)
    J, h, offset = qubo_to_ising(Q)

    results = {}

    # 0. Static (stale paths) - non-reactive reference baseline before failure
    G_full = create_random_network(num_nodes=len(G.nodes()), radius=0.35, seed=seed_val)
    stale_demands, stale_candidate_paths = generate_demands(G_full, num_demands=num_d, seed=seed_val, k=k)
    sel_stale = solve_static_dijkstra(len(stale_demands))
    res_stale = evaluate(G_full, stale_demands, stale_candidate_paths, sel_stale, load_m, disabled_l)
    results["Static (stale paths)"] = {**res_stale, "selection": sel_stale, "raw_feasibility": None, "runtime_ms": 0.0}

    # 1. Static Dijkstra (Reactive on surviving graph)
    t0 = time.time()
    sel_d = solve_static_dijkstra(num_d)
    t_d = (time.time() - t0) * 1000.0
    res_d = evaluate(G, demands, candidate_paths, sel_d, load_m, disabled_l)
    results["Static Dijkstra"] = {**res_d, "selection": sel_d, "raw_feasibility": None, "runtime_ms": t_d}

    # 2. Greedy Congestion
    t0 = time.time()
    sel_g = solve_greedy_congestion(G, demands, candidate_paths, k=k, load_mult=load_m)
    t_g = (time.time() - t0) * 1000.0
    res_g = evaluate(G, demands, candidate_paths, sel_g, load_m, disabled_l)
    results["Greedy Congestion"] = {**res_g, "selection": sel_g, "raw_feasibility": None, "runtime_ms": t_g}

    # 3. Simulated Annealing
    t0 = time.time()
    sel_sa = solve_simulated_annealing(Q, num_d, k=k, steps=500, seed=seed_val)
    t_sa = (time.time() - t0) * 1000.0
    res_sa = evaluate(G, demands, candidate_paths, sel_sa, load_m, disabled_l)
    results["Simulated Annealing"] = {**res_sa, "selection": sel_sa, "raw_feasibility": None, "runtime_ms": t_sa}

    # 4. SB FP32
    t0 = time.time()
    best_s_32, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float32, seed=seed_val)
    sel_32, raw_feas_32 = decode_with_repair(best_s_32, Q, num_d, k)
    t_32 = (time.time() - t0) * 1000.0
    res_32 = evaluate(G, demands, candidate_paths, sel_32, load_m, disabled_l)
    results["SB FP32"] = {**res_32, "selection": sel_32, "raw_feasibility": raw_feas_32, "runtime_ms": t_32}

    # 5. SB FP16
    t0 = time.time()
    best_s_16, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float16, seed=seed_val)
    sel_16, raw_feas_16 = decode_with_repair(best_s_16, Q, num_d, k)
    t_16 = (time.time() - t0) * 1000.0
    res_16 = evaluate(G, demands, candidate_paths, sel_16, load_m, disabled_l)
    results["SB FP16"] = {**res_16, "selection": sel_16, "raw_feasibility": raw_feas_16, "runtime_ms": t_16}

    # 6. SB FP16 + Polish
    t0 = time.time()
    best_s_16_p, _ = solve_simulated_bifurcation(J, h, meta["N"], dtype=torch.float16, seed=seed_val)
    sel_16_unp, raw_feas_16_p = decode_with_repair(best_s_16_p, Q, num_d, k)
    sel_16_pol = feasible_coordinate_descent_polish(sel_16_unp, Q, num_d, k)
    t_16_pol = (time.time() - t0) * 1000.0
    res_16_pol = evaluate(G, demands, candidate_paths, sel_16_pol, load_m, disabled_l)
    results["SB FP16 + Polish"] = {**res_16_pol, "selection": sel_16_pol, "raw_feasibility": raw_feas_16_p, "runtime_ms": t_16_pol}

    return results, Q

solver_results, Q_matrix = solve_all(G, demands, candidate_paths, load_mult, disabled_links, A_weight, B_weight, C_weight, seed)

# Plotly Network Plot Helper
def create_network_plot(G, demands, candidate_paths, selection, disabled_links, title="Network Routes"):
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    disabled_set = set(disabled_links)
    
    # Calculate link loads
    link_utils = evaluate(G, demands, candidate_paths, selection, load_mult, disabled_links)['link_utils']

    fig = go.Figure()

    # Draw all edges
    for u, v, d in G.edges(data=True):
        edge = (min(u, v), max(u, v))
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if edge in disabled_set:
            # Dashed Red line for disabled link
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='#E53E3E', width=3, dash='dash'),
                hoverinfo='text',
                text=f"Disabled Link ({u}-{v})",
                showlegend=False
            ))
        else:
            util = link_utils.get(edge, 0.0)
            if util > 1.0:
                color = '#F56565' # Red overloaded
                width = 3.5
            elif util > 0.7:
                color = '#ECC94B' # Yellow heavy
                width = 2.5
            else:
                color = '#4A5568' # Grey normal
                width = 1.5

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='text',
                text=f"Link ({u}-{v}) | Base Lat: {d['base_latency']:.1f}ms | Cap: {d['capacity']:.1f}M | Util: {util*100:.1f}%",
                showlegend=False
            ))

    # Highlight active flow paths
    colors = ['#3182CE', '#38A169', '#DD6B20', '#805AD5', '#D69E2E', '#E53E3E', '#319795']
    for d_idx, p_idx in enumerate(selection):
        path_edges = candidate_paths[d_idx][p_idx]
        flow_color = colors[d_idx % len(colors)]
        for u, v in path_edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=flow_color, width=3),
                opacity=0.6,
                showlegend=False
            ))

    # Draw nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=22, color='#63B3ED', line=dict(color='#2B6CB0', width=2)),
        text=[str(n) for n in G.nodes()],
        textposition="middle center",
        textfont=dict(color="#1A202C", size=11, family="sans-serif"),
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#E2E8F0")),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380
    )

    return fig

# Main Navigation Tabs
tab_compare, tab_bench, tab_ablation, tab_npu, tab_assistant = st.tabs([
    "🗺️ Live Route Comparator",
    "📊 Benchmark Suite",
    "🧪 Penalty Ablation Study",
    "⚡ Qualcomm / NPU Status",
    "🤖 Qualcomm Cloud AI Assistant"
])

# TAB 1: LIVE ROUTE COMPARATOR
with tab_compare:
    st.subheader("Side-by-Side Dual Method Routing Comparison")
    
    col_sel1, col_sel2 = st.columns(2)
    solver_names = list(solver_results.keys())
    
    m1 = col_sel1.selectbox("Select Method 1 (Baseline)", solver_names, index=0)
    m2 = col_sel2.selectbox("Select Method 2 (Quantum-Inspired SB)", solver_names, index=5)
    
    res1 = solver_results[m1]
    res2 = solver_results[m2]
    
    fig1 = create_network_plot(G, demands, candidate_paths, res1["selection"], disabled_links, title=f"Method 1: {m1}")
    fig2 = create_network_plot(G, demands, candidate_paths, res2["selection"], disabled_links, title=f"Method 2: {m2}")
    
    col_graph1, col_graph2 = st.columns(2)
    col_graph1.plotly_chart(fig1, use_container_width=True)
    col_graph2.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 📈 Quantitative Performance Metrics Comparison")
    
    metrics_cols = st.columns(7)
    
    def render_metric_diff(title, val1, val2, fmt="{:.2f}", lower_is_better=True):
        diff = val2 - val1
        if lower_is_better:
            color = "#48BB78" if diff <= 0 else "#F56565"
        else:
            color = "#48BB78" if diff >= 0 else "#F56565"
        
        sign = "+" if diff > 0 else ""
        return f"""
        <div class="metric-card">
            <div class="metric-lbl">{title}</div>
            <div class="metric-val">{fmt.format(val2)}</div>
            <div style="font-size:0.8rem; color:{color}; font-weight:600;">{m1}: {fmt.format(val1)} ({sign}{fmt.format(diff)})</div>
        </div>
        """

    metrics_cols[0].markdown(render_metric_diff("Avg Latency (ms)", res1["avg_latency"], res2["avg_latency"]), unsafe_allow_html=True)
    metrics_cols[1].markdown(render_metric_diff("Max Latency (ms)", res1["max_latency"], res2["max_latency"]), unsafe_allow_html=True)
    metrics_cols[2].markdown(render_metric_diff("Max Utilization", res1["max_utilisation"], res2["max_utilisation"], fmt="{:.2f}"), unsafe_allow_html=True)
    metrics_cols[3].markdown(render_metric_diff("Overloaded Links", res1["overloaded_links"], res2["overloaded_links"], fmt="{:d}"), unsafe_allow_html=True)
    metrics_cols[4].markdown(render_metric_diff("Failed Link Flows", res1.get("flows_on_failed_links", 0), res2.get("flows_on_failed_links", 0), fmt="{:d}"), unsafe_allow_html=True)
    metrics_cols[5].markdown(render_metric_diff("Overloaded Link Flows", res1.get("flows_on_overloaded_links", 0), res2.get("flows_on_overloaded_links", 0), fmt="{:d}"), unsafe_allow_html=True)
    metrics_cols[6].markdown(render_metric_diff("Solve Time (ms)", res1["runtime_ms"], res2["runtime_ms"], fmt="{:.1f}"), unsafe_allow_html=True)

# TAB 2: BENCHMARK SUITE
with tab_bench:
    st.subheader("📊 Multi-Scenario Statistical Benchmark Results")
    
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            bench_data = json.load(f)
            
        meta_info = bench_data.get("_meta", {})
        if meta_info:
            ts = meta_info.get("timestamp", "N/A")
            seeds_cnt = len(meta_info.get("seeds", []))
            sett = meta_info.get("solver_settings", {})
            st.caption(f"📅 **Benchmark Timestamp**: `{ts}` | 🎲 **Evaluation Seeds**: `{seeds_cnt} seeds` | ⚙️ **Settings**: `A={sett.get('A')}, B={sett.get('B')}, C={sett.get('C')}`")
            
        scenario_keys = [k for k in bench_data.keys() if not k.startswith('_')]
        selected_scenario = st.selectbox("Select Scenario View", scenario_keys)
        sc_data = bench_data[selected_scenario]
        
        # Display Statistical Summary Table
        st.markdown(f"#### 📈 Summary Table: {selected_scenario} (Mean, Std, 95% CI)")
        
        table_rows = []
        for method_name, m_stats in sc_data.items():
            rf_val = m_stats.get('raw_feasibility')
            rf_str = f"{rf_val*100:.1f}%" if rf_val is not None else "n/a"
            
            lat_st = m_stats.get("avg_latency_stats", {})
            lat_mean = lat_st.get("mean", m_stats.get("avg_latency", 0.0))
            lat_std = lat_st.get("std", 0.0)
            lat_ci = lat_st.get("ci95", [lat_mean, lat_mean])
            
            table_rows.append({
                "Method": method_name,
                "Avg Latency (Mean ± Std)": f"{lat_mean:.2f} ± {lat_std:.2f} ms",
                "95% CI Latency": f"[{lat_ci[0]:.2f}, {lat_ci[1]:.2f}]",
                "Max Utilization": f"{m_stats['max_utilisation']:.2f}",
                "Overloaded Link Flows": f"{m_stats.get('flows_on_overloaded_links', 0):.1f}",
                "Failed Link Flows": f"{m_stats.get('flows_on_failed_links', 0):.1f}",
                "Raw Feasibility": rf_str,
                "Runtime (ms)": f"{m_stats['runtime_ms']:.2f}"
            })
            
        st.dataframe(table_rows, use_container_width=True)
        
        # Display Paired Statistical Comparison vs Greedy
        st.markdown(f"#### 🔬 Paired Statistical Comparison vs Greedy Congestion: {selected_scenario}")
        st.caption("Note: A method is labeled **'better'** ONLY if its paired 95% confidence interval of difference excludes 0; otherwise labeled **'comparable'**.")
        
        paired_rows = []
        for method_name, m_stats in sc_data.items():
            p_vs_g = m_stats.get("paired_vs_greedy", {}).get("avg_latency", {})
            if p_vs_g:
                m_diff = p_vs_g.get("mean_diff", 0.0)
                ci_diff = p_vs_g.get("ci95_diff", [0.0, 0.0])
                w, t, l = p_vs_g.get("wins", 0), p_vs_g.get("ties", 0), p_vs_g.get("losses", 0)
                ex_zero = p_vs_g.get("excludes_zero", False)
                conc = p_vs_g.get("conclusion", "comparable") if method_name != "Greedy Congestion" else "baseline"
                
                paired_rows.append({
                    "Method": method_name,
                    "Mean Latency Diff vs Greedy": f"{m_diff:+.2f} ms",
                    "95% CI of Difference": f"[{ci_diff[0]:+.2f}, {ci_diff[1]:+.2f}]",
                    "Paired W / T / L": f"W:{w} / T:{t} / L:{l}",
                    "CI Excludes 0?": "Yes" if ex_zero else "No",
                    "Statistical Conclusion": conc
                })
        st.dataframe(paired_rows, use_container_width=True)
        
        # Comparative Charts
        st.markdown("#### Scenario Metric Comparisons")
        col_c1, col_c2 = st.columns(2)
        
        methods_list = list(sc_data.keys())
        avg_lats = [sc_data[m]["avg_latency"] for m in methods_list]
        max_utils = [sc_data[m]["max_utilisation"] for m in methods_list]
        runtimes = [sc_data[m]["runtime_ms"] for m in methods_list]
        raw_feas = [sc_data[m].get("raw_feasibility", 0) * 100 if sc_data[m].get("raw_feasibility") is not None else 0 for m in methods_list]
        
        fig_lat = go.Figure(data=[go.Bar(x=methods_list, y=avg_lats, marker_color='#63B3ED')])
        fig_lat.update_layout(title="Average Path Latency (ms)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
        col_c1.plotly_chart(fig_lat, use_container_width=True)
        
        fig_util = go.Figure(data=[go.Bar(x=methods_list, y=max_utils, marker_color='#F6AD55')])
        fig_util.update_layout(title="Max Link Utilization", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
        col_c2.plotly_chart(fig_util, use_container_width=True)
        
        col_c3, col_c4 = st.columns(2)
        
        fig_time = go.Figure(data=[go.Bar(x=methods_list, y=runtimes, marker_color='#68D391')])
        fig_time.update_layout(title="Solver Runtime (ms)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
        col_c3.plotly_chart(fig_time, use_container_width=True)

        fig_feas = go.Figure(data=[go.Bar(x=methods_list, y=raw_feas, marker_color='#FC8181')])
        fig_feas.update_layout(title="Raw SB Feasibility % BEFORE Repair (QUBO Solvers)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
        col_c4.plotly_chart(fig_feas, use_container_width=True)

    else:
        st.info("No benchmark results found. Run `python benchmark.py` to populate data.")

# TAB 3: PENALTY ABLATION STUDY
with tab_ablation:
    st.subheader("QUBO Penalty Hyperparameter Sensitivity & Ablation Analysis")
    st.markdown("""
    This ablation study evaluates the solver's performance under under-penalized vs over-penalized weights:
    - **Weak Penalty A**: Solvers violate one-hot constraints; raw output contains multiple or missing paths.
    - **Strong Penalty A**: Solvers strictly enforce valid path selection but may ignore congestion trade-offs.
    - **Weak Penalty B**: Solvers ignore link utilization, resulting in high link congestion and overloads.
    - **Strong Penalty B**: Solvers prioritize spreading traffic evenly across candidate paths.
    """)
    
    A_test_vals = [0.1, 5.0, 20.0, 100.0]
    
    ablation_results = []
    for test_A in A_test_vals:
        Q_test, _ = build_qubo_matrix(G, demands, candidate_paths, A=test_A, B=B_weight, C=C_weight, load_mult=load_mult)
        J_test, h_test, offset_test = qubo_to_ising(Q_test)
        
        best_s, _ = solve_simulated_bifurcation(J_test, h_test, len(demands)*4, dtype=torch.float32, seed=seed)
        sel, raw_feas = decode_with_repair(best_s, Q_test, len(demands), 4)
        res = evaluate(G, demands, candidate_paths, sel, load_mult, disabled_links)
        
        ablation_results.append({
            "Penalty A": test_A,
            "Raw Feasibility %": raw_feas * 100,
            "Max Utilization": res["max_utilisation"],
            "Overloaded Links": res["overloaded_links"],
            "Avg Latency (ms)": res["avg_latency"]
        })
        
    st.dataframe(ablation_results, use_container_width=True)
    
    col_ab1, col_ab2 = st.columns(2)
    
    fig_ab_feas = go.Figure(data=[go.Scatter(x=[r["Penalty A"] for r in ablation_results], y=[r["Raw Feasibility %"] for r in ablation_results], mode='lines+markers', line=dict(color='#F6AD55', width=3))])
    fig_ab_feas.update_layout(title="Raw SB Feasibility % vs Penalty A Weight", xaxis_title="Penalty A Weight", yaxis_title="Raw Feasibility %", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
    col_ab1.plotly_chart(fig_ab_feas, use_container_width=True)

    fig_ab_util = go.Figure(data=[go.Scatter(x=[r["Penalty A"] for r in ablation_results], y=[r["Max Utilization"] for r in ablation_results], mode='lines+markers', line=dict(color='#68D391', width=3))])
    fig_ab_util.update_layout(title="Max Link Utilization vs Penalty A Weight", xaxis_title="Penalty A Weight", yaxis_title="Max Utilization", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
    col_ab2.plotly_chart(fig_ab_util, use_container_width=True)

# TAB 4: QUALCOMM / NPU STATUS
with tab_npu:
    st.subheader("⚡ Qualcomm / NPU Status")
    st.markdown("""
    *Note: The Simulated Bifurcation routing solver itself executes on CPU in PyTorch. The PyTorch module can be exported to ONNX format for future NPU work. Qualcomm AI Hub was NOT used for on-device compilation in this run.*
    """)

    col_npu1, col_npu2 = st.columns([2, 1])

    with col_npu1:
        st.markdown("### 📊 Hardware Profiling Metrics")

        if os.path.exists("npu_profile.json"):
            with open("npu_profile.json", "r") as f:
                npu_data = json.load(f)

            source = npu_data.get("source", "none")
            measured_on = npu_data.get("measured_on", "None")
            lat_ms = npu_data.get("latency_ms")
            lat_lbl = npu_data.get("latency_label", "Hardware NPU Latency")
            mem_mb = npu_data.get("peak_memory_mb")
            cov_pct = npu_data.get("npu_coverage_pct")
            api_rt = npu_data.get("api_roundtrip_ms")
            msg = npu_data.get("message", "")

            # If source is none or hardware latency is None
            if source == "qualcomm_ai_hub_profile_job":
                st.success(f"**Status**: Real Qualcomm AI Hub profiling completed.")
            elif source == "cirrascale_models_endpoint":
                st.info(f"**Status**: Connected to Cirrascale models REST endpoint. *(On-device NPU profiler job not run)*")
            elif source == "local_onnxruntime_cpu":
                st.info(f"**Status**: Local ONNX Runtime CPU measurement completed.")
            else:
                st.warning(f"**Status**: No on-device hardware profiling has been performed.")

            st.markdown(f"**Data Source**: `{source}`")
            st.markdown(f"**Measured On**: `{measured_on}`")
            if api_rt is not None:
                st.markdown(f"**REST API Roundtrip Time**: `{api_rt:.2f} ms` *(HTTP network latency to `/models` endpoint)*")
            if msg:
                st.caption(f"Note: {msg}")

            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                lat_str = f"{lat_ms:.3f} ms" if lat_ms is not None else "Not measured"
                st.metric(label=lat_lbl, value=lat_str)

            with m_col2:
                mem_str = f"{mem_mb:.2f} MB" if mem_mb is not None else "Not measured"
                st.metric(label="Peak NPU Memory", value=mem_str)

            with m_col3:
                cov_str = f"{cov_pct:.1f}%" if cov_pct is not None else "Not measured"
                st.metric(label="NPU Compute Coverage", value=cov_str)

        else:
            st.info("No profile file found (`npu_profile.json`). Metrics: **Not measured**.")

    with col_npu2:
        st.markdown("### 🧩 PyTorch JIT Trace Verification")
        if st.button("Run PyTorch JIT Trace Check"):
            from export_onnx import trace_and_verify
            success = trace_and_verify(verbose=False)
            if success:
                st.success("PyTorch JIT Trace verified! Traced graph matches eager output.")
            else:
                st.error("JIT Trace verification failed!")

# TAB 5: QUALCOMM CLOUD AI ASSISTANT
with tab_assistant:
    st.subheader("🤖 Qualcomm Cloud AI Assistant (Natural Language Scenario & Explanation)")
    st.markdown("""
    <div class="disclaimer-box">
        🤖 <b>AI Assistant Architecture Notice</b>: Natural language parsing and explanation use hosted LLMs (Llama-3.1-8B / Llama-3.3-70B) on <b>Qualcomm Cloud AI 100 Ultra</b> via Cirrascale Inference Cloud. The routing solver (Simulated Bifurcation / QUBO) executes on <b>CPU in PyTorch</b>.
    </div>
    """, unsafe_allow_html=True)

    client, endpoint = get_openai_client()
    if not client:
        st.warning("⚠️ **Cirrascale Inference Cloud API key or endpoint not configured in `.env`.**\n\n"
                   "The LLM features will fall back to local keyword parsing. To enable hosted LLMs on Qualcomm Cloud AI 100:\n"
                   "- Add `INFERENCE_CLOUD_API_KEY=your_key` and `INFERENCE_CLOUD_ENDPOINT=https://aisuite.cirrascale.com/apis/v2` to `.env`.")
    else:
        model_p = os.getenv("INFERENCE_CLOUD_MODEL", "Llama-3.1-8B")
        model_e = os.getenv("INFERENCE_CLOUD_EXPLAIN_MODEL", "Llama-3.3-70B")
        st.success(f" Connected to Cirrascale Inference Cloud (`{endpoint}`).\nParser Model: `{model_p}` | Explanation Model: `{model_e}`")

    st.markdown("### 💬 Natural Language Scenario Input")
    
    example_prompts = [
        "Double traffic load and fail link 3-9",
        "Set demand count to 20 with 1.5x load multiplier",
        "Triple traffic load with 10 demands and break links 1-5 and 5-8",
        "Reduce load to 0.5x with 8 demands"
    ]
    
    col_ex1, col_ex2 = st.columns([3, 1])
    selected_ex = col_ex2.selectbox("Load Example Prompt", ["Select..."] + example_prompts)
    
    default_prompt = selected_ex if selected_ex != "Select..." else "Double traffic load and break link between nodes 3 and 9"
    user_prompt = col_ex1.text_area("Describe network scenario in plain English:", value=default_prompt, height=80)
    
    if st.button("🤖 Parse Scenario with Hosted LLM", type="primary"):
        with st.spinner("Calling Qualcomm Cloud AI 100 LLM..."):
            scenario_dict, parse_lat_ms, parse_msg = parse_scenario(user_prompt, G=G)
            st.session_state['parsed_scenario'] = scenario_dict
            st.session_state['parse_lat_ms'] = parse_lat_ms
            st.session_state['parse_msg'] = parse_msg

    if 'parsed_scenario' in st.session_state:
        sc = st.session_state['parsed_scenario']
        lat = st.session_state.get('parse_lat_ms', 0.0)
        msg = st.session_state.get('parse_msg', '')

        st.markdown("#### ⚙️ Parsed Scenario Parameters")
        st.info(f"**Parser Status**: `{sc.get('parsed_by', 'LLM')}` | {msg}")
        if lat > 0:
            st.metric(label="LLM API response time (network + inference)", value=f"{lat:.1f} ms")
        else:
            st.metric(label="LLM API response time (network + inference)", value="Not measured (Fallback)")

        c1, c2, c3 = st.columns(3)
        edit_load = c1.number_input("Load Multiplier (load_scale)", min_value=0.25, max_value=4.0, value=float(sc.get("load_scale", 1.0)), step=0.1)
        edit_demands = c2.number_input("Demands Count (n_demands)", min_value=4, max_value=40, value=int(sc.get("n_demands") or num_demands), step=1)
        
        fail_links_str = ", ".join([f"{u}-{v}" for u, v in sc.get("fail_links", [])])
        edit_links = c3.text_input("Active Failed Links (e.g. 3-9, 1-5)", value=fail_links_str)

        if sc.get("dropped_links"):
            for u, v in sc["dropped_links"]:
                st.error(f"Link {u}-{v} does not exist in the current graph (seed {seed})")
                near_edges = [f"{min(a,b)}-{max(a,b)}" for a, b in G.edges() if a in (u, v) or b in (u, v)]
                other_edges = [f"{min(a,b)}-{max(a,b)}" for a, b in G.edges() if f"{min(a,b)}-{max(a,b)}" not in near_edges]
                valid_edges_options = sorted(list(set(near_edges))) + sorted(list(set(other_edges)))
                replacement = st.selectbox(
                    f"Select a valid link from graph near nodes {u}/{v} to replace invalid link {u}-{v}:",
                    ["Select valid link..."] + valid_edges_options,
                    key=f"select_replace_{u}_{v}"
                )
                if replacement != "Select valid link...":
                    parts = replacement.split("-")
                    if len(parts) == 2:
                        ru, rv = int(parts[0]), int(parts[1])
                        pair = [min(ru, rv), max(ru, rv)]
                        if pair not in sc.get("fail_links", []):
                            sc["fail_links"].append(pair)

        if st.button("🚀 Apply & Run Solvers on Parsed Scenario"):
            parsed_links = []
            if edit_links.strip():
                for item in edit_links.split(","):
                    parts = item.strip().split("-")
                    if len(parts) == 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            parsed_links.append((min(u, v), max(u, v)))
                        except ValueError:
                            pass

            g_new, dem_new, cand_new = get_network_data(num_nodes, edit_demands, seed)
            
            with st.spinner("Running routing solvers on CPU..."):
                sol_results, _ = solve_all(g_new, dem_new, cand_new, edit_load, parsed_links, A_weight, B_weight, C_weight, seed)
                st.session_state['assistant_sol_results'] = sol_results
                st.session_state['assistant_scenario'] = {
                    "load_scale": edit_load,
                    "n_demands": edit_demands,
                    "fail_links": parsed_links
                }

    if 'assistant_sol_results' in st.session_state:
        st.markdown("### 📊 Scenario Routing Solvers Performance")
        sol_res = st.session_state['assistant_sol_results']
        sc_info = st.session_state['assistant_scenario']

        summary_rows = []
        for name, data in sol_res.items():
            summary_rows.append({
                "Solver Method": name,
                "Avg Latency (ms)": f"{data['avg_latency']:.2f}",
                "Max Latency (ms)": f"{data['max_latency']:.2f}",
                "Max Utilization": f"{data['max_utilisation']:.2f}",
                "Overloaded Links": f"{data.get('overloaded_links', 0):d}",
                "Failed Link Flows": f"{data.get('flows_on_failed_links', 0):d}",
                "Overloaded Link Flows": f"{data.get('flows_on_overloaded_links', 0):d}",
                "Solver Execution Time (ms)": f"{data['runtime_ms']:.1f}"
            })
        st.dataframe(summary_rows, use_container_width=True)

        if st.button("💡 Explain Results with Hosted LLM (Llama-3.3-70B)"):
            with st.spinner("Generating AI explanation on Cloud AI 100..."):
                explanation, exp_lat_ms, exp_msg = explain_results(sol_res, sc_info)
                st.session_state['assistant_explanation'] = explanation
                st.session_state['exp_lat_ms'] = exp_lat_ms
                st.session_state['exp_msg'] = exp_msg

    if 'assistant_explanation' in st.session_state:
        st.markdown("#### 📝 AI Performance Analysis")
        st.markdown(f"> {st.session_state['assistant_explanation']}")
        
        exp_lat = st.session_state.get('exp_lat_ms', 0.0)
        if exp_lat > 0:
            st.metric(label="LLM API response time (network + inference)", value=f"{exp_lat:.1f} ms")
        else:
            st.metric(label="LLM API response time (network + inference)", value="Not measured")



