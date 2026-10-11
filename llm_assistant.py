import os
import re
import json
import time
import networkx as nx
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from local .env if present
load_dotenv()

def clean_llm_response(text: str) -> str:
    """
    Cleans raw LLM output:
    1. Strips <think>...</think> reasoning blocks (for DeepSeek / reasoning models).
    2. Strips Markdown code fences (e.g. ```json ... ```).
    3. Strips leading/trailing whitespace.
    """
    if not text:
        return ""
    
    # Strip <think>...</think> blocks (including multiline & case-insensitive)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip unclosed <think> if truncated by max tokens
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

def get_openai_client() -> Tuple[Optional[Any], str]:
    """
    Initializes OpenAI client using INFERENCE_CLOUD_API_KEY and INFERENCE_CLOUD_ENDPOINT.
    Returns (client, endpoint_url). Never logs or prints the API key.
    """
    api_key = os.getenv("INFERENCE_CLOUD_API_KEY")
    endpoint = os.getenv("INFERENCE_CLOUD_ENDPOINT")

    if not api_key or not endpoint:
        return None, ""

    clean_endpoint = endpoint.strip().rstrip('/')
    if not clean_endpoint.startswith("http"):
        clean_endpoint = f"https://{clean_endpoint}"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=clean_endpoint,
            timeout=60.0  # 60 second timeout requirement
        )
        return client, clean_endpoint
    except Exception:
        return None, clean_endpoint

def extract_link_pairs(text: str) -> List[List[int]]:
    """
    Tightened regex parser for link pairs.
    Only extracts pairs near failure/link words, inside fail_links=... values, or from digit pairs.
    Strips stray tokens like 'none', percentages, demand counts, and load multipliers.
    Accepts separators: '-', '/', ',', 'to', 'and'. Accepts bare space only right after link/links/nodes.
    Normalizes every pair as [min(u,v), max(u,v)] and de-duplicates.
    """
    if not text:
        return []

    # Strip stray token 'none'
    text_clean = re.sub(r'\bnone\b', '', text, flags=re.IGNORECASE)

    # Strip percentages (e.g. 50%, 100%) so numbers in percentages are not treated as links
    text_clean = re.sub(r'(\d+(?:\.\d+)?)\s*%', '', text_clean)

    # Strip demand counts (e.g. "5 to 10 demands", "4 and 5 demands", "25 demands", "10 flows")
    text_clean = re.sub(r'\b\d+\s*(?:to|and|-)\s*\d+\s*(?:demands|flows)\b', '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'\b\d+\s*(?:demands|flows)\b', '', text_clean, flags=re.IGNORECASE)

    # Strip load multipliers (e.g. "2 to 3 times", "1.5x load", "2.5x", "2 to 3 x load")
    text_clean = re.sub(r'\b\d+(?:\.\d+)?\s*to\s*\d+(?:\.\d+)?\s*(?:times|x|\*)\s*(?:load)?\b', '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'\b\d+(?:\.\d+)?\s*to\s*\d+(?:\.\d+)?\s+load\b', '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'\b\d+(?:\.\d+)?\s*(?:times|x|\*)\s*(?:load)?\b', '', text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r'\bload\s*(?:by\s*)?\d+(?:\.\d+)?\s*to\s*\d+(?:\.\d+)?\s*(?:times|x|\*)?\b', '', text_clean, flags=re.IGNORECASE)

    link_keywords = [r'fail', r'break', r'cut', r'kill', r'down', r'disable', r'link', r'links', r'nodes', r'between']
    has_keyword = any(re.search(r'\b' + kw + r'\b', text_clean, re.IGNORECASE) for kw in link_keywords)
    is_fail_links_val = bool(re.search(r'fail_links\s*=\s*', text, re.IGNORECASE))
    is_pure_pair = bool(re.match(r'^\s*(\d+)\s*(?:-|\/|,|\bto\b|\band\b)\s*(\d+)\s*$', text_clean, re.IGNORECASE))
    has_digit_pair = bool(re.search(r'\d+\s*(?:-|\/|,|\bto\b|\band\b|\s+)\s*\d+', text_clean, re.IGNORECASE))

    if not has_keyword and not is_fail_links_val and not has_digit_pair:
        return []

    fail_links_match = re.search(r'fail_links\s*=\s*([^;\n]+)', text_clean, re.IGNORECASE)
    if fail_links_match:
        target_text = fail_links_match.group(1)
    elif is_fail_links_val or is_pure_pair or not has_keyword:
        target_text = text_clean
    else:
        windows = []
        for match in re.finditer(r'\b(?:fail|break|cut|kill|down|disable|link|links|nodes|between)\b', text_clean, re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(text_clean), match.end() + 40)
            windows.append(text_clean[start:end])
        target_text = " ".join(windows) if windows else text_clean

    fail_links = []

    # 1. Matches with separators: -, /, comma, to, and
    pattern_sep = r'(\d+)\s*(?:-|\/|,|\bto\b|\band\b)\s*(\d+)'
    for u_str, v_str in re.findall(pattern_sep, target_text, re.IGNORECASE):
        try:
            u, v = int(u_str), int(v_str)
            if u != v:
                pair = [min(u, v), max(u, v)]
                if pair not in fail_links:
                    fail_links.append(pair)
        except ValueError:
            pass

    # 2. Bare space only right after link, links, or nodes
    pattern_space = r'\b(?:link|links|nodes)\s+(\d+)\s+(\d+)\b'
    for u_str, v_str in re.findall(pattern_space, target_text, re.IGNORECASE):
        try:
            u, v = int(u_str), int(v_str)
            if u != v:
                pair = [min(u, v), max(u, v)]
                if pair not in fail_links:
                    fail_links.append(pair)
        except ValueError:
            pass

    return fail_links

def parse_scenario_fallback(text: str, G: Optional[nx.Graph] = None) -> Dict[str, Any]:
    """
    Regex / keyword fallback parser for natural language scenario requests.
    Used when LLM API is unavailable or returns unparseable output.
    """
    text_lower = text.lower()
    
    # Default parameters
    load_scale = 1.0
    n_demands = None
    seed = 42

    # Percentage changes
    pct_increase = re.search(r'(?:increase|boost|up|add|raise|grow)\s*(?:traffic\s*|load\s*)?(?:by\s*)?(\d+(?:\.\d+)?)\s*%', text_lower)
    pct_inc2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|boost|up|add|more|higher)', text_lower)
    pct_decrease = re.search(r'(?:reduce|decrease|drop|down|cut|lower)\s*(?:traffic\s*|load\s*)?(?:by\s*)?(\d+(?:\.\d+)?)\s*%', text_lower)
    pct_dec2 = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:decrease|reduce|drop|down|cut|less|lower)', text_lower)

    if pct_increase:
        try:
            load_scale = 1.0 + float(pct_increase.group(1)) / 100.0
        except ValueError:
            pass
    elif pct_inc2:
        try:
            load_scale = 1.0 + float(pct_inc2.group(1)) / 100.0
        except ValueError:
            pass
    elif pct_decrease:
        try:
            load_scale = max(0.0, 1.0 - float(pct_decrease.group(1)) / 100.0)
        except ValueError:
            pass
    elif pct_dec2:
        try:
            load_scale = max(0.0, 1.0 - float(pct_dec2.group(1)) / 100.0)
        except ValueError:
            pass
    elif "double" in text_lower or "2x" in text_lower or "2.0x" in text_lower:
        load_scale = 2.0
    elif "triple" in text_lower or "3x" in text_lower or "3.0x" in text_lower:
        load_scale = 3.0
    elif "1.5x" in text_lower:
        load_scale = 1.5
    elif "half" in text_lower or "0.5x" in text_lower:
        load_scale = 0.5
    else:
        load_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:x|times|\*)\s*load', text_lower)
        if load_match:
            try:
                load_scale = float(load_match.group(1))
            except ValueError:
                pass

    # Parse demands count only if present in user text
    demand_match = re.search(r'(\d+)\s*(?:demands|flows)', text_lower)
    if demand_match:
        try:
            val = int(demand_match.group(1))
            n_demands = max(4, min(40, val))
        except ValueError:
            n_demands = None

    # Extract link pairs using tightened regex
    fail_links = extract_link_pairs(text)

    # Validate fail_links against graph G if provided
    validated_fail_links = []
    dropped_links = []
    if G is not None:
        edges_set = set((min(u, v), max(u, v)) for u, v in G.edges())
        for pair in fail_links:
            edge = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            if edge in edges_set:
                if [edge[0], edge[1]] not in validated_fail_links:
                    validated_fail_links.append([edge[0], edge[1]])
            else:
                if [edge[0], edge[1]] not in dropped_links:
                    dropped_links.append([edge[0], edge[1]])
    else:
        validated_fail_links = fail_links

    # Clamp load_scale bounds for fallback
    load_scale = max(0.25, min(4.0, float(load_scale)))

    return {
        "load_scale": load_scale,
        "n_demands": n_demands,
        "fail_links": validated_fail_links,
        "dropped_links": dropped_links,
        "seed": seed,
        "parsed_by": "keyword_fallback"
    }

def parse_scenario_text_line(line: str) -> Dict[str, Any]:
    """
    Parses output line from LLM:
    Handles JSON payloads OR plain text key-value strings:
    load_scale=<number>; fail_links=<a-b,c-d or none>; n_demands=<integer or default>
    """
    line_clean = line.strip()
    
    # Try parsing JSON first if line is JSON formatted
    if line_clean.startswith("{") and line_clean.endswith("}"):
        try:
            json_dict = json.loads(line_clean)
            if isinstance(json_dict, dict):
                if "fail_links" in json_dict and isinstance(json_dict["fail_links"], list):
                    parsed_pairs = []
                    for item in json_dict["fail_links"]:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            parsed_pairs.append([min(int(item[0]), int(item[1])), max(int(item[0]), int(item[1]))])
                    json_dict["fail_links"] = parsed_pairs
                return json_dict
        except Exception:
            pass

    result = {}
    has_match = False
    
    # Load scale
    load_match = re.search(r'load_scale\s*=\s*([\d\.]+)', line_clean, re.IGNORECASE)
    if not load_match:
        load_match = re.search(r'load\s*=\s*([\d\.]+)', line_clean, re.IGNORECASE)
    if load_match:
        try:
            result["load_scale"] = float(load_match.group(1))
            has_match = True
        except ValueError:
            result["load_scale"] = load_match.group(1)
            has_match = True

    # Demands count
    demand_match = re.search(r'n_demands\s*=\s*([a-zA-Z0-9_-]+)', line_clean, re.IGNORECASE)
    if not demand_match:
        demand_match = re.search(r'demands\s*=\s*([a-zA-Z0-9_-]+)', line_clean, re.IGNORECASE)
    if demand_match:
        val = demand_match.group(1)
        if val.isdigit():
            result["n_demands"] = int(val)
        else:
            result["n_demands"] = val
        has_match = True

    # Fail links
    fail_match = re.search(r'fail_links\s*=\s*([^;\n]+)', line_clean, re.IGNORECASE)
    if fail_match:
        links_str = fail_match.group(1).strip()
        result["fail_links"] = extract_link_pairs(links_str)
        has_match = True

    if not has_match:
        extracted = extract_link_pairs(line_clean)
        if extracted:
            result["fail_links"] = extracted
            has_match = True

    return result if has_match else {}

def parse_scenario(text: str, G: Optional[nx.Graph] = None) -> Tuple[Dict[str, Any], float, str]:
    """
    Parses natural language requests into scenario parameters using hosted LLM.
    Uses plain-text prompt (NO JSON, NO braces, NO "schema"/"function").
    Returns (scenario_dict, elapsed_ms, status_message).
    """
    client, endpoint = get_openai_client()
    model_p = os.getenv("INFERENCE_CLOUD_MODEL", "Llama-3.1-8B")
    model_e = os.getenv("INFERENCE_CLOUD_EXPLAIN_MODEL", "Llama-3.3-70B")

    # Extract user text links via regex to cross-check model output
    user_text_links = extract_link_pairs(text)

    if not client:
        fallback = parse_scenario_fallback(text, G)
        fallback["parsed_by"] = "keyword_fallback (No API key set)"
        applied_str = ", ".join(f"{u}-{v}" for u, v in fallback["fail_links"]) if fallback["fail_links"] else "none"
        dropped_str = ", ".join(f"{u}-{v}" for u, v in fallback["dropped_links"]) if fallback["dropped_links"] else "none"
        return fallback, 0.0, f"API credentials not set. Used keyword fallback parser. Failed links applied: {applied_str}. Dropped (not in this network): {dropped_str}."

    # Plain text prompt with zero JSON or braces
    system_prompt = (
        "You are a network scenario parser.\n"
        "Convert the request into exactly one line in this format:\n"
        "load_scale=<number>; fail_links=<a-b,c-d or none>; n_demands=<integer or default>\n"
        "Percentages are changes: increase by 50% means load_scale=1.5. Write n_demands=default unless the request states a number of demands.\n"
        "Do not output markdown formatting, code blocks, headers, or explanations."
    )

    user_prompt = f"Request: {text}"
    
    elapsed_ms = 0.0
    last_error_reason = ""
    used_model = model_p

    # Retry using backup model if attempt 1 fails or is rejected
    models_to_try = [model_p, model_e]

    for attempt, current_model in enumerate(models_to_try):
        try:
            used_model = current_model
            t0 = time.time()
            response = client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            choice = response.choices[0]
            finish_reason = getattr(choice, 'finish_reason', None)
            msg_obj = choice.message
            tool_calls = getattr(msg_obj, 'tool_calls', None)
            raw_content = getattr(msg_obj, 'content', '') or ""

            # Check if model returned tool_calls or empty content
            if (isinstance(finish_reason, str) and finish_reason == 'tool_calls') or (isinstance(tool_calls, (list, tuple)) and len(tool_calls) > 0):
                last_error_reason = "model returned empty content with tool_calls (finish_reason='tool_calls')"
                time.sleep(0.5)
                continue

            cleaned_content = clean_llm_response(raw_content)
            if not cleaned_content:
                last_error_reason = "model returned empty response content"
                time.sleep(0.5)
                continue

            parsed_res = parse_scenario_text_line(cleaned_content)
            if not parsed_res:
                last_error_reason = "model returned unparseable output"
                time.sleep(0.5)
                continue

            # Strict Validation (No Clamping!)
            # 1. load_scale validation
            if "load_scale" in parsed_res and parsed_res["load_scale"] is not None:
                raw_l = parsed_res["load_scale"]
                try:
                    flt_l = float(raw_l)
                    if flt_l < 0.25 or flt_l > 4.0:
                        last_error_reason = f"model returned load_scale={flt_l} (outside 0.25-4.0)"
                        time.sleep(0.5)
                        continue
                    else:
                        load_scale = flt_l
                except (ValueError, TypeError):
                    last_error_reason = f"model returned non-numeric load_scale={raw_l}"
                    time.sleep(0.5)
                    continue
            else:
                load_scale = 1.0

            # 2. n_demands validation
            n_demands = None
            if "n_demands" in parsed_res and parsed_res["n_demands"] is not None:
                raw_d = parsed_res["n_demands"]
                if isinstance(raw_d, str) and raw_d.lower() in ["default", "none"]:
                    n_demands = None
                else:
                    try:
                        int_d = int(raw_d)
                        if int_d < 4 or int_d > 40:
                            last_error_reason = f"model returned n_demands={int_d} (outside 4-40)"
                            time.sleep(0.5)
                            continue
                        else:
                            n_demands = int_d
                    except (ValueError, TypeError):
                        last_error_reason = f"model returned non-integer n_demands={raw_d}"
                        time.sleep(0.5)
                        continue

            # 3. Combine model-returned fail_links and user text regex fail_links (normalized & de-duplicated)
            raw_fail_links = parsed_res.get("fail_links", [])
            requested_links = []
            if isinstance(raw_fail_links, list):
                for item in raw_fail_links:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        try:
                            u, v = int(item[0]), int(item[1])
                            if u != v:
                                pair = [min(u, v), max(u, v)]
                                if pair not in requested_links:
                                    requested_links.append(pair)
                        except (ValueError, TypeError):
                            pass

            model_missed_links = False
            for pair in user_text_links:
                norm_pair = [min(pair[0], pair[1]), max(pair[0], pair[1])]
                if norm_pair not in requested_links:
                    requested_links.append(norm_pair)
                    model_missed_links = True

            # Validate requested_links against graph G (Invariant: every requested link goes to fail_links or dropped_links)
            validated_fail_links = []
            dropped_links = []

            if G is not None:
                edges_set = set((min(u, v), max(u, v)) for u, v in G.edges())
                for pair in requested_links:
                    edge = (pair[0], pair[1])
                    if edge in edges_set:
                        if pair not in validated_fail_links:
                            validated_fail_links.append(pair)
                    else:
                        if pair not in dropped_links:
                            dropped_links.append(pair)
            else:
                validated_fail_links = requested_links

            if model_missed_links:
                parsed_by_str = f"Hosted LLM ({used_model}) (model missed links; used keyword parser)"
            else:
                parsed_by_str = f"Hosted LLM ({used_model})"

            final_scenario = {
                "load_scale": load_scale,
                "n_demands": n_demands,
                "fail_links": validated_fail_links,
                "dropped_links": dropped_links,
                "seed": 42,
                "parsed_by": parsed_by_str
            }

            applied_str = ", ".join(f"{u}-{v}" for u, v in validated_fail_links) if validated_fail_links else "none"
            dropped_str = ", ".join(f"{u}-{v}" for u, v in dropped_links) if dropped_links else "none"
            status_msg = f"Parsed successfully by {used_model} in {elapsed_ms:.1f} ms. Failed links applied: {applied_str}. Dropped (not in this network): {dropped_str}."
            return final_scenario, elapsed_ms, status_msg

        except Exception as e:
            safe_err = str(e).encode('ascii', errors='ignore').decode('ascii')
            last_error_reason = f"Attempt {attempt+1} exception: {safe_err[:100]}"
            time.sleep(0.5)

    fallback = parse_scenario_fallback(text, G)
    fallback["parsed_by"] = f"keyword_fallback ({last_error_reason})"
    applied_str = ", ".join(f"{u}-{v}" for u, v in fallback["fail_links"]) if fallback["fail_links"] else "none"
    dropped_str = ", ".join(f"{u}-{v}" for u, v in fallback["dropped_links"]) if fallback["dropped_links"] else "none"
    return fallback, elapsed_ms, f"LLM parse failed: {last_error_reason}. Used fallback parser. Failed links applied: {applied_str}. Dropped (not in this network): {dropped_str}."

def explain_results(metrics: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[str, float, str]:
    """
    Explains routing results in plain English (under ~150 words) using ONLY computed numbers.
    Model: INFERENCE_CLOUD_EXPLAIN_MODEL (default Llama-3.3-70B).
    Uses plain-text prompt (NO JSON, NO braces, NO "schema"/"function").
    Strictly forbids inventing numbers or claiming quantum advantage / hardware acceleration.
    """
    client, endpoint = get_openai_client()
    explain_model = os.getenv("INFERENCE_CLOUD_EXPLAIN_MODEL", "Llama-3.3-70B")

    if not client:
        return (
            "API credentials not set. Explanation feature requires INFERENCE_CLOUD_API_KEY and INFERENCE_CLOUD_ENDPOINT.",
            0.0,
            "API key not set."
        )

    # Format numbers string for prompt
    metrics_summary = []
    for solver_name, stats in metrics.items():
        if isinstance(stats, dict):
            avg_lat = stats.get("avg_latency", 0.0)
            max_util = stats.get("max_utilisation", 0.0)
            overloads = stats.get("overloaded_links", 0)
            metrics_summary.append(
                f"- {solver_name}: Avg Latency = {avg_lat:.2f} ms, Max Util = {max_util:.2f}, Overloaded Links = {overloads}"
            )

    metrics_text = "\n".join(metrics_summary)
    load_scale = scenario.get("load_scale", 1.0)
    failed_cnt = len(scenario.get("fail_links", []))

    system_prompt = (
        "You are an expert network routing analyst.\n"
        "STRICT HONESTY RULES:\n"
        "1. Write a short plain English explanation (UNDER 150 WORDS) of why QUBO/SB route selection differs from Greedy and Static Dijkstra.\n"
        "2. Use ONLY the exact numbers provided below. NEVER invent or hallucinate numbers.\n"
        "3. NEVER claim quantum advantage or hardware acceleration of the solver.\n"
        "4. Note that the SB solver executes on CPU in PyTorch.\n"
        "5. Keep the tone technical, concise, and clear."
    )

    user_prompt = (
        f"Network Scenario: Load Multiplier = {load_scale:.2f}x, Link Failures = {failed_cnt}.\n"
        f"Computed Metrics:\n{metrics_text}\n\n"
        "Explain the routing choices based strictly on these metrics."
    )

    elapsed_ms = 0.0
    explanation = ""
    log_msg = ""

    for attempt in range(2):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=explain_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            raw_reply = response.choices[0].message.content or ""
            explanation = clean_llm_response(raw_reply)
            log_msg = f"Explained by {explain_model} in {elapsed_ms:.1f} ms."
            break
        except Exception as e:
            safe_err = str(e).encode('ascii', errors='ignore').decode('ascii')
            log_msg = f"Attempt {attempt+1} LLM explain error: {safe_err[:100]}"
            time.sleep(0.5)

    if not explanation:
        explanation = "Unable to generate explanation from LLM service at this time."

    return explanation, elapsed_ms, log_msg
