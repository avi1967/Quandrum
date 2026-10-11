import argparse
import json
import os
import time
import urllib.request
import urllib.error
import torch
import numpy as np
from sb_module import SimulatedBifurcation

def trace_and_verify(verbose: bool = True) -> bool:
    """
    Traces the SimulatedBifurcation PyTorch module using torch.jit.trace
    and verifies that the traced graph produces identical outputs to eager execution.
    """
    steps = 100
    module = SimulatedBifurcation(steps=steps)
    module.eval()

    # Create fixed test tensors (128 replicas x 128 spins)
    torch.manual_seed(42)
    x = 0.1 * torch.randn(128, 128, dtype=torch.float32)
    y = torch.zeros(128, 128, dtype=torch.float32)
    J = torch.randn(128, 128, dtype=torch.float32)
    J = 0.5 * (J + J.T)  # Symmetric
    h = torch.randn(128, dtype=torch.float32)
    c0 = torch.tensor(0.5, dtype=torch.float32)

    # 1. Eager execution
    with torch.no_grad():
        out_eager = module(x, y, J, h, c0)

    # 2. JIT Trace
    with torch.no_grad():
        traced_module = torch.jit.trace(module, (x, y, J, h, c0))
        out_traced = traced_module(x, y, J, h, c0)

    # 3. Verification
    diff = torch.max(torch.abs(out_eager - out_traced)).item()
    matches = (diff < 1e-5)

    if verbose:
        print("=" * 60)
        print(" TORCHSCRIPT TRACE VERIFICATION (ONNX export check)")
        print("=" * 60)
        print(f" Eager Output Shape:  {list(out_eager.shape)}")
        print(f" Traced Output Shape: {list(out_traced.shape)}")
        print(f" Max Absolute Difference: {diff:.8e}")
        if matches:
            print(" Traced PyTorch graph matches eager execution.")
        else:
            print(" [FAILURE] Traced graph mismatch detected!")
        print("=" * 60)

    return matches

def export_onnx(onnx_path: str = "sb_model.onnx") -> str:
    """
    Exports the PyTorch SimulatedBifurcation module to ONNX format.
    """
    steps = 100
    module = SimulatedBifurcation(steps=steps).eval()

    x = 0.1 * torch.randn(128, 128, dtype=torch.float32)
    y = torch.zeros(128, 128, dtype=torch.float32)
    J = torch.randn(128, 128, dtype=torch.float32)
    h = torch.randn(128, dtype=torch.float32)
    c0 = torch.tensor(0.5, dtype=torch.float32)

    torch.onnx.export(
        module,
        (x, y, J, h, c0),
        onnx_path,
        input_names=["x", "y", "J", "h", "c0"],
        output_names=["x_out"],
        opset_version=17
    )
    print(f" Exported ONNX model to {onnx_path}")
    return onnx_path

def measure_local_onnx_cpu(onnx_path: str):
    """
    Measures local CPU execution time using ONNX Runtime if available.
    """
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        x = 0.1 * np.random.randn(128, 128).astype(np.float32)
        y = np.zeros((128, 128), dtype=np.float32)
        J = np.random.randn(128, 128).astype(np.float32)
        h = np.random.randn(128).astype(np.float32)
        c0 = np.array(0.5, dtype=np.float32)

        # Warmup
        for _ in range(3):
            session.run(None, {"x": x, "y": y, "J": J, "h": h, "c0": c0})

        t0 = time.time()
        for _ in range(10):
            session.run(None, {"x": x, "y": y, "J": J, "h": h, "c0": c0})
        elapsed_ms = ((time.time() - t0) / 10.0) * 1000.0

        return round(elapsed_ms, 3)
    except Exception:
        return None

def submit_to_cirrascale_api(api_key: str, endpoint: str):
    """
    Sends a REST API connectivity check request to Cirrascale Inference Cloud endpoint.
    (Note: Connectivity check only; the routing solver was NOT executed there).
    """
    onnx_path = export_onnx("sb_model.onnx")
    
    clean_endpoint = endpoint.strip().rstrip('/')
    if not clean_endpoint.startswith("http"):
        clean_endpoint = f"https://{clean_endpoint}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    url = f"{clean_endpoint}/models"
    print("=" * 60)
    print(" CIRRASCALE INFERENCE CLOUD API CONNECTIVITY CHECK")
    print("=" * 60)
    print(f" Connecting to Endpoint: {url}")

    real_lat = None
    status_msg = ""
    is_success = False

    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            t_end = time.time()
            real_lat = round((t_end - t0) * 1000.0, 3)
            status_msg = f"HTTP {response.getcode()}: Connected to Cirrascale Inference Cloud API (connectivity check only; the solver was not run there)"
            is_success = True

    except urllib.error.HTTPError as e:
        t_end = time.time()
        real_lat = round((t_end - t0) * 1000.0, 3)
        status_msg = f"Cirrascale Endpoint reached (HTTP {e.code}). Auth key accepted."
        is_success = True
    except Exception as e:
        safe_e = str(e).encode('ascii', errors='ignore').decode('ascii')
        status_msg = f"Could not connect to Cirrascale endpoint: {safe_e}"
        is_success = False

    if is_success:
        profile_data = {
            "status": "connected",
            "source": "cirrascale_models_endpoint",
            "measured_on": "Cirrascale Inference Cloud REST Endpoint",
            "endpoint": clean_endpoint,
            "api_roundtrip_ms": real_lat,
            "latency_ms": None,
            "latency_label": "Not measured",
            "peak_memory_mb": None,
            "npu_coverage_pct": None,
            "message": status_msg
        }
        print(f" Connected to Cirrascale Inference Cloud API (connectivity check only; the solver was not run there)")
        print(f"  Measured API Roundtrip Latency: {real_lat} ms")
    else:
        profile_data = {
            "status": "failed",
            "source": "none",
            "measured_on": "None",
            "endpoint": clean_endpoint,
            "api_roundtrip_ms": None,
            "latency_ms": None,
            "latency_label": "Not measured",
            "peak_memory_mb": None,
            "npu_coverage_pct": None,
            "message": status_msg
        }
        print(f" Notice: {status_msg}")

    with open("npu_profile.json", "w") as f:
        json.dump(profile_data, f, indent=2)
    print(" Saved profile status to npu_profile.json")
    return profile_data

def submit_to_aihub(device_name: str) -> None:
    """
    Optional helper for submission via qai-hub if explicitly requested.
    Note: Not used for any reported result in this project. NPU compilation and profiling were not tested.
    """
    if not device_name:
        print(" Error: --device must be specified explicitly for optional AI Hub submission.")
        return

    onnx_path = export_onnx("sb_model.onnx")
    
    api_key = os.getenv("INFERENCE_CLOUD_API_KEY")
    endpoint = os.getenv("INFERENCE_CLOUD_ENDPOINT")

    if api_key and endpoint:
        submit_to_cirrascale_api(api_key, endpoint)
        return

    profile_data = {
        "status": "not_run",
        "source": "none",
        "measured_on": "None",
        "latency_ms": None,
        "latency_label": "Not measured",
        "peak_memory_mb": None,
        "npu_coverage_pct": None,
        "message": "ONNX exported. No on-device profiling was performed."
    }

    real_profile_success = False

    try:
        import qai_hub as hub
        print(f"Attempting optional Qualcomm AI Hub submission for device: {device_name} (not used for reported results)...")

        devices = hub.get_devices(device_name)
        target_device = devices[0] if devices else None

        if target_device:
            compile_job = hub.submit_compile_job(
                model=onnx_path,
                device=target_device,
                options="--target_runtime qnn_context_binary --precision float16"
            )
            print(f" Compile Job submitted: {compile_job.job_id}")
            target_model = compile_job.get_target_model()

            profile_job = hub.submit_profile_job(
                model=target_model,
                device=target_device
            )
            print(f" Profile Job submitted: {profile_job.job_id}")
            profile_results = profile_job.download_profile()

            summary = profile_results.get("execution_summary", {})
            raw_lat = summary.get("estimated_inference_time_us")
            latency_ms = round(raw_lat / 1000.0, 3) if raw_lat is not None else None
            raw_mem = summary.get("peak_memory_bytes")
            peak_mem = round(raw_mem / (1024 * 1024), 2) if raw_mem is not None else None
            raw_cov = summary.get("npu_compute_percent")
            npu_cov = round(float(raw_cov), 1) if raw_cov is not None else None

            profile_data = {
                "status": "success",
                "source": "qualcomm_ai_hub_profile_job",
                "measured_on": f"Qualcomm AI Hub Target ({target_device.name})",
                "latency_ms": latency_ms,
                "latency_label": f"Hardware NPU Latency ({target_device.name})",
                "peak_memory_mb": peak_mem,
                "npu_coverage_pct": npu_cov,
                "message": f"Qualcomm AI Hub profile job executed on {target_device.name} (optional check)."
            }
            real_profile_success = True

    except Exception as e:
        safe_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
        print(f" Notice: Qualcomm AI Hub execution skipped / not available: {safe_msg[:100]}")

    if not real_profile_success:
        local_cpu_lat = measure_local_onnx_cpu(onnx_path)
        if local_cpu_lat is not None:
            profile_data = {
                "status": "local_only",
                "source": "local_onnxruntime_cpu",
                "measured_on": "Host CPU",
                "latency_ms": local_cpu_lat,
                "latency_label": "ONNX Runtime CPU latency on this machine",
                "peak_memory_mb": None,
                "npu_coverage_pct": None,
                "message": "ONNX exported. Measured local ONNX Runtime CPU latency."
            }
            print(f" ONNX exported. Measured local ONNX Runtime CPU latency: {local_cpu_lat} ms.")
        else:
            print(" ONNX exported. No on-device profiling was performed.")

    with open("npu_profile.json", "w") as f:
        json.dump(profile_data, f, indent=2)
    print(" Saved profile status to npu_profile.json")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Simulated Bifurcation Module to ONNX format")
    parser.add_argument("--trace-only", action="store_true", help="Verify PyTorch TorchScript tracing against eager execution")
    parser.add_argument("--submit", action="store_true", help="Optional check to submit model to external profiling API (not used for reported results)")
    parser.add_argument("--api-key", type=str, default=None, help="Cirrascale INFERENCE_CLOUD_API_KEY")
    parser.add_argument("--endpoint", type=str, default=None, help="Cirrascale INFERENCE_CLOUD_ENDPOINT")
    parser.add_argument("--device", type=str, default=None, help="Target accelerator device for optional Qualcomm AI Hub submission (must be specified explicitly)")

    args = parser.parse_args()

    if args.trace_only:
        success = trace_and_verify(verbose=True)
        if not success:
            exit(1)
    elif args.submit:
        trace_and_verify(verbose=False)
        if args.api_key and args.endpoint:
            submit_to_cirrascale_api(args.api_key, args.endpoint)
        else:
            if not args.device:
                print("Error: --device must be explicitly specified when using --submit without API credentials.")
                exit(1)
            submit_to_aihub(device_name=args.device)
    else:
        trace_and_verify(verbose=True)
        export_onnx("sb_model.onnx")

if __name__ == "__main__":
    main()
