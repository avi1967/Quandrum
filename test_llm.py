from dotenv import load_dotenv
load_dotenv()

import os
import time
import requests
import pytest
from openai import OpenAI

@pytest.mark.skipif(
    not os.getenv("INFERENCE_CLOUD_API_KEY") or not os.getenv("INFERENCE_CLOUD_ENDPOINT"),
    reason="INFERENCE_CLOUD_API_KEY or INFERENCE_CLOUD_ENDPOINT not set"
)
def test_inference_cloud_llm():
    api_key = os.getenv("INFERENCE_CLOUD_API_KEY")
    endpoint = os.getenv("INFERENCE_CLOUD_ENDPOINT")
    model_override = os.getenv("INFERENCE_CLOUD_MODEL")

    print("=" * 70)
    print(" CIRRASCALE INFERENCE CLOUD / QUALCOMM CLOUD AI 100 - LLM VERIFICATION")
    print("=" * 70)

    if not api_key or not endpoint:
        print(" [WARNING] INFERENCE_CLOUD_API_KEY or INFERENCE_CLOUD_ENDPOINT not set.")
        print(" Please set them in your environment or in a local .env file.")
        print(" Example .env:")
        print("   INFERENCE_CLOUD_API_KEY=your_api_key_here")
        print("   INFERENCE_CLOUD_ENDPOINT=https://aisuite.cirrascale.com/apis/v2")
        print("=" * 70)
        pytest.skip("INFERENCE_CLOUD_API_KEY or INFERENCE_CLOUD_ENDPOINT not set")

    # Clean endpoint URL
    clean_endpoint = endpoint.strip().rstrip('/')
    if not clean_endpoint.startswith("http"):
        clean_endpoint = f"https://{clean_endpoint}"

    print(f" Target Endpoint: {clean_endpoint}")
    print(f" API Key Present: Yes")

    live_llm_models = []
    try:
        url = f"{clean_endpoint}/models"
        headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            resp_json = res.json()
            if isinstance(resp_json, dict) and "llm" in resp_json:
                live_llm_models = resp_json["llm"]
                print(f" Querying available models via /models GET...")
                print(f" Live Hosted LLMs on Qualcomm Cloud AI 100: {live_llm_models}")
    except Exception as re:
        print(f" Notice querying /models HTTP endpoint: {re}")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=clean_endpoint
        )

        # Candidates to try
        candidates = []
        if model_override:
            candidates.append(model_override)
        if live_llm_models:
            candidates.extend(live_llm_models)
        
        # Hardcoded fallback list matching Cirrascale Cloud AI 100 hosted models
        default_candidates = [
            "Llama-3.1-8B",
            "Llama-3.3-70B",
            "DeepSeek-R1-Distill-Llama-70B",
            "Qwen-QwQ-32B"
        ]
        for c in default_candidates:
            if c not in candidates:
                candidates.append(c)

        print(f"\n Testing Chat Completions across model candidates: {candidates}...")

        successful_model = None
        reply_text = None
        elapsed_time = None

        for cand in candidates:
            print(f" Attempting chat completion with model: '{cand}'...")
            try:
                t0 = time.time()
                completion = client.chat.completions.create(
                    model=cand,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant hosted on Qualcomm Cloud AI 100 via Cirrascale Inference Cloud."},
                        {"role": "user", "content": "Hello! Confirm in one short sentence that you are online."}
                    ],
                    max_tokens=60
                )
                elapsed_time = (time.time() - t0) * 1000.0
                reply_text = completion.choices[0].message.content.strip()
                successful_model = cand
                print(f" [SUCCESS] Model '{cand}' responded in {elapsed_time:.2f} ms!")
                print(f"  Model Reply: \"{reply_text}\"")
                break
            except Exception as ce:
                safe_ce = str(ce).encode('ascii', errors='ignore').decode('ascii')
                print(f"  Model '{cand}' failed: {safe_ce[:120]}")

        if successful_model:
            print("=" * 70)
            print(f" LLM VERIFICATION PASSED with model: '{successful_model}'")
            print("=" * 70)
            assert successful_model is not None, f"Expected successful model, got {successful_model}"
        else:
            print("=" * 70)
            print(" [ERROR] None of the model candidates succeeded.")
            print("=" * 70)
            assert False, "None of the model candidates succeeded"

    except Exception as e:
        safe_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
        print(f"\n [ERROR] OpenAI client initialization failed: {safe_msg}")
        print("=" * 70)
        assert False, f"OpenAI client initialization failed: {safe_msg}"

if __name__ == "__main__":
    test_inference_cloud_llm()
