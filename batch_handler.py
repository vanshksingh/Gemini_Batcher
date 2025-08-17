import os
import json
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Global Client Variable ---
client: Optional[genai.Client] = None


def _normalize_model_id(model: str) -> str:
    """Ensure model id has 'models/' prefix."""
    return model if model.startswith("models/") else f"models/{model}"


def initialize_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize a global genai.Client using your preferred style.
    Priority:
      1) explicit api_key arg
      2) GEMINI_API_KEY from environment (.env supported)
    """
    global client
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY not provided. Set it in your environment or pass api_key to initialize_client()."
        )
    client = genai.Client(api_key=key)
    return client


# =======================
# ===== Batch Mode  =====
# =======================

def create_inline_batch_job(model: str, requests: List[dict], display_name: str):
    """Create a batch job from inline GenerateContentRequest dicts."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    model = _normalize_model_id(model)
    job = client.batches.create(
        model=model,
        src=requests,
        config={"display_name": display_name},
    )
    return job


def create_file_batch_job(model: str, jsonl_path: str, display_name: str):
    """Upload a JSONL file and create a batch job."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    model = _normalize_model_id(model)

    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="application/jsonl",
        ),
    )
    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    return job


def poll_batch_job_status(job_name: str):
    """Fetch the current status of a batch job (single request)."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.get(name=job_name)


def _inline_responses_to_json_lines(job) -> List[str]:
    """
    Convert inline responses to a JSON-lines-like list[str].
    Each entry is a JSON object string (response or error).
    """
    lines: List[str] = []
    dest = getattr(job, "dest", None)
    inl = getattr(dest, "inlined_responses", None)
    if not inl:
        return [json.dumps({"error": "No inline responses found."})]

    for r in inl:
        if getattr(r, "response", None):
            try:
                lines.append(r.response.to_json())  # SDK helper
            except Exception:
                # Fallback: produce a minimal JSON
                text = ""
                try:
                    # try text extraction similar to regular mode
                    text = getattr(r.response, "text", "") or ""
                except Exception:
                    pass
                lines.append(json.dumps({"response": {"text": text}}))
        elif getattr(r, "error", None):
            try:
                lines.append(r.error.to_json())  # SDK helper
            except Exception:
                lines.append(json.dumps({"error": str(r.error)}))
        else:
            lines.append(json.dumps({"warning": "Empty inline entry"}))
    return lines


def retrieve_batch_job_results(job_name: str) -> List[str]:
    """
    Retrieve results for a SUCCEEDED job.
    Always returns List[str] where each string is a JSON object per line.
    """
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")

    job = client.batches.get(name=job_name)
    state = job.state.name

    if state != "JOB_STATE_SUCCEEDED":
        payload = {"status": state}
        if getattr(job, "error", None):
            payload["error"] = str(job.error)
        return [json.dumps(payload)]

    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None)
    if file_name:
        file_bytes: bytes = client.files.download(file=file_name)
        return file_bytes.decode("utf-8").rstrip("\n").split("\n")

    return _inline_responses_to_json_lines(job)


def cancel_batch_job(job_name: str):
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.cancel(name=job_name)


def delete_batch_job(job_name: str):
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.delete(name=job_name)


# =========================
# ===== Regular Mode  =====
# =========================

def _extract_text_from_sdk(resp: Any) -> str:
    """
    Pull best-effort text directly from the SDK response object.
    Priority:
      1) resp.text
      2) resp.candidates[0].content.parts[*].text (first non-empty)
    """
    # 1) Simple
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # 2) Candidates path
    try:
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list) and cands:
            c0 = cands[0]
            content = getattr(c0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if isinstance(parts, list):
                for p in parts:
                    pt = getattr(p, "text", None)
                    if isinstance(pt, str) and pt.strip():
                        return pt.strip()
    except Exception:
        pass

    return ""


def _usage_from_sdk(resp: Any) -> Dict[str, int]:
    """
    Normalize usage/tokens from the SDK object if present.
    Returns integers with sensible defaults.
    """
    out = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "billed_tokens": 0,
    }

    um = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)
    if um is None:
        return out

    # common Gemini fields present in your log sample
    for field, target in [
        ("prompt_token_count", "input_tokens"),
        ("candidates_token_count", "output_tokens"),
        ("total_token_count", "total_tokens"),
        ("cached_content_token_count", "cached_input_tokens"),
        ("billable_token_count", "billed_tokens"),
    ]:
        val = getattr(um, field, None)
        if isinstance(val, int):
            out[target] = int(val)

    # derive total if missing
    if out["total_tokens"] == 0:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]

    return out


def _snapshot_response(resp: Any) -> Dict[str, Any]:
    """
    Build a compact, JSON-serializable snapshot from the SDK response,
    avoiding giant repr blobs.
    """
    snap: Dict[str, Any] = {}
    try:
        # Try SDK json first
        to_json = getattr(resp, "to_json", None)
        if callable(to_json):
            js = to_json()
            if isinstance(js, str):
                return json.loads(js)
            if isinstance(js, dict):
                return js
    except Exception:
        pass

    # Manual compact snapshot
    snap["model_version"] = getattr(resp, "model_version", None)
    snap["response_id"] = getattr(resp, "response_id", None)

    # candidates -> text list
    texts = []
    try:
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list):
            for c in cands:
                content = getattr(c, "content", None)
                if content and isinstance(getattr(content, "parts", None), list):
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
    except Exception:
        pass
    if texts:
        snap["candidates_text"] = texts

    # usage (compact)
    snap["usage_metadata"] = _usage_from_sdk(resp)
    return snap


def _usage_from_response_dict(resp_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract usage metadata from a *dict* response (post-normalization).
    Tries multiple field names used by Gemini APIs.
    """
    meta = resp_dict.get("usage_metadata") or resp_dict.get("usageMetadata") or {}
    keys = {
        "input_tokens": ("prompt_token_count", "input_tokens", "promptTokens"),
        "output_tokens": ("candidates_token_count", "output_tokens", "completionTokens"),
        "total_tokens": ("total_token_count", "total_tokens", "totalTokens"),
        "cached_input_tokens": ("cached_content_token_count", "cachedInputTokens", "cacheHitInputTokens"),
        "billed_tokens": ("billable_token_count", "billableTokens"),
    }
    out = {}
    for k, aliases in keys.items():
        v = 0
        for a in aliases:
            if isinstance(meta.get(a), int):
                v = meta[a]
                break
        out[k] = int(v)
    if out.get("total_tokens", 0) == 0:
        out["total_tokens"] = out.get("input_tokens", 0) + out.get("output_tokens", 0)
    # ensure all keys
    for k in ["input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "billed_tokens"]:
        out.setdefault(k, 0)
    return out


def generate_content_regular(
    model: str,
    user_texts: List[str],
    implicit_prefix: Optional[str] = None,
    cache_name: Optional[str] = None,
    retries: int = 2,
    retry_backoff: float = 1.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Run normal (non-batch) GenerateContent for a list of user prompts.

    Returns:
      - list of normalized dicts with fields: prompt, text, response (compact), usage_metadata
      - aggregate usage dict (sum across calls)
    """
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    model = _normalize_model_id(model)

    results: List[Dict[str, Any]] = []
    agg_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "billed_tokens": 0,
    }

    for prompt in user_texts:
        contents = [{"role": "user", "parts": []}]
        if implicit_prefix:
            contents[0]["parts"].append({"text": implicit_prefix})
        contents[0]["parts"].append({"text": prompt})

        config = {}
        if cache_name:
            config["cached_content"] = cache_name

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config or None,
                )

                # Extract text & usage straight from SDK object
                text_out = _extract_text_from_sdk(resp)
                usage_obj = _usage_from_sdk(resp)

                # Also keep a compact response snapshot for "View Raw"
                resp_snapshot = _snapshot_response(resp)

                # Build normalized record
                record = {
                    "prompt": prompt,
                    "text": text_out or "",
                    "usage_metadata": usage_obj,
                    "response": resp_snapshot,  # compact (JSON-serializable)
                }
                results.append(record)

                # Aggregate usage
                for k in agg_usage:
                    agg_usage[k] += usage_obj.get(k, 0)

                break  # success
            except Exception as e:
                last_err = e
                if attempt < retries:
                    import time as _t
                    _t.sleep((retry_backoff ** attempt))
                else:
                    results.append({"prompt": prompt, "error": str(last_err), "text": "", "usage_metadata": {}})

    return results, agg_usage


# ==========================
# ===== Cache helpers   ====
# ==========================

def create_context_cache(
    model: str,
    context_text: str,
    ttl_seconds: int = 3600,
    display_name: Optional[str] = None,
    system_instruction: Optional[str] = None,
):
    """
    Create an explicit context cache and return the cache object.
    The cache is tied to the provided model. You can include an optional system instruction.
    """
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    model = _normalize_model_id(model)

    contents = [
        {"role": "user", "parts": [{"text": context_text}]}
    ]

    cfg = {
        "display_name": display_name or "batch-context-cache",
        "model": model,
        "ttl": f"{int(ttl_seconds)}s",
        "contents": contents,
    }

    if system_instruction and system_instruction.strip():
        cfg["system_instruction"] = {"parts": [{"text": system_instruction.strip()}]}

    cache = client.caches.create(**cfg)
    return cache


def delete_context_cache(cache_name: str):
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.caches.delete(name=cache_name)


def augment_requests_with_cached_content(requests: List[dict], cache_name: str) -> List[dict]:
    """Add config.cached_content to each inline batch request."""
    out: List[dict] = []
    for req in requests:
        r = json.loads(json.dumps(req))  # deep copy
        cfg = r.get("config", {})
        cfg["cached_content"] = cache_name
        r["config"] = cfg
        out.append(r)
    return out


def inject_cached_content_into_jsonl_file(src_jsonl: str, dst_jsonl: str, cache_name: str):
    """
    Add request.config.cached_content to every line in a JSONL file
    that has {"key": "...", "request": {...}} structure.
    """
    with open(src_jsonl, "r", encoding="utf-8") as fin, open(dst_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            req = obj.setdefault("request", {})
            cfg = req.get("config", {})
            cfg["cached_content"] = cache_name
            req["config"] = cfg
            obj["request"] = req
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Optional smoke test (requires GEMINI_API_KEY)
    try:
        initialize_client()
        print("Gemini client initialized OK.")
    except Exception as e:
        print(f"Init failed: {e}")
