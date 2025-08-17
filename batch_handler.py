import os
import json
from typing import List, Optional

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
                lines.append(
                    json.dumps(
                        {"response": {"text": getattr(r.response, "text", "")}}
                    )
                )
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

    # File-based output (JSONL)
    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None)
    if file_name:
        file_bytes: bytes = client.files.download(file=file_name)
        return file_bytes.decode("utf-8").rstrip("\n").split("\n")

    # Inline output (normalize to JSONL-style list[str])
    return _inline_responses_to_json_lines(job)


def cancel_batch_job(job_name: str):
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.cancel(name=job_name)


def delete_batch_job(job_name: str):
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.delete(name=job_name)


if __name__ == "__main__":
    # Optional quick smoke test (requires GEMINI_API_KEY)
    try:
        initialize_client()
        print("Gemini client initialized OK.")
    except Exception as e:
        print(f"Init failed: {e}")
