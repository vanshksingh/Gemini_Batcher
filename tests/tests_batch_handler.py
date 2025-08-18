# tests/tests_batch_handler.py
# -------------------------------------------------------------------
# Final test suite with robust import bootstrap.
# - Default TARGET_MODULE -> "gemini_batcher.batch_handler"
# - You can override with: TARGET_MODULE=your.module.path pytest ...
# - Works from nested tests/ dirs (adds multiple parents to sys.path)
# -------------------------------------------------------------------

import os
import sys
import json
import types as pytypes
import importlib
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

# =========================
# --- Import bootstrap  ---
# =========================
TEST_FILE = pathlib.Path(__file__).resolve()

# Add a few ancestor directories to sys.path so the package can be found
for up in (1, 2, 3, 4, 5):
    p = TEST_FILE.parents[up] if len(TEST_FILE.parents) > up else None
    if p and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Allow env override, default to your likely module path
MODULE_NAME = os.getenv("TARGET_MODULE", "gemini_batcher.batch_handler")


# ---------------------------
# Minimal fake SDK structures
# ---------------------------

@dataclass
class FakeUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    cached_content_token_count: int = 0
    billable_token_count: int = 0


class FakeResp:
    """Fake response object that mimics google.genai generate_content result."""
    def __init__(
        self,
        text: str = "",
        usage: Optional[FakeUsage] = None,
        candidates_texts: Optional[List[str]] = None,
        model_version: str = "mv1",
        response_id: str = "rid-123",
    ):
        self.text = text
        self.usage_metadata = usage or FakeUsage(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            cached_content_token_count=2,
            billable_token_count=13,
        )
        self.model_version = model_version
        self.response_id = response_id
        self._candidates_texts = candidates_texts or []

        # Build optional candidates/content.parts structure
        if not text and self._candidates_texts:
            Part = lambda t: pytypes.SimpleNamespace(text=t)
            Content = lambda parts: pytypes.SimpleNamespace(parts=parts)
            Cand = lambda parts: pytypes.SimpleNamespace(content=Content(parts))
            self.candidates = [Cand([Part(t) for t in self._candidates_texts])]

    def to_json(self):
        # Provide a compact json that your code accepts
        return json.dumps(
            {
                "response_id": self.response_id,
                "model_version": self.model_version,
                "text": self.text,
                "usage_metadata": {
                    "prompt_token_count": self.usage_metadata.prompt_token_count,
                    "candidates_token_count": self.usage_metadata.candidates_token_count,
                    "total_token_count": self.usage_metadata.total_token_count,
                    "cached_content_token_count": self.usage_metadata.cached_content_token_count,
                    "billable_token_count": self.usage_metadata.billable_token_count,
                },
            }
        )


class FakeErrorObj:
    def __init__(self, msg: str):
        self.message = msg

    def to_json(self):
        return json.dumps({"error": self.message})


class FakeInlineEntry:
    def __init__(self, text: Optional[str] = None, error: Optional[str] = None):
        self.response = None
        self.error = None
        if text is not None:
            self.response = FakeResp(text=text)
        if error is not None:
            self.error = FakeErrorObj(error)


class FakeDest:
    def __init__(self, file_name: Optional[str] = None, inlined_responses: Optional[List[FakeInlineEntry]] = None):
        self.file_name = file_name
        self.inlined_responses = inlined_responses or []


class FakeState:
    def __init__(self, name: str):
        self.name = name  # e.g., "JOB_STATE_RUNNING" / "JOB_STATE_SUCCEEDED"


class FakeBatchJob:
    def __init__(
        self,
        name: str,
        state: str = "JOB_STATE_RUNNING",
        dest: Optional[FakeDest] = None,
        error: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        self.name = name
        self.state = FakeState(state)
        self.dest = dest
        self.error = error
        self.display_name = display_name


@dataclass
class FakeUploadedFile:
    name: str


# ------------------
# Fake Client & SDK
# ------------------

class FakeBatches:
    def __init__(self, client):
        self._client = client
        self._jobs: Dict[str, FakeBatchJob] = {}
        self._counter = 0

    def create(self, model: str, src: Any, config: Optional[dict] = None):
        self._counter += 1
        name = f"batches/auto-{self._counter}"
        job = FakeBatchJob(name=name, state="JOB_STATE_RUNNING", display_name=(config or {}).get("display_name"))
        self._jobs[name] = job
        return job

    def get(self, name: str):
        return self._jobs[name]

    def cancel(self, name: str):
        job = self._jobs[name]
        job.state = FakeState("JOB_STATE_CANCELLED")
        return job

    def delete(self, name: str):
        return self._jobs.pop(name, None)


class FakeFiles:
    def __init__(self):
        self._store: Dict[str, bytes] = {}
        self._up_counter = 0

    def upload(self, file: str, config: Any = None):
        self._up_counter += 1
        upname = f"files/upload-{self._up_counter}"
        return FakeUploadedFile(name=upname)

    def download(self, file: str) -> bytes:
        return self._store.get(file, b"")


class FakeCaches:
    def __init__(self):
        self._items: Dict[str, dict] = {}
        self._c = 0

    def create(self, **cfg):
        self._c += 1
        name = f"caches/ctx-{self._c}"
        item = {"name": name, **cfg}
        self._items[name] = item
        return item

    def delete(self, name: str):
        return self._items.pop(name, None)


class FakeModels:
    def __init__(self):
        self._calls = 0
        self._fail_first = False
        self._planned_responses: List[FakeResp] = []

    def set_fail_first(self, flag: bool):
        self._fail_first = flag

    def plan(self, responses: List[FakeResp]):
        self._planned_responses = responses[:]

    def generate_content(self, model: str, contents: List[dict], config: Optional[dict] = None):
        self._calls += 1
        if self._fail_first and self._calls == 1:
            raise RuntimeError("Transient failure")
        if self._planned_responses:
            return self._planned_responses.pop(0)
        return FakeResp(text="ok")


class FakeGenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.batches = FakeBatches(self)
        self.files = FakeFiles()
        self.caches = FakeCaches()
        self.models = FakeModels()


# -------------------------
# Pytest fixtures & config
# -------------------------

@pytest.fixture(autouse=True)
def reload_module(monkeypatch):
    """
    Reload the target module fresh for each test and wire in fakes:
      - google.genai.Client -> FakeGenAIClient
      - types.UploadFileConfig -> accept any kwargs
    """
    # Import (or re-import) the target module
    m = importlib.import_module(MODULE_NAME)
    importlib.reload(m)

    # Stub load_dotenv to no-op (we'll control env explicitly)
    monkeypatch.setattr(m, "load_dotenv", lambda: None, raising=True)

    # Replace google.genai.Client in the already-imported module namespace
    monkeypatch.setattr(m.genai, "Client", FakeGenAIClient, raising=True)

    # Stub UploadFileConfig to accept kwargs
    class _UploadCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(m.types, "UploadFileConfig", _UploadCfg, raising=True)

    # Ensure global client is None at start
    monkeypatch.setattr(m, "client", None, raising=True)
    return m


@pytest.fixture
def mod(reload_module):
    return reload_module


# -------------------
# Unit test coverage
# -------------------

def test_normalize_model_id(mod):
    f = mod._normalize_model_id
    assert f("models/gemini-2.5-flash") == "models/gemini-2.5-flash"
    assert f("gemini-2.0-pro") == "models/gemini-2.0-pro"


def test_initialize_client_with_env_success(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    c = mod.initialize_client()
    # Instance of our fake
    assert isinstance(c, FakeGenAIClient)
    assert mod.client is c
    assert c.api_key == "abc123"


def test_initialize_client_with_arg_overrides_env(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    c = mod.initialize_client(api_key="arg-key")
    assert c.api_key == "arg-key"


def test_initialize_client_raises_without_key(mod, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        mod.initialize_client()


def test_create_inline_batch_job_and_poll(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod.initialize_client()

    job = mod.create_inline_batch_job("gemini-2.5-flash", [{"fake": "req"}], "my-batch")
    assert job.name.startswith("batches/auto-")
    assert job.display_name == "my-batch"

    got = mod.poll_batch_job_status(job.name)
    assert got is job


def test_create_file_batch_job_and_retrieve_succeeded_file(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c = mod.initialize_client()

    # Create file-based batch
    job = mod.create_file_batch_job("gemini-2.5-flash", str(tmp_path / "in.jsonl"), "file-batch")

    # Simulate SUCCEEDED with a dest file
    file_name = "files/result-1.jsonl"
    lines = [json.dumps({"response": {"text": "A"}}), json.dumps({"response": {"text": "B"}})]
    c.files._store[file_name] = ("\n".join(lines) + "\n").encode("utf-8")
    job.dest = FakeDest(file_name=file_name)
    job.state = FakeState("JOB_STATE_SUCCEEDED")

    out_lines = mod.retrieve_batch_job_results(job.name)
    assert out_lines == lines


def test_retrieve_succeeded_inline_responses(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod.initialize_client()

    # Create inline job and then mark as succeeded with inline payloads
    job = mod.create_inline_batch_job("gemini-2.5-flash", [{"a": 1}], "inline")
    job.state = FakeState("JOB_STATE_SUCCEEDED")
    job.dest = FakeDest(
        file_name=None,
        inlined_responses=[FakeInlineEntry(text="Hello"), FakeInlineEntry(error="Boom")],
    )

    lines = mod.retrieve_batch_job_results(job.name)
    parsed = [json.loads(x) for x in lines]
    assert parsed[0]["text"] == "Hello" or parsed[0].get("response", {}).get("text") == "Hello"
    assert "error" in parsed[1]


def test_retrieve_non_succeeded_returns_status(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod.initialize_client()

    job = mod.create_inline_batch_job("gemini-2.5-flash", [{"a": 1}], "inline")
    job.state = FakeState("JOB_STATE_RUNNING")

    lines = mod.retrieve_batch_job_results(job.name)
    info = json.loads(lines[0])
    assert info["status"] == "JOB_STATE_RUNNING"


def test_cancel_and_delete_batch_job(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod.initialize_client()

    job = mod.create_inline_batch_job("gemini-2.5-flash", [{"a": 1}], "x")
    cancelled = mod.cancel_batch_job(job.name)
    assert cancelled.state.name == "JOB_STATE_CANCELLED"

    deleted = mod.delete_batch_job(job.name)
    assert deleted is not None


def test_generate_content_regular_happy_path_aggregates_usage(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c = mod.initialize_client()

    # plan two responses with distinct usage
    c.models.plan(
        [
            FakeResp(text="T1", usage=FakeUsage(10, 5, 15, 2, 13)),
            FakeResp(text="T2", usage=FakeUsage(7, 3, 10, 1, 9)),
        ]
    )

    results, agg = mod.generate_content_regular(
        model="gemini-2.0-pro",
        user_texts=["p1", "p2"],
        implicit_prefix="CTX:",
        cache_name="caches/ctx-1",
    )

    assert [r["text"] for r in results] == ["T1", "T2"]
    assert agg["input_tokens"] == 17
    assert agg["output_tokens"] == 8
    assert agg["total_tokens"] == 25
    assert agg["cached_input_tokens"] == 3
    assert agg["billed_tokens"] == 22


def test_generate_content_regular_retries_on_failure(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c = mod.initialize_client()

    # fail first call, then succeed
    c.models.set_fail_first(True)
    c.models.plan([FakeResp(text="OK after retry")])

    results, agg = mod.generate_content_regular(
        model="gemini-2.5-flash",
        user_texts=["hello"],
        retries=1,
        retry_backoff=0.0,  # avoid sleep if executed
    )

    assert results[0]["text"] == "OK after retry"
    assert agg["total_tokens"] == 15


def test_create_and_delete_context_cache(mod, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    mod.initialize_client()

    cache = mod.create_context_cache(
        model="gemini-2.0-pro",
        context_text="Knowledge base content",
        ttl_seconds=42,
        display_name="dn",
        system_instruction="act wise",
    )
    assert cache["name"].startswith("caches/ctx-")
    assert cache["ttl"] == "42s"
    assert cache["model"] == "models/gemini-2.0-pro"

    deleted = mod.delete_context_cache(cache["name"])
    assert deleted is not None


def test_augment_requests_with_cached_content(mod):
    reqs = [
        {"model": "models/g", "contents": [{"role": "user", "parts": [{"text": "a"}]}]},
        {"model": "models/g", "contents": [{"role": "user", "parts": [{"text": "b"}]}], "config": {"x": 1}},
    ]
    out = mod.augment_requests_with_cached_content(reqs, "caches/ctx-9")
    for r in out:
        assert r["config"]["cached_content"] == "caches/ctx-9"
    # ensure deep copy (not the same dict object)
    assert "cached_content" not in (reqs[0].get("config") or {})


def test_inject_cached_content_into_jsonl_file(mod, tmp_path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    lines = [
        json.dumps({"key": "1", "request": {"model": "models/m", "config": {"a": 1}}}),
        json.dumps({"key": "2", "request": {"model": "models/m"}}),
        json.dumps({"key": "3"}),  # even if request missing, code sets default
    ]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mod.inject_cached_content_into_jsonl_file(str(src), str(dst), "caches/ctx-77")
    out = [json.loads(l) for l in dst.read_text(encoding="utf-8").strip().split("\n")]

    assert out[0]["request"]["config"]["cached_content"] == "caches/ctx-77"
    assert out[1]["request"]["config"]["cached_content"] == "caches/ctx-77"
    assert out[2]["request"]["config"]["cached_content"] == "caches/ctx-77"
