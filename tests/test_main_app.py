# test_main_app.py
# ---------------------------------------------------------------------
# Pytest suite for the provided Streamlit app.
# This version fixes "must be absolute import path string" by stubbing
# required modules via sys.modules BEFORE importing `main`.
#
# To run:
#   pytest -q test_main_app.py
# ---------------------------------------------------------------------

import sys
import json
from types import ModuleType, SimpleNamespace
import pytest

# --------------------------- Shared Test Helpers ---------------------------

class FakeCookies:
    """Minimal cookie manager API used by the app."""
    def __init__(self, data=None, ready=True):
        self._data = dict(data or {})
        self._ready = ready
        self._saved = False

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        self._data[key] = val

    def __delitem__(self, key):
        if key in self._data:
            del self._data[key]

    def save(self):
        self._saved = True

    def ready(self):
        return self._ready


class StopCalled(Exception):
    """Raised when the app triggers st.stop() or st.rerun()."""


# ----------------------------- Stub Modules ------------------------------

def _install_streamlit_stub():
    """Create a tiny 'streamlit' module that matches APIs used by the app."""
    st_mod = ModuleType("streamlit")

    # Session state and simple collectors for assertions
    st_mod.session_state = {}
    st_mod._warnings = []
    st_mod._errors = []
    st_mod._toasts = []
    st_mod._titles = []
    st_mod._headers = []
    st_mod._captions = []
    st_mod._jsons = []
    st_mod._codes = []
    st_mod._download_payloads = []
    st_mod._metrics = []

    # Config
    def set_page_config(*_a, **_k):  # must exist as first call
        return None
    st_mod.set_page_config = set_page_config

    # Core UI used in tests
    st_mod.title = lambda txt: st_mod._titles.append(txt)
    st_mod.header = lambda txt: st_mod._headers.append(txt)
    st_mod.subheader = lambda txt: st_mod._headers.append(txt)
    st_mod.caption = lambda txt: st_mod._captions.append(txt)
    st_mod.json = lambda obj: st_mod._jsons.append(obj)
    st_mod.code = lambda txt, language=None: st_mod._codes.append((txt, language))
    st_mod.metric = lambda label, value: st_mod._metrics.append((label, value))
    st_mod.toast = lambda msg: st_mod._toasts.append(msg)

    # Inputs / widgets
    st_mod.text_input = lambda *a, **k: k.get("value") or ""
    st_mod.text_area = lambda *a, **k: k.get("value") or ""
    st_mod.number_input = lambda *a, **k: k.get("value", 1)
    st_mod.selectbox = lambda *a, **k: (k.get("options") or ["x"])[0]
    st_mod.radio = lambda label, options, **_k: options[0]
    st_mod.button = lambda *a, **k: False
    st_mod.file_uploader = lambda *a, **k: None

    def download_button(**_k):
        st_mod._download_payloads.append(True)
    st_mod.download_button = download_button

    st_mod.divider = lambda: None
    st_mod.info = lambda *a, **k: None
    st_mod.warning = lambda msg: st_mod._warnings.append(msg)
    st_mod.error = lambda msg: st_mod._errors.append(msg)
    st_mod.markdown = lambda *a, **k: None

    def columns(n):
        return [st_mod for _ in range(n)]
    st_mod.columns = columns

    # popover/expander/form/spinner context managers
    class _Ctx:
        def __enter__(self): return st_mod
        def __exit__(self, *a): return False

    st_mod.popover = lambda *a, **k: _Ctx()
    st_mod.expander = lambda *a, **k: _Ctx()

    class _Form:
        def __enter__(self): return st_mod
        def __exit__(self, exc_type, exc, tb): return False
    st_mod.form = lambda *a, **k: _Form()
    st_mod.form_submit_button = lambda *a, **k: False

    class _Spin:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    st_mod.spinner = lambda *a, **k: _Spin()

    # Cache decorators
    def _identity_cache_decorator(*_a, **_k):
        def _wrap(fn): return fn
        return _wrap
    st_mod.cache_resource = _identity_cache_decorator
    st_mod.cache_data = _identity_cache_decorator

    # Flow control
    st_mod.stop = lambda: (_ for _ in ()).throw(StopCalled("st.stop() invoked"))
    st_mod.rerun = lambda: (_ for _ in ()).throw(StopCalled("st.rerun() invoked"))

    # Sidebar proxy
    class _Side:
        def __getattr__(self, name): return getattr(st_mod, name)
        def __call__(self, *a, **k): return st_mod
        def __enter__(self): return st_mod
        def __exit__(self, *a): return False
    st_mod.sidebar = _Side()

    sys.modules["streamlit"] = st_mod
    return st_mod


def _install_cookies_stub():
    m = ModuleType("streamlit_cookies_manager")

    class EncryptedCookieManager:
        def __init__(self, *a, **k):
            self._cookies = FakeCookies()
        # Delegate API to FakeCookies
        def get(self, *a, **k): return self._cookies.get(*a, **k)
        def __getitem__(self, k): return self._cookies[k]
        def __setitem__(self, k, v): self._cookies[k] = v
        def __delitem__(self, k): del self._cookies[k]
        def save(self): return self._cookies.save()
        def ready(self): return self._cookies.ready()

    class CookieManager(EncryptedCookieManager):
        pass

    m.EncryptedCookieManager = EncryptedCookieManager
    m.CookieManager = CookieManager
    sys.modules["streamlit_cookies_manager"] = m
    return m


def _install_dotenv_stub():
    m = ModuleType("dotenv")
    m.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = m
    return m


def _install_google_exceptions_stub():
    # Build package path: google -> google.api_core -> google.api_core.exceptions
    g = sys.modules.setdefault("google", ModuleType("google"))
    api_core = ModuleType("google.api_core")
    exceptions = ModuleType("google.api_core.exceptions")

    class GoogleAPIError(Exception): ...
    class GoogleAPICallError(Exception): ...
    class ResourceExhausted(Exception): ...

    exceptions.GoogleAPIError = GoogleAPIError
    exceptions.GoogleAPICallError = GoogleAPICallError
    exceptions.ResourceExhausted = ResourceExhausted

    sys.modules["google.api_core"] = api_core
    sys.modules["google.api_core.exceptions"] = exceptions
    return exceptions


def _install_batch_handler_stub():
    m = ModuleType("batch_handler")

    class FakeClient: ...
    class FakeJob:
        def __init__(self, name="jobs/123", state="JOB_STATE_RUNNING"):
            self.name = name
            self.state = SimpleNamespace(name=state)

    # Globals
    m.client = None

    # API used by the app
    def initialize_client(api_key: str):
        m.client = FakeClient()
        return m.client

    def create_context_cache(**_k):
        return SimpleNamespace(name="cache/abc")

    def delete_context_cache(_name): return None

    def augment_requests_with_cached_content(reqs, _cache_name): return reqs

    def inject_cached_content_into_jsonl_file(_in_path, _out_path, _cache_name): return None

    def create_inline_batch_job(_model, _reqs, _name):
        return FakeJob(name="jobs/inline", state="JOB_STATE_SUCCEEDED")

    def create_file_batch_job(_model, _path, _name):
        return FakeJob(name="jobs/file", state="JOB_STATE_SUCCEEDED")

    def poll_batch_job_status(name):
        return FakeJob(name=name, state="JOB_STATE_SUCCEEDED")

    def retrieve_batch_job_results(_name):
        return [
            json.dumps({"response": {"text": "hello"}}),
            json.dumps({"candidates": [{"content": {"parts": [{"text": "world"}]}}]}),
        ]

    def cancel_batch_job(_name): return None
    def delete_batch_job(_name): return None

    def generate_content_regular(**_k):
        return (
            [
                {"prompt": "p1", "text": "a1", "response": {"text": "a1"}},
                {"prompt": "p2", "text": "a2", "response": {"text": "a2"}},
            ],
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cached_input_tokens": 4, "billed_tokens": 11},
        )

    # Assign
    m.initialize_client = initialize_client
    m.create_context_cache = create_context_cache
    m.delete_context_cache = delete_context_cache
    m.augment_requests_with_cached_content = augment_requests_with_cached_content
    m.inject_cached_content_into_jsonl_file = inject_cached_content_into_jsonl_file
    m.create_inline_batch_job = create_inline_batch_job
    m.create_file_batch_job = create_file_batch_job
    m.poll_batch_job_status = poll_batch_job_status
    m.retrieve_batch_job_results = retrieve_batch_job_results
    m.cancel_batch_job = cancel_batch_job
    m.delete_batch_job = delete_batch_job
    m.generate_content_regular = generate_content_regular

    sys.modules["batch_handler"] = m
    return m


# ------------------------------- Fixtures --------------------------------

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None, raising=False)


@pytest.fixture
def import_app(monkeypatch):
    """Install stubs and import `main` fresh for each test."""
    # Ensure a clean slate before each import
    for name in list(sys.modules):
        # Do NOT aggressively purge stdlib or pytest internals;
        # only remove our prior stubs to allow re-creation.
        if name in {
            "streamlit",
            "streamlit_cookies_manager",
            "dotenv",
            "google.api_core",
            "google.api_core.exceptions",
            "batch_handler",
        }:
            del sys.modules[name]

    st = _install_streamlit_stub()
    _install_cookies_stub()
    _install_dotenv_stub()
    _install_google_exceptions_stub()
    _install_batch_handler_stub()

    # Import the app AFTER stubs are in place
    import importlib
    app = importlib.import_module("main")
    # Expose streamlit stub on app for assertions
    app.st = sys.modules["streamlit"]
    return app


# ------------------------------- Unit Tests -------------------------------

def test_extract_best_text_from_obj_variants(import_app):
    app = import_app
    assert app._extract_best_text_from_obj({"response": {"text": "alpha"}}) == "alpha"
    assert app._extract_best_text_from_obj({"response": {"candidates": [{"content": {"parts": [{"text": "beta"}]}}]}}) == "beta"
    assert app._extract_best_text_from_obj({"candidates": [{"content": {"parts": [{"text": "gamma"}]}}]}) == "gamma"
    assert app._extract_best_text_from_obj({"content": {"parts": [{"text": "delta"}]}}) == "delta"
    assert app._extract_best_text_from_obj({"text": "epsilon"}) == "epsilon"
    assert app._extract_best_text_from_obj({"foo": "bar"}) is None


def test_extract_texts_for_report_mixed(import_app):
    app = import_app
    lines = [
        json.dumps({"response": {"text": "one"}}),
        json.dumps({"candidates": [{"content": {"parts": [{"text": "two"}]}}]}),
        json.dumps({"error": "rate limit"}),
        "{not json}",
    ]
    out = app._extract_texts_for_report(lines)
    assert "--- Response 1 ---" in out and "one" in out
    assert "--- Response 2 ---" in out and "two" in out
    assert "(ERROR)" in out
    assert "[RAW]" in out


def test_regular_report_and_txt(import_app):
    app = import_app
    results = [{"prompt": "Q1", "text": "A1"}, {"prompt": "Q2", "text": "A2"}]
    usage = {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16, "cached_input_tokens": 3, "billed_tokens": 13}
    report, combined = app._regular_report_and_txt(results, usage, implicit_prefix="PREFIX")
    assert "Total prompts: 2" in report
    assert "Input tokens (sum): 10" in report
    assert "Cached input tokens (reported): 3" in report
    assert "Billable tokens (reported): 13" in report
    assert "Implicit shared prefix used: True" in report
    assert "=== Q&A Pairs ===" in combined
    assert "Q:\nQ1" in combined and "A:\nA1" in combined


def test_cookie_history_helpers(import_app):
    app = import_app
    cookies = FakeCookies()

    # Initially empty
    assert app.get_job_history(cookies) == []

    # Save history
    hist = ["jobs/1", "jobs/2"]
    app.save_job_history(cookies, hist)
    assert json.loads(cookies.get(app.COOKIE_JOB_HISTORY)) == hist

    # Add item
    app.add_to_job_history(cookies, "jobs/3")
    loaded = app.get_job_history(cookies)
    assert loaded[0] == "jobs/3"
    assert "jobs/2" in loaded

    # Clear one
    app.clear_history_item(cookies, "jobs/2")
    loaded2 = app.get_job_history(cookies)
    assert "jobs/2" not in loaded2


def test_init_state_sets_defaults(import_app):
    app = import_app
    app.st.session_state.clear()
    app.init_state()
    for k in [
        app.STATE_JOB_NAME,
        app.STATE_JOB_STATUS,
        app.STATE_JOB_RESULTS,
        app.STATE_API_KEY,
        app.STATE_LAST_REQUEST,
        app.STATE_CACHE_NAME,
        app.STATE_IMPLICIT_PREFIX,
        app.STATE_CTX_MODE,
        app.STATE_REGULAR_RESULTS,
        app.STATE_REGULAR_USAGE,
    ]:
        assert k in app.st.session_state


def test_reset_app_state_clears_and_reruns(import_app):
    app = import_app
    cookies = FakeCookies({app.COOKIE_JOB_HISTORY: json.dumps(["jobs/keep", "jobs/remove"])})

    app.st.session_state[app.STATE_JOB_NAME] = "jobs/remove"
    app.st.session_state[app.STATE_JOB_STATUS] = "X"
    app.st.session_state[app.STATE_JOB_RESULTS] = ["Y"]
    app.st.session_state[app.STATE_LAST_REQUEST] = {"Z": 1}
    app.st.session_state[app.STATE_REGULAR_RESULTS] = ["R"]
    app.st.session_state[app.STATE_REGULAR_USAGE] = {"U": 1}

    with pytest.raises(StopCalled):
        app.reset_app_state(cookies, clear_history=False)

    hist = json.loads(cookies.get(app.COOKIE_JOB_HISTORY))
    assert "jobs/remove" not in hist and "jobs/keep" in hist

    for k in [
        app.STATE_JOB_NAME,
        app.STATE_JOB_STATUS,
        app.STATE_JOB_RESULTS,
        app.STATE_LAST_REQUEST,
        app.STATE_REGULAR_RESULTS,
        app.STATE_REGULAR_USAGE,
    ]:
        assert app.st.session_state[k] is None


# ------------------------------- Smoke Paths -------------------------------

def test_run_app_stops_without_api_key(import_app):
    app = import_app
    app.st.session_state.clear()
    app.init_state()
    with pytest.raises(StopCalled):
        app.run_app()
    assert any("Provide an API key" in w for w in app.st._warnings)


def test_run_app_regular_mode_success_flow(import_app):
    app = import_app
    # Provide API key in session to bypass stop
    app.st.session_state.clear()
    app.init_state()
    app.st.session_state[app.STATE_API_KEY] = "DUMMY"

    # Force "Regular" for Run Mode and submit the form with 2 prompts
    original_radio = app.st.radio
    original_form_submit_button = app.st.form_submit_button
    original_text_area = app.st.text_area

    def radio(label, options, **k):
        if label == "Run Mode":
            return "Regular"
        return original_radio(label, options, **k)

    def form_submit_button(*a, **k):
        return True  # pretend user pressed submit

    def text_area(label, **k):
        if "prompts" in label.lower():
            return "p1\np2"
        return original_text_area(label, **k)

    app.st.radio = radio
    app.st.form_submit_button = form_submit_button
    app.st.text_area = text_area

    # Execute
    app.run_app()

    # We should have produced at least one downloadable artifact
    assert len(app.st._download_payloads) >= 1

    # Restore (good hygiene)
    app.st.radio = original_radio
    app.st.form_submit_button = original_form_submit_button
    app.st.text_area = original_text_area
