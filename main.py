import os
import json
import time
import random
import streamlit as st

# --- set_page_config MUST be the first Streamlit call ---
st.set_page_config(
    page_title="Gemini Batch Runner Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError, GoogleAPICallError, ResourceExhausted

from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_cookies_manager import CookieManager as PlainCookieManager

# Import module so we can set its global client
import batch_handler as bh

# ---------------- Constants / Keys ----------------
COOKIE_JOB_HISTORY = "gemini_job_history"
COOKIE_API_KEY = "gemini_api_key"

STATE_JOB_NAME = "job_name"
STATE_JOB_STATUS = "job_status"
STATE_JOB_RESULTS = "job_results"
STATE_API_KEY = "api_key"
STATE_LAST_REQUEST = "last_request"
STATE_CACHE_NAME = "cache_name"             # explicit cache name
STATE_IMPLICIT_PREFIX = "implicit_prefix"   # implicit shared prefix
STATE_CTX_MODE = "ctx_mode"                 # None | Implicit shared prefix | Explicit cache

# Regular mode state
STATE_REGULAR_RESULTS = "regular_results"
STATE_REGULAR_USAGE = "regular_usage"

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

DEFAULT_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-pro-latest",
]


# ---------------- Cache (new API) ----------------
@st.cache_resource(show_spinner=False)
def _cached_initialize_client(api_key: str):
    return bh.initialize_client(api_key)


@st.cache_data(show_spinner=False)
def _to_jsonl_blob(lines: list[str]) -> str:
    return "\n".join(lines)


# ---------- Common result parsing for batch-only consolidated file ----------
def _extract_best_text_from_obj(obj: dict) -> str | None:
    resp = obj.get("response")
    if isinstance(resp, dict):
        t = resp.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()
        cand = resp.get("candidates")
        if isinstance(cand, list) and cand:
            c0 = cand[0] or {}
            content = c0.get("content", {})
            parts = content.get("parts", [])
            for p in parts:
                if isinstance(p, dict):
                    pt = p.get("text")
                    if isinstance(pt, str) and pt.strip():
                        return pt.strip()
    cand2 = obj.get("candidates")
    if isinstance(cand2, list) and cand2:
        c0 = cand2[0] or {}
        content = c0.get("content", {})
        parts = content.get("parts", [])
        for p in parts:
            if isinstance(p, dict):
                pt = p.get("text")
                if isinstance(pt, str) and pt.strip():
                    return pt.strip()
    content2 = obj.get("content", {})
    if isinstance(content2, dict):
        parts2 = content2.get("parts", [])
        for p in parts2:
            if isinstance(p, dict):
                pt = p.get("text")
                if isinstance(pt, str) and pt.strip():
                    return pt.strip()
    t2 = obj.get("text")
    if isinstance(t2, str) and t2.strip():
        return t2.strip()
    return None


@st.cache_data(show_spinner=False)
def _extract_texts_for_report(lines: list[str]) -> str:
    chunks = []
    for i, line in enumerate(lines, start=1):
        line = (line or "").strip()
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            chunks.append(f"--- Response {i} ---\n[RAW]\n{line}\n")
            continue
        txt = _extract_best_text_from_obj(obj)
        if txt:
            chunks.append(f"--- Response {i} ---\n{txt}\n")
        elif "error" in obj:
            chunks.append(f"--- Response {i} (ERROR) ---\n{obj.get('error')}\n")
        else:
            chunks.append(f"--- Response {i} ---\n[No text content found]\n")
    return "\n".join(chunks).strip()


def _regular_report_and_txt(results: list[dict], usage: dict, implicit_prefix: str | None) -> tuple[str, str]:
    """Build (report_text, combined_txt_with_QA) for regular mode."""
    inp = int(usage.get("input_tokens", 0))
    out = int(usage.get("output_tokens", 0))
    total = int(usage.get("total_tokens", inp + out))
    cached_in = int(usage.get("cached_input_tokens", 0))
    billed = usage.get("billed_tokens")
    billed_line = f"- Billable tokens (reported): {int(billed)}\n" if isinstance(billed, int) else ""
    savings_line = f"- Cached input tokens (reported): {cached_in}\n" if cached_in > 0 else "- Cached input tokens: (not reported by API)\n"
    prefix_note = f"- Implicit shared prefix used: {bool(implicit_prefix)}\n"

    report = (
        "=== Usage Report (Regular Mode) ===\n"
        f"- Total prompts: {len(results)}\n"
        f"- Input tokens (sum): {inp}\n"
        f"- Output tokens (sum): {out}\n"
        f"- Total tokens (sum): {total}\n"
        f"{savings_line}"
        f"{billed_line}"
        f"{prefix_note}"
        "Note: Cache savings are estimated based on fields exposed by the API. "
        "If fields are missing, the API did not provide them for this run.\n"
    )

    chunks = ["=== Q&A Pairs ==="]
    for i, item in enumerate(results, 1):
        q = item.get("prompt", "")
        a = (item.get("text") or "").strip()
        err = item.get("error")
        chunks.append(f"\n--- #{i} ---\nQ:\n{q}\n")
        if err:
            chunks.append(f"A:\n[ERROR] {err}\n")
        else:
            chunks.append(f"A:\n{a}\n")

    combined_txt = report + "\n" + "\n".join(chunks)
    return report, combined_txt


# ---------------- Cookies ----------------
def get_cookie_manager():
    secret = os.getenv("COOKIE_SECRET")
    if secret:
        cookies = EncryptedCookieManager(prefix="gem_batch_runner", password=secret)
    else:
        cookies = PlainCookieManager(prefix="gem_batch_runner")
    if not cookies.ready():
        st.stop()
    return cookies


# ---------------- State Helpers ----------------
def init_state():
    for k in [
        STATE_JOB_NAME,
        STATE_JOB_STATUS,
        STATE_JOB_RESULTS,
        STATE_API_KEY,
        STATE_LAST_REQUEST,
        STATE_CACHE_NAME,
        STATE_IMPLICIT_PREFIX,
        STATE_CTX_MODE,
        STATE_REGULAR_RESULTS,
        STATE_REGULAR_USAGE,
    ]:
        st.session_state.setdefault(k, None)


def get_job_history(cookies) -> list[str]:
    raw = cookies.get(COOKIE_JOB_HISTORY)
    try:
        return json.loads(raw) if raw else []
    except Exception:
        return []


def save_job_history(cookies, history: list[str]):
    cookies[COOKIE_JOB_HISTORY] = json.dumps(history[:10])
    try:
        cookies.save()
    except Exception:
        pass


def add_to_job_history(cookies, job_name: str):
    history = get_job_history(cookies)
    if job_name not in history:
        history.insert(0, job_name)
        save_job_history(cookies, history)


def clear_history_item(cookies, job_name: str):
    history = get_job_history(cookies)
    if job_name in history:
        history.remove(job_name)
        save_job_history(cookies, history)


def reset_app_state(cookies, clear_history=False):
    if clear_history:
        try:
            del cookies[COOKIE_JOB_HISTORY]
            cookies.save()
        except Exception:
            pass

    if st.session_state.get(STATE_JOB_NAME):
        clear_history_item(cookies, st.session_state[STATE_JOB_NAME])

    for k in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS, STATE_LAST_REQUEST,
              STATE_REGULAR_RESULTS, STATE_REGULAR_USAGE]:
        st.session_state[k] = None

    st.toast("Application state cleared.")
    st.rerun()


# ---------------- Main App ----------------
def run_app():
    load_dotenv()
    init_state()
    cookies = get_cookie_manager()

    st.title("📦 Gemini Batch Runner Pro")

    # ----- Sidebar -----
    with st.sidebar:
        st.header("🛠️ Configuration")
        st.subheader("API Key")

        if st.session_state.get(STATE_API_KEY) is None:
            st.session_state[STATE_API_KEY] = cookies.get(COOKIE_API_KEY) or ""

        user_api_key_input = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            value=st.session_state.get(STATE_API_KEY) or "",
        )

        if user_api_key_input != (st.session_state.get(STATE_API_KEY) or ""):
            st.session_state[STATE_API_KEY] = user_api_key_input
            if user_api_key_input:
                cookies[COOKIE_API_KEY] = user_api_key_input
            else:
                try:
                    del cookies[COOKIE_API_KEY]
                except Exception:
                    pass
            try:
                cookies.save()
            except Exception:
                pass
            st.rerun()

        if st.button("Clear & Forget API Key"):
            st.session_state[STATE_API_KEY] = ""
            try:
                del cookies[COOKIE_API_KEY]
                cookies.save()
            except Exception:
                pass
            st.rerun()

        is_api_key_set = False
        if st.session_state.get(STATE_API_KEY):
            try:
                client_obj = _cached_initialize_client(st.session_state[STATE_API_KEY])
                bh.client = client_obj  # ensure module-global is set
                is_api_key_set = True
            except Exception as e:
                st.error(f"Failed to initialize client: {e}")
                st.stop()
        else:
            st.warning("Provide an API key to continue.")
            st.stop()

        st.divider()
        model = st.selectbox("Model", DEFAULT_MODELS, index=0, disabled=not is_api_key_set)

        # Run Mode selector: Batch vs Regular
        run_mode = st.radio(
            "Run Mode",
            ["Batch", "Regular"],
            help="Choose Batch for asynchronous large jobs at 50% cost, or Regular for immediate, per-prompt calls.",
        )

        # For Batch only: Batch Mode selector + helper
        batch_mode_help = (
            "Inline → pass a small list of requests directly in the API call (quick tests, ≤~20MB total).\n\n"
            "File (JSONL) → upload a .jsonl file with hundreds/thousands of requests (up to 2GB). "
            "Results come back as a downloadable JSONL file."
        )
        if run_mode == "Batch":
            mode = st.radio(
                "Batch Mode",
                ["Inline", "File (JSONL)"],
                disabled=not is_api_key_set,
                help=batch_mode_help,
            )
        else:
            mode = None  # not used

        # Display Name (used for Batch jobs)
        job_display_name = st.text_input(
            "Display Name",
            value="my-gemini-job",
            disabled=(not is_api_key_set or run_mode != "Batch")
        )

        # Context mode selector (sidebar ONLY the selector)
        st.divider()
        st.header("🧠 Context Mode")
        ctx_choice = st.radio(
            "How to add context?",
            ["None", "Implicit shared prefix", "Explicit cache"],
            help=(
                "Implicit (Gemini 2.5): We prepend a shared prefix to every request; "
                "the API automatically applies cache-hit discounts when prefixes match. "
                "Explicit: We create a cached context once (with TTL) and reference it from each request."
            ),
        )
        st.session_state[STATE_CTX_MODE] = ctx_choice

        # Job history (Batch only)
        st.divider()
        st.header("📜 Job History")
        job_history = get_job_history(cookies)
        if job_history:
            for job_id in job_history:
                short_id = job_id.split("/")[-1]
                if st.button(f"Load: {short_id}", key=f"load_{job_id}", use_container_width=True):
                    st.session_state[STATE_JOB_NAME] = job_id
                    st.session_state[STATE_JOB_STATUS] = None
                    st.session_state[STATE_JOB_RESULTS] = None
                    st.session_state[STATE_LAST_REQUEST] = None
                    st.rerun()
        else:
            st.info("No past jobs found.")

    # ----- MAIN AREA -----

    # --- Context controls in MAIN AREA ---
    st.header("🧠 Context (optional)")
    if st.session_state.get(STATE_CTX_MODE) == "Implicit shared prefix":
        st.session_state[STATE_CACHE_NAME] = None
        st.session_state[STATE_IMPLICIT_PREFIX] = st.text_area(
            "Shared prefix to prepend to every request:",
            height=160,
            help="Place common instructions/background here to maximize implicit cache hits (Gemini 2.5).",
            value=st.session_state.get(STATE_IMPLICIT_PREFIX) or ""
        )

    elif st.session_state.get(STATE_CTX_MODE) == "Explicit cache":
        st.session_state[STATE_IMPLICIT_PREFIX] = None

        with st.form("explicit_cache_form_main", clear_on_submit=False):
            context_text = st.text_area(
                "Cache this context once (used by all requests):",
                height=200,
                help="Long instructions, docs, code, etc. This is stored server-side and referenced by name.",
            )
            ttl_minutes = st.number_input(
                "TTL (minutes)",
                min_value=1,
                max_value=24 * 60,
                value=60,
                help="Cache expires automatically after TTL."
            )
            sys_instructions = st.text_area(
                "Optional system instruction (developer message):",
                height=100,
            )
            make_cache = st.form_submit_button("Create / Refresh Cache", type="primary")

        if make_cache:
            try:
                cache = bh.create_context_cache(
                    model=model,
                    context_text=context_text or "",
                    ttl_seconds=int(ttl_minutes) * 60,
                    display_name="gem-batch-cache",
                    system_instruction=sys_instructions or None,
                )
            except Exception as e:
                st.error(f"Failed to create cache: {e}")
            else:
                st.session_state[STATE_CACHE_NAME] = cache.name
                st.success(f"Cache ready: {cache.name}")

        if st.session_state.get(STATE_CACHE_NAME):
            cols = st.columns(2)
            cols[0].info(f"Active cache: `{st.session_state[STATE_CACHE_NAME]}`")
            if cols[1].button("Delete Cache"):
                try:
                    bh.delete_context_cache(st.session_state[STATE_CACHE_NAME])
                    st.session_state[STATE_CACHE_NAME] = None
                    st.toast("Cache deleted.")
                except Exception as e:
                    st.error(f"Failed to delete cache: {e}")

    else:
        # None
        st.session_state[STATE_CACHE_NAME] = None
        st.session_state[STATE_IMPLICIT_PREFIX] = None
        st.caption("No additional context will be added to requests.")

    st.divider()

    # ===========================
    # ====== RUN: REGULAR  ======
    # ===========================
    if run_mode == "Regular":
        st.header("⚡ Regular Mode")
        with st.form("regular_form"):
            prompts_raw = st.text_area(
                "Enter prompts (one per line)",
                height=220,
                help="Each line is sent as a separate GenerateContent call (immediate, non-batch).",
            )
            submit_reg = st.form_submit_button("▶ Run Regular", type="primary", use_container_width=True)

        if submit_reg:
            # Ensure client exists
            if not bh.client and st.session_state.get(STATE_API_KEY):
                try:
                    bh.initialize_client(st.session_state[STATE_API_KEY])
                except Exception as e:
                    st.error(f"Client init failed: {e}")
                    st.stop()

            prompts = [p.strip() for p in (prompts_raw or "").splitlines() if p.strip()]
            if not prompts:
                st.warning("Please enter at least one prompt.")
            else:
                with st.spinner("Running requests..."):
                    try:
                        results, usage = bh.generate_content_regular(
                            model=model,
                            user_texts=prompts,
                            implicit_prefix=st.session_state.get(STATE_IMPLICIT_PREFIX),
                            cache_name=st.session_state.get(STATE_CACHE_NAME),
                        )
                        st.session_state[STATE_REGULAR_RESULTS] = results
                        st.session_state[STATE_REGULAR_USAGE] = usage
                    except Exception as e:
                        st.error(f"Regular run failed: {e}")

        # Show results if present
        if st.session_state.get(STATE_REGULAR_RESULTS) is not None:
            results = st.session_state[STATE_REGULAR_RESULTS] or []
            usage = st.session_state.get(STATE_REGULAR_USAGE) or {}

            st.subheader("✅ Results (Regular Mode)")
            for i, item in enumerate(results, 1):
                with st.expander(f"Response {i}", expanded=False):
                    if item.get("error"):
                        st.error(item["error"])
                    txt = (item.get("text") or "").strip()
                    if txt:
                        st.markdown(txt)
                    else:
                        st.caption("No plain-text answer parsed; see raw JSON below.")
                    with st.popover("View Raw Response JSON"):
                        st.json(item.get("response") or {})

            # Report + downloadable combined .txt (report + Q&A)
            rep, combined_txt = _regular_report_and_txt(
                results=results,
                usage=usage,
                implicit_prefix=st.session_state.get(STATE_IMPLICIT_PREFIX),
            )
            st.divider()
            st.subheader("📊 Usage Report")
            st.code(rep, language="text")
            st.download_button(
                label="📝 Download Q&A + Report (.txt)",
                data=combined_txt,
                file_name="gemini_regular_output_with_report.txt",
                mime="text/plain",
            )

    # ===========================
    # ====== RUN: BATCH    ======
    # ===========================
    if run_mode == "Batch":
        # ----- Section 1: Submit -----
        is_job_active = bool(st.session_state[STATE_JOB_NAME])

        st.header("✍️ 1. Submit a New Batch Job")
        with st.form("submission_form"):
            if mode == "Inline":
                prompt_input = st.text_area("Enter prompts (one per line)", height=250, key="prompt_input_area")
                uploaded_file = None
            else:
                uploaded_file = st.file_uploader("Upload a JSONL file", type=["jsonl"])
                prompt_input = None

            submitted = st.form_submit_button("🚀 Submit Job", type="primary")

        if submitted:
            if not bh.client and st.session_state.get(STATE_API_KEY):
                try:
                    bh.initialize_client(st.session_state[STATE_API_KEY])
                except Exception as e:
                    st.error(f"Client init failed at submit time: {e}")
                    st.stop()

            job_fn = None
            job_args = None

            if mode == "Inline":
                prompts = [p.strip() for p in (prompt_input or "").splitlines() if p.strip()]
                if not prompts:
                    st.warning("Please enter at least one prompt.")
                else:
                    # Build inline requests, with optional implicit prefix
                    inline_requests = []
                    for p in prompts:
                        parts = []
                        if st.session_state.get(STATE_IMPLICIT_PREFIX):
                            parts.append({"text": st.session_state[STATE_IMPLICIT_PREFIX]})
                        parts.append({"text": p})
                        inline_requests.append({"contents": [{"parts": parts, "role": "user"}]})

                    # If explicit cache is set, inject cached_content into each request
                    if st.session_state.get(STATE_CACHE_NAME):
                        inline_requests = bh.augment_requests_with_cached_content(
                            inline_requests, st.session_state[STATE_CACHE_NAME]
                        )

                    st.session_state[STATE_LAST_REQUEST] = {
                        "mode": "Inline",
                        "prompts": prompts,
                        "model": model,
                        "name": job_display_name,
                        "implicit_prefix": st.session_state.get(STATE_IMPLICIT_PREFIX),
                        "cache_name": st.session_state.get(STATE_CACHE_NAME),
                    }

                    job_fn = bh.create_inline_batch_job
                    job_args = (model, inline_requests, job_display_name)

            else:  # File mode
                if not uploaded_file:
                    st.warning("Please upload a JSONL file.")
                else:
                    # Optionally rewrite JSONL to inject cached_content
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    in_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(in_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    out_path = in_path
                    cache_name = st.session_state.get(STATE_CACHE_NAME)
                    if cache_name:
                        base, _ = os.path.splitext(in_path)
                        out_path = base + ".with_cache.jsonl"
                        try:
                            bh.inject_cached_content_into_jsonl_file(in_path, out_path, cache_name)
                            st.info("Injected cached_content into the uploaded JSONL for explicit caching.")
                        except Exception as e:
                            st.error(f"Failed to inject cached_content into JSONL: {e}")
                            out_path = in_path  # fallback

                    st.session_state[STATE_LAST_REQUEST] = {
                        "mode": "File",
                        "path": out_path,
                        "model": model,
                        "name": job_display_name,
                        "implicit_prefix": None,
                        "cache_name": cache_name,
                    }

                    job_fn = bh.create_file_batch_job
                    job_args = (model, out_path, job_display_name)

            if job_fn and job_args:
                with st.spinner("Submitting job..."):
                    try:
                        base_wait = 2
                        max_retries = 5
                        job = None
                        for i in range(max_retries):
                            try:
                                job = job_fn(*job_args)
                                break
                            except ResourceExhausted as e:
                                if i < max_retries - 1:
                                    wait = (base_wait ** i) + random.random()
                                    st.warning(f"429 quota hit. Retrying in {wait:.2f}s…")
                                    time.sleep(wait)
                                else:
                                    st.error(f"Quota limit hit and max retries exceeded. {e}")
                                    raise
                        if job:
                            st.session_state[STATE_JOB_NAME] = job.name
                            st.session_state[STATE_JOB_STATUS] = job.state.name
                            add_to_job_history(cookies, job.name)
                            st.rerun()
                    except (GoogleAPICallError, GoogleAPIError, Exception) as e:
                        st.error(f"🚨 Failed to submit job. {e}")

        # ----- Section 2: Monitor -----
        is_job_active = bool(st.session_state[STATE_JOB_NAME])
        if is_job_active:
            st.header("⏳ 2. Monitor Active Job")
            st.info(f"**Job Name:** `{st.session_state[STATE_JOB_NAME]}`")

            if st.session_state.get(STATE_JOB_STATUS) not in TERMINAL_STATES:
                if st.button("🔄 Refresh Job Status"):
                    with st.spinner("Polling status…"):
                        try:
                            job = bh.poll_batch_job_status(st.session_state[STATE_JOB_NAME])
                            st.session_state[STATE_JOB_STATUS] = job.state.name
                            if st.session_state[STATE_JOB_STATUS] == "JOB_STATE_SUCCEEDED":
                                st.session_state[STATE_JOB_RESULTS] = bh.retrieve_batch_job_results(
                                    st.session_state[STATE_JOB_NAME]
                                )
                            st.rerun()
                        except Exception as e:
                            st.error(f"🚨 API Error while polling: {e}")
                            st.session_state[STATE_JOB_STATUS] = "JOB_STATE_FAILED"
                            st.rerun()

            final_status = st.session_state.get(STATE_JOB_STATUS, "UNKNOWN")
            st.metric("Current Status", final_status)

            if final_status == "JOB_STATE_SUCCEEDED":
                st.success("Job completed successfully.")
                if st.session_state.get(STATE_JOB_RESULTS):
                    results_list = st.session_state[STATE_JOB_RESULTS]
                    # Summary for batch (no per-line usage generally)
                    st.subheader("📊 Results Summary")
                    st.markdown(f"- **Total Responses:** {len(results_list)}")

                    # Download JSONL
                    jsonl_blob = _to_jsonl_blob(results_list)
                    st.download_button(
                        label="⬇️ Download Full Results (JSONL)",
                        data=jsonl_blob,
                        file_name=f"gemini_batch_results_{st.session_state[STATE_JOB_NAME].split('/')[-1]}.jsonl",
                        mime="application/x-ndjson",
                        type="primary",
                    )

                    # Also offer TXT with only extracted texts
                    extracted_txt = _extract_texts_for_report(results_list)
                    st.download_button(
                        label="📝 Download Consolidated Answers (.txt)",
                        data=extracted_txt,
                        file_name=f"gemini_batch_answers_{st.session_state[STATE_JOB_NAME].split('/')[-1]}.txt",
                        mime="text/plain",
                    )

                    st.divider()
                    preview_count = st.number_input(
                        "Number of results to preview:",
                        min_value=1,
                        max_value=len(results_list),
                        value=min(50, len(results_list)),
                        step=10,
                    )
                    # Show previews
                    for i, line in enumerate(results_list[: int(preview_count)], start=1):
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            obj = {"raw": line}
                        with st.expander(f"Response {i}", expanded=False):
                            text = _extract_best_text_from_obj(obj) or ""
                            if text:
                                st.markdown(text)
                            else:
                                st.caption("No plain-text answer parsed; see raw JSON below.")
                            with st.popover("View JSON"):
                                st.json(obj)

            elif final_status == "JOB_STATE_FAILED":
                st.error("Job failed.")
                if st.session_state.get(STATE_LAST_REQUEST):
                    if st.button("🔁 Retry Last Submission", type="primary"):
                        last = st.session_state[STATE_LAST_REQUEST]
                        try:
                            if not bh.client and st.session_state.get(STATE_API_KEY):
                                bh.initialize_client(st.session_state[STATE_API_KEY])

                            if last["mode"] == "Inline":
                                inline_requests = []
                                for p in last.get("prompts", []):
                                    parts = []
                                    if last.get("implicit_prefix"):
                                        parts.append({"text": last["implicit_prefix"]})
                                    parts.append({"text": p})
                                    inline_requests.append({"contents": [{"parts": parts, "role": "user"}]})

                                if last.get("cache_name"):
                                    inline_requests = bh.augment_requests_with_cached_content(
                                        inline_requests, last["cache_name"]
                                    )

                                job = bh.create_inline_batch_job(last["model"], inline_requests, last["name"])
                            else:
                                job = bh.create_file_batch_job(last["model"], last["path"], last["name"])
                            st.session_state[STATE_JOB_NAME] = job.name
                            st.session_state[STATE_JOB_STATUS] = job.state.name
                            st.session_state[STATE_JOB_RESULTS] = None
                            add_to_job_history(cookies, job.name)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Retry failed: {e}")

            elif final_status == "JOB_STATE_CANCELLED":
                st.warning("Job was cancelled.")

            elif final_status == "JOB_STATE_EXPIRED":
                st.warning("Job expired (no results). Consider splitting into smaller batches.")

            st.divider()
            st.header("⚙️ 3. Manage Job")
            c1, c2 = st.columns(2)
            with c1:
                is_cancellable = final_status not in TERMINAL_STATES
                if st.button("❌ Cancel Job", disabled=not is_cancellable, use_container_width=True):
                    with st.spinner("Sending cancellation request..."):
                        try:
                            bh.cancel_batch_job(st.session_state[STATE_JOB_NAME])
                            st.session_state[STATE_JOB_STATUS] = "JOB_STATE_CANCELLED"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cancel failed: {e}")

            with c2:
                if st.button("🗑️ Delete & Reset App", type="secondary", use_container_width=True):
                    try:
                        bh.delete_batch_job(st.session_state[STATE_JOB_NAME])
                    except Exception as e:
                        st.warning(f"Could not delete job from API (may already be deleted): {e}")
                    finally:
                        reset_app_state(cookies)


if __name__ == "__main__":
    run_app()
