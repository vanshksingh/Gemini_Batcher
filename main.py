import streamlit as st
import time
import json
import os
import datetime
import random

# --- Page and State Configuration ---
# This MUST be the first Streamlit command in your script.
st.set_page_config(page_title="Gemini Batch Runner Pro", layout="wide", initial_sidebar_state="expanded")

# Import third-party and local modules AFTER page config
import streamlit_cookies_manager
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from batch_handler import (
    initialize_client,
    create_inline_batch_job,
    create_file_batch_job,
    poll_batch_job_status,
    retrieve_batch_job_results,
    cancel_batch_job,
    delete_batch_job
)

# Constants for session state keys for cleaner access
COOKIE_JOB_HISTORY = "gemini_job_history"
COOKIE_API_KEY = "gemini_api_key"
STATE_JOB_NAME = "job_name"
STATE_JOB_STATUS = "job_status"
STATE_JOB_RESULTS = "job_results"
STATE_API_KEY = "api_key"
STATE_LAST_REQUEST = "last_request"  # To store prompts/file for retry
TERMINAL_STATES = ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]

# --- Cookie and Session State Initialization ---
cookies = streamlit_cookies_manager.CookieManager()
if not cookies.ready():
    st.spinner()
    st.stop()

# Initialize all session state keys
for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS, STATE_API_KEY, STATE_LAST_REQUEST]:
    if key not in st.session_state:
        st.session_state[key] = None


# --- Helper Functions ---
def display_job_output(results_str, preview_limit=50):
    """Parses and neatly displays a preview of the results from a batch job."""
    st.subheader("✅ Job Results Preview")
    try:
        results_list = [json.loads(res) for res in results_str]
        total_results = len(results_list)

        if total_results > preview_limit:
            st.info(
                f"Displaying the first {preview_limit} of {total_results} total results. Use the download button for the full output.")

        for i, res_json in enumerate(results_list[:preview_limit]):
            with st.expander(f"**Response for Request {i + 1}**", expanded=True):
                text_content = \
                    res_json.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[
                        0].get('text', 'No text content found.')
                st.markdown(text_content)
                with st.popover("View Raw JSON"):
                    st.json(res_json)
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        st.error(f"Could not parse results. Error: {e}")
        st.code(str(results_str), language='json')


def generate_local_summary(results_str):
    """Analyzes results locally to generate a simple summary."""
    st.subheader("📊 Results Summary")
    try:
        results_list = [json.loads(res) for res in results_str]
        total = len(results_list)
        errors = 0
        refusals = 0

        refusal_keywords = ["cannot", "unable", "sorry", "i am not able"]

        for res_json in results_list:
            text_content = \
            res_json.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text',
                                                                                                                 '').lower()
            if not text_content or "error" in text_content:
                errors += 1
            elif any(keyword in text_content for keyword in refusal_keywords):
                refusals += 1

        success = total - errors - refusals

        st.markdown(f"""
        - **Total Responses:** {total}
        - **Successful Responses:** {success}
        - **Content Refusals:** {refusals}
        - **Errors/Empty Responses:** {errors}
        """)

    except Exception as e:
        st.warning(f"Could not generate local summary: {e}")


def reset_app_state(clear_history=False):
    """Fully resets the application state and cookies."""
    if clear_history and COOKIE_JOB_HISTORY in cookies:
        del cookies[COOKIE_JOB_HISTORY]

    history = get_job_history()
    if st.session_state.get(STATE_JOB_NAME) in history:
        history.remove(st.session_state[STATE_JOB_NAME])
        save_job_history(history)

    for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS, STATE_LAST_REQUEST]:
        st.session_state[key] = None
    st.toast("Application has been reset.")
    st.rerun()


def get_job_history():
    """Retrieves job history from cookies."""
    history = cookies.get(COOKIE_JOB_HISTORY)
    return json.loads(history) if history else []


def save_job_history(history):
    """Saves job history to cookies, keeping only the last 10."""
    cookies[COOKIE_JOB_HISTORY] = json.dumps(history[:10])


def add_to_job_history(job_name):
    """Adds a new job to the history."""
    history = get_job_history()
    if job_name not in history:
        history.insert(0, job_name)
        save_job_history(history)


# --- Main App UI ---
st.title("📦 Gemini Batch Runner Pro")

# --- Sidebar for Configuration and API Key ---
with st.sidebar:
    st.header("🛠️ Configuration")
    st.subheader("API Key")

    if not st.session_state.get(STATE_API_KEY):
        st.session_state[STATE_API_KEY] = cookies.get(COOKIE_API_KEY)

    user_api_key_input = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.get(STATE_API_KEY, "")
    )

    if user_api_key_input != st.session_state.get(STATE_API_KEY):
        st.session_state[STATE_API_KEY] = user_api_key_input
        cookies[COOKIE_API_KEY] = user_api_key_input
        st.rerun()

    if st.button("Clear & Forget API Key"):
        st.session_state[STATE_API_KEY] = None
        if COOKIE_API_KEY in cookies:
            del cookies[COOKIE_API_KEY]
        st.rerun()

    is_api_key_set = False
    if st.session_state.get(STATE_API_KEY):
        try:
            initialize_client(st.session_state[STATE_API_KEY])
            is_api_key_set = True
        except Exception as e:
            st.error(f"Failed to initialize client: {e}")
            st.stop()
    else:
        st.warning("Please provide an API key to proceed.")
        st.stop()

    st.divider()
    model = st.selectbox("Model", ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"], disabled=not is_api_key_set)
    mode = st.radio("Batch Mode", ["Inline", "File (JSONL)"], disabled=not is_api_key_set)
    job_display_name = st.text_input("Display Name", value="my-gemini-job", disabled=not is_api_key_set)

    st.divider()
    st.header("📜 Job History")
    job_history = get_job_history()
    if job_history:
        for job_id in job_history:
            job_short_id = job_id.split('/')[-1]
            if st.button(f"Load: {job_short_id}", key=job_id, use_container_width=True):
                st.session_state[STATE_JOB_NAME] = job_id
                st.session_state[STATE_JOB_STATUS] = None
                st.session_state[STATE_JOB_RESULTS] = None
                st.session_state[STATE_LAST_REQUEST] = None
                st.rerun()
    else:
        st.info("No past jobs found.")

# A job is active if its name is in the session state.
is_job_active = st.session_state[STATE_JOB_NAME] is not None

# --- SECTION 1: Job Submission Form ---
if not is_job_active:
    st.header("✍️ 1. Submit a New Batch Job")
    with st.form("submission_form"):
        if mode == "Inline":
            prompt_input = st.text_area("Enter prompts (one per line)", height=250, key="prompt_input_area")
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader("Upload a JSONL file", type=["jsonl"])
            prompt_input = None
        submitted = st.form_submit_button("🚀 Submit Job", type="primary", disabled=not is_api_key_set)

    if submitted:
        job_function = None
        job_args = None

        if mode == "Inline":
            if prompt_input:
                prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
                if prompts:
                    st.session_state[STATE_LAST_REQUEST] = {"mode": "Inline", "prompts": prompts, "model": model,
                                                            "name": job_display_name}
                    inline_requests = [{'contents': [{'parts': [{'text': p}], 'role': 'user'}]} for p in prompts]
                    job_function = create_inline_batch_job
                    job_args = (model, inline_requests, job_display_name)
                else:
                    st.warning("Please enter at least one prompt.")
            else:
                st.warning("Please enter at least one prompt.")

        elif mode == "File (JSONL)":
            if uploaded_file:
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                jsonl_path = os.path.join(temp_dir, uploaded_file.name)
                with open(jsonl_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state[STATE_LAST_REQUEST] = {"mode": "File", "path": jsonl_path, "model": model,
                                                        "name": job_display_name}
                job_function = create_file_batch_job
                job_args = (model, jsonl_path, job_display_name)
            else:
                st.warning("Please upload a JSONL file.")

        if job_function and job_args:
            with st.spinner("Submitting job..."):
                try:
                    base_wait_time = 2
                    max_retries = 5
                    job = None
                    for i in range(max_retries):
                        try:
                            job = job_function(*job_args)
                            break
                        except ResourceExhausted as e:
                            if i < max_retries - 1:
                                wait_time = (base_wait_time ** i) + random.random()
                                st.warning(f"Quota limit hit (429). Retrying in {wait_time:.2f} seconds...", icon="⏳")
                                time.sleep(wait_time)
                            else:
                                st.error(f"🚨 API Error: Quota limit hit and max retries exceeded. {e}", icon="🔥")
                                raise e
                    if job:
                        st.session_state[STATE_JOB_NAME] = job.name
                        st.session_state[STATE_JOB_STATUS] = job.state.name
                        add_to_job_history(job.name)
                        st.rerun()
                except (GoogleAPICallError, Exception) as e:
                    st.error(f"🚨 Critical Error: Failed to submit job. {e}", icon="🔥")

# --- SECTION 2: Job Monitoring & Management ---
if is_job_active:
    st.header("⏳ 2. Monitor Active Job")
    st.info(f"**Job Name:** `{st.session_state[STATE_JOB_NAME]}`")

    if st.session_state.get(STATE_JOB_STATUS) not in TERMINAL_STATES:
        if st.button("🔄 Refresh Job Status"):
            with st.spinner("Polling status..."):
                try:
                    job = poll_batch_job_status(st.session_state[STATE_JOB_NAME])
                    st.session_state[STATE_JOB_STATUS] = job.state.name
                    if st.session_state[STATE_JOB_STATUS] == "JOB_STATE_SUCCEEDED":
                        st.session_state[STATE_JOB_RESULTS] = retrieve_batch_job_results(
                            st.session_state[STATE_JOB_NAME])
                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 API Error while polling: {e}", icon="🔥")
                    st.session_state[STATE_JOB_STATUS] = "JOB_STATE_FAILED"
                    st.rerun()

    final_status = st.session_state.get(STATE_JOB_STATUS, "UNKNOWN")
    st.metric("Current Status", final_status)

    if final_status == "JOB_STATE_SUCCEEDED":
        st.success("Job completed successfully.")
        if st.session_state.get(STATE_JOB_RESULTS):
            generate_local_summary(st.session_state[STATE_JOB_RESULTS])

            results_json_string = json.dumps([json.loads(res) for res in st.session_state[STATE_JOB_RESULTS]], indent=2)
            st.download_button(
                label="⬇️ Download Full Results (JSON)",
                data=results_json_string,
                file_name=f"gemini_batch_results_{st.session_state[STATE_JOB_NAME].split('/')[-1]}.json",
                mime="application/json",
                type="primary"
            )
            st.divider()

            preview_count = st.number_input(
                "Number of results to preview:",
                min_value=1,
                max_value=len(st.session_state[STATE_JOB_RESULTS]),
                value=min(50, len(st.session_state[STATE_JOB_RESULTS])),
                step=10
            )
            display_job_output(st.session_state[STATE_JOB_RESULTS], preview_limit=preview_count)

    elif final_status == "JOB_STATE_FAILED":
        st.error("Job failed.")
        if st.session_state.get(STATE_LAST_REQUEST):
            if st.button("🔁 Retry Job", type="primary"):
                last_request = st.session_state[STATE_LAST_REQUEST]
                reset_app_state()
                # Use the stored request to resubmit
                if last_request["mode"] == "Inline":
                    st.session_state.submitted_prompts = last_request["prompts"]
                else:  # File mode
                    st.session_state.submitted_file_path = last_request["path"]
                st.rerun()  # This will trigger the submission logic again

    elif final_status == "JOB_STATE_CANCELLED":
        st.warning("Job was cancelled.")

    st.divider()
    st.header("⚙️ 3. Manage Job")
    c1, c2 = st.columns(2)
    with c1:
        is_cancellable = final_status not in TERMINAL_STATES
        if st.button("❌ Cancel Job", disabled=not is_cancellable, use_container_width=True):
            with st.spinner("Sending cancellation request..."):
                cancel_batch_job(st.session_state[STATE_JOB_NAME])
                st.session_state[STATE_JOB_STATUS] = "JOB_STATE_CANCELLED"
                st.rerun()
    with c2:
        if st.button("🗑️ Delete & Reset App", type="secondary", use_container_width=True):
            try:
                delete_batch_job(st.session_state[STATE_JOB_NAME])
            except Exception as e:
                st.warning(f"Could not delete job from API (it may have been deleted): {e}")
            finally:
                reset_app_state()
