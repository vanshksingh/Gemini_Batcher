import streamlit as st
import time
import json
import os
import datetime
import random
import streamlit_cookies_manager
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

# This is a placeholder for your actual batch handler functions.
# You need a 'batch_handler.py' file in the same directory.
# It should contain functions that initialize the Gemini client with an API key.
from batch_handler import (
    initialize_client,
    create_inline_batch_job,
    create_file_batch_job,
    poll_batch_job_status,
    retrieve_batch_job_results,
    cancel_batch_job,
    delete_batch_job
)

# --- Page and State Configuration ---
st.set_page_config(page_title="Gemini Batch Runner Pro", layout="wide", initial_sidebar_state="expanded")

# Constants for session state keys for cleaner access
COOKIE_NAME = "gemini_batch_job_name"
STATE_JOB_NAME = "job_name"
STATE_JOB_STATUS = "job_status"
STATE_JOB_RESULTS = "job_results"
STATE_API_KEY = "api_key"
TERMINAL_STATES = ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]

# --- Cookie and Session State Initialization ---
cookies = streamlit_cookies_manager.CookieManager()

# Initialize all session state keys to prevent errors
for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS, STATE_API_KEY]:
    if key not in st.session_state:
        st.session_state[key] = None

# Restore job name from cookie on first load
if not st.session_state[STATE_JOB_NAME] and (previous_job := cookies.get(COOKIE_NAME)):
    st.session_state[STATE_JOB_NAME] = previous_job
    st.toast(f"Restored monitoring for job: {previous_job}")


# --- Helper Functions ---
def display_job_output(results_str):
    """Parses and neatly displays the results from a batch job."""
    st.subheader("✅ Job Results")
    try:
        # The result from the API is a JSON string, so we parse it.
        results_list = [json.loads(res) for res in results_str]

        # Display each result in an expander
        for i, res_json in enumerate(results_list):
            with st.expander(f"**Response for Request {i + 1}**", expanded=True):
                # Extract and display the actual generated text.
                text_content = \
                res_json.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get(
                    'text', 'No text content found.')
                st.markdown(text_content)
                with st.popover("View Raw JSON"):
                    st.json(res_json)
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        st.error(f"Could not parse results. Error: {e}")
        st.code(results_str, language='json')


def reset_app_state():
    """Fully resets the application state and cookies."""
    cookies.delete(COOKIE_NAME)
    for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS]:
        st.session_state[key] = None
    st.toast("Application has been reset.")
    st.rerun()


# --- Main App UI ---
st.title("📦 Gemini Batch Runner Pro")

# --- Sidebar for Configuration and API Key ---
with st.sidebar:
    st.header("🛠️ Configuration")

    # --- API Key Management ---
    st.subheader("API Key")
    # Try to get key from Streamlit secrets first
    try:
        st.session_state[STATE_API_KEY] = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded from secrets.", icon="✅")
    except (FileNotFoundError, KeyError):
        st.info("No API Key found in secrets. Please provide one below.")
        api_key_input = st.text_input("Enter your Gemini API Key", type="password", key="api_key_input")
        if api_key_input:
            st.session_state[STATE_API_KEY] = api_key_input
            st.success("API Key accepted.", icon="👍")

    # Disable submission form if API key is missing
    is_api_key_set = bool(st.session_state.get(STATE_API_KEY))
    if not is_api_key_set:
        st.warning("Please provide an API key to proceed.")

    st.divider()
    model = st.selectbox("Model", ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"], disabled=not is_api_key_set)
    mode = st.radio("Batch Mode", ["Inline", "File (JSONL)"], disabled=not is_api_key_set)
    job_display_name = st.text_input("Display Name", value="my-gemini-job", disabled=not is_api_key_set)

# A job is active if its name is in the session state.
is_job_active = st.session_state[STATE_JOB_NAME] is not None

# --- SECTION 1: Job Submission Form ---
if not is_job_active:
    st.header("✍️ 1. Submit a New Batch Job")
    with st.form("submission_form"):
        prompt_input = st.text_area("Enter prompts (one per line)", height=250, key="prompt_input_area")
        submitted = st.form_submit_button("🚀 Submit & Start Polling", type="primary", disabled=not is_api_key_set)

    if submitted:
        prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
        if not prompts:
            st.warning("Please enter at least one prompt.")
        else:
            with st.spinner("Initializing client and submitting job..."):
                try:
                    # Initialize the client with the provided API key
                    initialize_client(st.session_state[STATE_API_KEY])

                    # --- Exponential Backoff Logic ---
                    base_wait_time = 2  # seconds
                    max_retries = 5
                    job = None
                    for i in range(max_retries):
                        try:
                            if mode == "Inline":
                                inline_requests = [{'contents': [{'parts': [{'text': p}], 'role': 'user'}]} for p in
                                                   prompts]
                                job = create_inline_batch_job(model, inline_requests, job_display_name)
                            else:  # File mode
                                jsonl_path = "temp_batch.jsonl"
                                try:
                                    with open(jsonl_path, "w") as f:
                                        for i, p in enumerate(prompts):
                                            obj = {"custom_id": f"request-{i + 1}",
                                                   "request": {"contents": [{"parts": [{"text": p}]}]}}
                                            f.write(json.dumps(obj) + "\n")
                                    job = create_file_batch_job(model, jsonl_path, job_display_name)
                                finally:
                                    if os.path.exists(jsonl_path):
                                        os.remove(jsonl_path)

                            # If successful, break the retry loop
                            break

                        except ResourceExhausted as e:
                            if i < max_retries - 1:
                                wait_time = (base_wait_time ** i) + random.random()
                                st.warning(f"Quota limit hit (429). Retrying in {wait_time:.2f} seconds...", icon="⏳")
                                time.sleep(wait_time)
                            else:
                                st.error(f"🚨 API Error: Quota limit hit and max retries exceeded. {e}", icon="🔥")
                                raise e  # Re-raise the exception to be caught by the outer block

                    if job:
                        st.session_state[STATE_JOB_NAME] = job.name
                        st.session_state[STATE_JOB_STATUS] = job.state.name
                        cookies.set(COOKIE_NAME, job.name,
                                    expires_at=datetime.datetime.now() + datetime.timedelta(days=7))
                        st.rerun()

                except (GoogleAPICallError, Exception) as e:
                    st.error(f"🚨 Critical Error: Failed to submit job. {e}", icon="🔥")

# --- SECTION 2: Job Monitoring & Management ---
if is_job_active:
    st.header("⏳ 2. Monitor Active Job")
    st.info(f"**Job Name:** `{st.session_state[STATE_JOB_NAME]}`")

    # Initialize client for monitoring if not already done
    if not is_api_key_set:
        st.error("API Key is required to monitor job status.")
    else:
        initialize_client(st.session_state[STATE_API_KEY])

        # Auto-polling logic
        if st.session_state[STATE_JOB_STATUS] not in TERMINAL_STATES:
            with st.spinner(f"Polling status... Current state: {st.session_state.get(STATE_JOB_STATUS, 'UNKNOWN')}"):
                time.sleep(20)  # Wait before the next poll
                try:
                    job = poll_batch_job_status(st.session_state[STATE_JOB_NAME])
                    st.session_state[STATE_JOB_STATUS] = job.state.name

                    if st.session_state[STATE_JOB_STATUS] == "JOB_STATE_SUCCEEDED":
                        st.session_state[STATE_JOB_RESULTS] = retrieve_batch_job_results(
                            st.session_state[STATE_JOB_NAME])

                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 API Error while polling: {e}", icon="🔥")
                    st.session_state[STATE_JOB_STATUS] = "JOB_STATE_FAILED"  # Assume failure

        # --- Display Final Status, Results, and Actions ---
        final_status = st.session_state[STATE_JOB_STATUS]

        if final_status == "JOB_STATE_SUCCEEDED":
            st.success(f"**Status:** `{final_status}`")
            if st.session_state[STATE_JOB_RESULTS]:
                # Prepare data for download
                results_json_string = json.dumps(
                    [json.loads(res) for res in st.session_state[STATE_JOB_RESULTS]],
                    indent=2
                )

                # --- DOWNLOAD BUTTON ---
                st.download_button(
                    label="⬇️ Download Results (JSON)",
                    data=results_json_string,
                    file_name=f"gemini_batch_results_{st.session_state[STATE_JOB_NAME].split('/')[-1]}.json",
                    mime="application/json",
                    type="primary"
                )
                st.divider()
                display_job_output(st.session_state[STATE_JOB_RESULTS])

        elif final_status in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
            st.error(f"**Status:** `{final_status}`") if final_status == "JOB_STATE_FAILED" else st.warning(
                f"**Status:** `{final_status}`")

        st.divider()
        st.header("⚙️ 3. Manage Job")
        col1, col2 = st.columns(2)
        with col1:
            is_cancellable = st.session_state[STATE_JOB_STATUS] not in TERMINAL_STATES
            if st.button("❌ Cancel Job", disabled=not is_cancellable):
                with st.spinner("Sending cancellation request..."):
                    cancel_batch_job(st.session_state[STATE_JOB_NAME])
                    st.session_state[STATE_JOB_STATUS] = "JOB_STATE_CANCELLED"
                    st.rerun()

        with col2:
            if st.button("🗑️ Delete & Reset App", type="secondary"):
                try:
                    delete_batch_job(st.session_state[STATE_JOB_NAME])
                except Exception as e:
                    st.warning(f"Could not delete job from API (it may have been deleted): {e}")
                finally:
                    reset_app_state()
