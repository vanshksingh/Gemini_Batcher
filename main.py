import streamlit as st
import time
import json
import os
import datetime
import random
import streamlit_cookies_manager
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

# --- Page and State Configuration ---
# This MUST be the first Streamlit command in your script.
st.set_page_config(page_title="Gemini Batch Runner Pro", layout="wide", initial_sidebar_state="expanded")

# This is a placeholder for your actual batch handler functions.
# You need a 'batch_handler.py' file in the same directory.
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
COOKIE_JOB_NAME = "gemini_batch_job_name"
COOKIE_API_KEY = "gemini_api_key"  # New cookie for the API key
STATE_JOB_NAME = "job_name"
STATE_JOB_STATUS = "job_status"
STATE_JOB_RESULTS = "job_results"
STATE_API_KEY = "api_key"
TERMINAL_STATES = ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]

# --- Cookie and Session State Initialization ---
# This now comes AFTER st.set_page_config()
cookies = streamlit_cookies_manager.CookieManager()
if not cookies.ready():
    # Wait for the frontend to send cookies to the backend.
    st.spinner()
    st.stop()

# Initialize all session state keys to prevent errors
for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS, STATE_API_KEY]:
    if key not in st.session_state:
        st.session_state[key] = None

# Restore job name from cookie on first load
if not st.session_state[STATE_JOB_NAME] and (previous_job := cookies.get(COOKIE_JOB_NAME)):
    st.session_state[STATE_JOB_NAME] = previous_job
    st.toast(f"Restored monitoring for job: {previous_job}")


# --- Helper Functions ---
def display_job_output(results_str):
    """Parses and neatly displays the results from a batch job."""
    st.subheader("✅ Job Results")
    try:
        results_list = [json.loads(res) for res in results_str]
        for i, res_json in enumerate(results_list):
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


def reset_app_state():
    """Fully resets the application state and cookies."""
    cookies.delete(COOKIE_JOB_NAME)
    for key in [STATE_JOB_NAME, STATE_JOB_STATUS, STATE_JOB_RESULTS]:
        st.session_state[key] = None
    st.toast("Application has been reset.")
    st.rerun()


# --- Main App UI ---
st.title("📦 Gemini Batch Runner Pro")

# --- Sidebar for Configuration and API Key ---
with st.sidebar:
    st.header("🛠️ Configuration")
    st.subheader("API Key")

    # Load API key from cookies if not in session state
    if not st.session_state.get(STATE_API_KEY):
        st.session_state[STATE_API_KEY] = cookies.get(COOKIE_API_KEY)

    user_api_key_input = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.get(STATE_API_KEY, "")
    )

    if user_api_key_input != st.session_state.get(STATE_API_KEY):
        st.session_state[STATE_API_KEY] = user_api_key_input
        cookies.set(COOKIE_API_KEY, user_api_key_input,
                    expires_at=datetime.datetime.now() + datetime.timedelta(days=30))
        st.rerun()

    if st.button("Clear & Forget API Key"):
        st.session_state[STATE_API_KEY] = None
        cookies.delete(COOKIE_API_KEY)
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

# A job is active if its name is in the session state.
is_job_active = st.session_state[STATE_JOB_NAME] is not None

# --- SECTION 1: Job Submission Form ---
if not is_job_active:
    st.header("✍️ 1. Submit a New Batch Job")
    with st.form("submission_form"):

        # --- DYNAMIC UI FOR INPUT MODE ---
        if mode == "Inline":
            prompt_input = st.text_area("Enter prompts (one per line)", height=250, key="prompt_input_area")
            uploaded_file = None
        else:  # File (JSONL) mode
            uploaded_file = st.file_uploader("Upload a JSONL file", type=["jsonl"])
            prompt_input = None

        submitted = st.form_submit_button("🚀 Submit Job", type="primary", disabled=not is_api_key_set)

    if submitted:
        job_started = False
        # --- UPDATED SUBMISSION LOGIC ---
        if mode == "Inline":
            if prompt_input:
                prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
                if prompts:
                    job_started = True
                    with st.spinner("Submitting inline job..."):
                        inline_requests = [{'contents': [{'parts': [{'text': p}], 'role': 'user'}]} for p in prompts]
                        job_function = create_inline_batch_job
                        job_args = (model, inline_requests, job_display_name)
                else:
                    st.warning("Please enter at least one prompt in the text area.")
            else:
                st.warning("Please enter at least one prompt in the text area.")

        elif mode == "File (JSONL)":
            if uploaded_file is not None:
                job_started = True
                # To handle the uploaded file, we need to save it temporarily
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                jsonl_path = os.path.join(temp_dir, uploaded_file.name)
                with open(jsonl_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Submitting job with file: {uploaded_file.name}..."):
                    job_function = create_file_batch_job
                    job_args = (model, jsonl_path, job_display_name)
            else:
                st.warning("Please upload a JSONL file.")

        if job_started:
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
                    cookies.set(COOKIE_JOB_NAME, job.name,
                                expires_at=datetime.datetime.now() + datetime.timedelta(days=7))
                    st.rerun()
            except (GoogleAPICallError, Exception) as e:
                st.error(f"🚨 Critical Error: Failed to submit job. {e}", icon="🔥")
            finally:
                # Clean up the temporary file if it was created
                if mode == "File (JSONL)" and 'jsonl_path' in locals() and os.path.exists(jsonl_path):
                    os.remove(jsonl_path)

# --- SECTION 2: Job Monitoring & Management ---
if is_job_active:
    st.header("⏳ 2. Monitor Active Job")
    st.info(f"**Job Name:** `{st.session_state[STATE_JOB_NAME]}`")

    if st.session_state[STATE_JOB_STATUS] not in TERMINAL_STATES:
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
            results_json_string = json.dumps([json.loads(res) for res in st.session_state[STATE_JOB_RESULTS]], indent=2)
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
        st.error("Job did not complete successfully.") if final_status == "JOB_STATE_FAILED" else st.warning(
            "Job was cancelled.")

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
