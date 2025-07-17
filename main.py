import streamlit as st
import time
from batch_handler import (
    create_inline_batch_job,
    create_file_batch_job,
    poll_batch_job_status,
    retrieve_batch_job_results,
    cancel_batch_job,
    delete_batch_job
)
import json
import os

st.set_page_config(page_title="Gemini Batch Job Runner", layout="centered")
st.title("📦 Gemini Batch Mode Demo")

# --- Sidebar for config
st.sidebar.header("🛠 Configuration")
model = st.sidebar.selectbox("Model", ["models/gemini-2.5-flash", "models/gemini-1.5-pro"])
mode = st.sidebar.radio("Batch Mode", ["Inline", "File (JSONL)"])
job_display_name = st.sidebar.text_input("Display Name", value="my-batch-job")

# --- Session State
if "job_name" not in st.session_state:
    st.session_state.job_name = None
if "job_created" not in st.session_state:
    st.session_state.job_created = False

# --- Prompt input
st.subheader("✍️ Input Prompts")
prompt_input = st.text_area("Enter prompts (one per line)", height=200)

# --- Submit batch job
if st.button("🚀 Submit Batch Job"):
    prompts = [p.strip() for p in prompt_input.split("\n") if p.strip()]
    if not prompts:
        st.warning("Please enter at least one prompt.")
    else:
        try:
            if mode == "Inline":
                inline_requests = [{'contents': [{'parts': [{'text': p}], 'role': 'user'}]} for p in prompts]
                job = create_inline_batch_job(model, inline_requests, job_display_name)
            else:
                jsonl_path = "temp_batch.jsonl"
                with open(jsonl_path, "w") as f:
                    for i, p in enumerate(prompts):
                        obj = {
                            "key": f"request-{i+1}",
                            "request": {"contents": [{"parts": [{"text": p}]}]}
                        }
                        f.write(json.dumps(obj) + "\n")
                job = create_file_batch_job(model, jsonl_path, job_display_name)
                os.remove(jsonl_path)

            st.session_state.job_name = job.name
            st.session_state.job_created = True
            st.success(f"Batch job submitted: {job.name}")

        except Exception as e:
            st.error(f"Error submitting job: {e}")

# --- Monitor status and get results
if st.session_state.job_created:
    st.markdown("### ⏳ Job Status")
    if st.button("🔍 Poll Job Status"):
        with st.spinner("Polling..."):
            job = poll_batch_job_status(st.session_state.job_name)
            state = job.state.name
            st.write(f"📌 Current Job State: `{state}`")

            if state == "JOB_STATE_SUCCEEDED":
                with st.spinner("Fetching results..."):
                    result = retrieve_batch_job_results(st.session_state.job_name)
                    if isinstance(result, list):
                        for idx, r in enumerate(result):
                            st.markdown(f"**Prompt {idx + 1}**")
                            st.code(r)
                    else:
                        st.markdown("📁 File output:")
                        st.code(result)
            elif state == "JOB_STATE_FAILED":
                st.error("Job failed.")
            elif state == "JOB_STATE_CANCELLED":
                st.warning("Job was cancelled.")

# --- Cancel or Delete Job
st.markdown("---")
st.subheader("⚙️ Manage Job")
col1, col2 = st.columns(2)
with col1:
    if st.button("❌ Cancel Job"):
        if st.session_state.job_name:
            try:
                cancel_batch_job(st.session_state.job_name)
                st.warning("Job cancelled.")
            except Exception as e:
                st.error(f"Cancel error: {e}")
        else:
            st.info("No active job to cancel.")
with col2:
    if st.button("🗑️ Delete Job"):
        if st.session_state.job_name:
            try:
                delete_batch_job(st.session_state.job_name)
                st.info("Job deleted.")
                st.session_state.job_created = False
            except Exception as e:
                st.error(f"Delete error: {e}")
        else:
            st.info("No job to delete.")
