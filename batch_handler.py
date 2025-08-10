import os
import json
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPICallError

# --- Global Client Variable ---
# This will be initialized by the Streamlit app with the user's API key.
client = None


def initialize_client(api_key: str):
    """
    Initializes the global client with the provided API key.
    This function MUST be called by the Streamlit app before any other handler function.
    """
    global client
    if not api_key:
        raise ValueError("API key cannot be empty.")

    # CORRECT WAY: Initialize the client object directly with the API key.
    # This replaces the incorrect genai.configure() call.
    client = genai.Client(api_key=api_key)
    print("Gemini Client Initialized successfully.")


def create_inline_batch_job(model: str, requests: list, display_name: str):
    """Creates a batch job from an inline list of requests."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")

    job = client.batches.create(
        model=model,
        src=requests,
        config={'display_name': display_name}
    )
    return job


def create_file_batch_job(model: str, jsonl_path: str, display_name: str):
    """Uploads a JSONL file and creates a batch job."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")

    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type='application/jsonl'  # Corrected MIME type
        )
    )

    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={'display_name': display_name}
    )
    return job


def poll_batch_job_status(job_name: str):
    """
    Fetches the current status of a batch job ONCE.
    The polling loop is handled by the Streamlit app.
    """
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")

    job = client.batches.get(name=job_name)
    return job


def retrieve_batch_job_results(job_name: str):
    """Retrieves the final results of a SUCCEEDED job."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")

    job = client.batches.get(name=job_name)

    if job.state.name != 'JOB_STATE_SUCCEEDED':
        # Create a serializable error dictionary
        error_info = {"message": "Job did not succeed."}
        if hasattr(job, 'error') and job.error:
            error_info['details'] = str(job.error)  # Convert error object to string
        return json.dumps({"status": job.state.name, "error": error_info})

    # For file-based output
    if hasattr(job.dest, 'file_name') and job.dest.file_name:
        file_content = client.files.download(file=job.dest.file_name)
        # The result from a file is a single string with multiple JSON objects
        # We need to wrap it in a list to be consistent with inline results
        return [line for line in file_content.decode("utf-8").strip().split('\n')]

    # For inline output
    elif hasattr(job.dest, 'inlined_responses') and job.dest.inlined_responses:
        results = []
        for inline_response in job.dest.inlined_responses:
            # Convert the protobuf response to a JSON-serializable dictionary
            results.append(types.BatchJob.to_json(inline_response))
        return results

    else:
        return [json.dumps({"error": "No results found."})]


def cancel_batch_job(job_name: str):
    """Sends a request to cancel a running job."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.cancel(name=job_name)


def delete_batch_job(job_name: str):
    """Sends a request to delete a job."""
    if not client:
        raise RuntimeError("Client not initialized. Call initialize_client() first.")
    return client.batches.delete(name=job_name)
