import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def create_inline_batch_job(model: str, requests: list, display_name: str):
    job = client.batches.create(
        model=model,
        src=requests,
        config={'display_name': display_name}
    )
    return job


def create_file_batch_job(model: str, jsonl_path: str, display_name: str):
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type='jsonl'
        )
    )

    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={'display_name': display_name}
    )
    return job


def poll_batch_job_status(job_name: str, interval: int = 30):
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"Current state: {state}")
        if state in completed_states:
            break
        time.sleep(interval)
    return job


def retrieve_batch_job_results(job_name: str):
    job = client.batches.get(name=job_name)

    if job.state.name != 'JOB_STATE_SUCCEEDED':
        return {"status": job.state.name, "error": getattr(job, 'error', None)}

    if job.dest and job.dest.file_name:
        file_content = client.files.download(file=job.dest.file_name)
        return file_content.decode("utf-8")

    elif job.dest and job.dest.inlined_responses:
        results = []
        for i, inline_response in enumerate(job.dest.inlined_responses):
            if inline_response.response:
                try:
                    results.append(inline_response.response.text)
                except AttributeError:
                    results.append(str(inline_response.response))
            elif inline_response.error:
                results.append(f"Error: {inline_response.error}")
        return results

    else:
        return "No results found."


def cancel_batch_job(job_name: str):
    return client.batches.cancel(name=job_name)


def delete_batch_job(job_name: str):
    return client.batches.delete(name=job_name)


if __name__ == "__main__":
    # Example usage — choose either inline or file batch mode

    # Inline mode example
    inline_requests = [
        {
            'contents': [{
                'parts': [{'text': 'Tell me a one-sentence joke.'}],
                'role': 'user'
            }]
        },
        {
            'contents': [{
                'parts': [{'text': 'Why is the sky blue?'}],
                'role': 'user'
            }]
        }
    ]

    inline_job = create_inline_batch_job(
        model="models/gemini-2.5-flash",
        requests=inline_requests,
        display_name="inline-example-job"
    )

    final_job = poll_batch_job_status(inline_job.name)
    results = retrieve_batch_job_results(inline_job.name)
    print("Results:", results)

    # File mode example
    """
    # Prepare a JSONL file first
    file_path = "my-batch-requests.jsonl"
    with open(file_path, "w") as f:
        requests = [
            {"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}]}},
            {"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}
        ]
        for req in requests:
            f.write(json.dumps(req) + "\n")

    file_job = create_file_batch_job(
        model="models/gemini-2.5-flash",
        jsonl_path=file_path,
        display_name="file-example-job"
    )

    final_file_job = poll_batch_job_status(file_job.name)
    results = retrieve_batch_job_results(file_job.name)
    print("Results:", results)
    """
