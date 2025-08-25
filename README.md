# 📦 Gemini Batch Runner Pro

<img width="1199" height="674" alt="Screenshot 2025-08-19 at 8 59 37 PM" src="https://github.com/user-attachments/assets/e77e9849-24a8-4fda-be4c-3725095d9200" />


🚀 Streamlit-powered UI for **regular and batch predictions** with Gemini 2.5 API
🔗 GSoC 2025 Project Extension — Adds **Context Caching, Multi-Run Modes, Usage Reports, and Job Management**

---
## ⚠️ Known API issue:
In free tier of Gemini API Explicit cache is not functioning, it is being internally escalated for a fix at the moment.
The Batch Mode is also limited to the Paid Tier of Gemini Api, although it is not reflected in documents, escaleted for a fix at the moment.

---

## 📖 Overview

**Gemini Batch Runner Pro** is an interactive Streamlit app for developers, researchers, and AI engineers to **run, manage, and optimize Gemini API calls**.

It supports **Batch Mode (Inline or File)** and **Regular Mode**, integrates **context caching (explicit & implicit)**, and generates detailed **usage reports** with token savings estimation.

---

## ✨ Key Features

* 🛠️ **Configuration Panel**

  * API key management (secure cookie storage)
  * Model selection (`gemini-2.5-flash`, `gemini-2.0-pro`, etc.)
  * Run mode selector: **Batch** | **Regular**
  * Context Mode: **None** | **Implicit shared prefix** | **Explicit cache**

* ⚡ **Regular Mode**

  * Run single or multiple prompts (one per line)
  * Add context (implicit/explicit cache)
  * Get structured results inline
  * **Automatic usage report** showing token breakdown + estimated cost savings
  * Export final Q\&A + report as **TXT file**

* 📦 **Batch Mode**

  * **Inline input**: quick multi-query jobs
  * **File input**: upload `.jsonl` file with thousands of queries
  * Add **context** (explicit or implicit cache)
  * Submit and track long-running jobs
  * **Helper tooltip** (`ⓘ`) explaining modes

* ⏳ **Job Monitoring**

  * Live job status (e.g., `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`)
  * Job history log with re-run support
  * Error & refusal breakdowns

* 📊 **Results Dashboard**

  * Inline preview of responses
  * Export results as JSON/CSV/TXT
  * Downloadable **final report with answers + usage stats**

* 💾 **Context Caching**

  * **Explicit cache**: Upload once, guaranteed reuse (`@use_cache {name}`)
  * **Implicit cache**: Auto-reuse overlapping prefixes
  * **No cache**: Always send full context

---

## 🖥️ UI & Screenshots

### 1. Main Homepage
<img width="1429" height="1167" alt="Screenshot 2025-08-25 at 6 59 52 AM" src="https://github.com/user-attachments/assets/cce7d5c9-9b28-4304-aa97-53ea5e52975f" />

### 2. Cache Creation Page
<img width="1429" height="1167" alt="Screenshot 2025-08-25 at 7 00 38 AM" src="https://github.com/user-attachments/assets/50eaf2b1-2436-44ba-a560-a9b3d45d837d" />


### 3. More Screenshots
Additional screenshots (Query page, Cache/File management, Reports, etc.) are available in the  
[`/Screenshots`](./Screenshots) folder.

---


## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/vanshksingh/Gemini_BatchRunnerPro.git
cd Gemini_BatchRunnerPro
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your Gemini API Key

👉 [Generate API Key](https://aistudio.google.com/app/apikey)

### 5. Configure `.env`

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📂 Repository Structure

```
Gemini_BatchRunnerPro/
├── main.py               # Streamlit app entrypoint (UI + logic)
├── cache_utils.py        # Explicit/implicit cache helpers
├── gem_cache.py          # Cache-aware planning logic
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
```

---

## 🧠 Run Modes & Context Options

| Mode        | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| **Regular** | Run single/multiple prompts directly with caching + inline usage reports.   |
| **Batch**   | Large-scale execution (inline or JSONL file). Includes monitoring & export. |

| Context  | Benefit                        | Tradeoff                   |
| -------- | ------------------------------ | -------------------------- |
| Explicit | Guaranteed reuse (75% cheaper) | Immutable, one-time upload |
| Implicit | Auto-reuse overlaps            | Not guaranteed             |
| None     | Always send full context       | Most expensive             |

---

## 🔍 Example Usage

### Run the App

```bash
streamlit run main.py
```

### Use Case Example

1. Select **Model**: `gemini-2.5-flash`
2. Choose **Run Mode**: `Regular`
3. Enter prompts:

   ```
   hi
   bye
   ```
4. Context Mode: `Explicit cache` (optional)
5. Get answers + **download TXT report with Q\&A and token savings**

---

## 📊 Reports & Exports

* ✅ **Usage Report**

  * Total prompts
  * Input, output, total tokens
  * Cached vs billable tokens
  * Cost savings from cache

* 📜 **Q\&A Export**

  * Clean **TXT file** containing:

    * Each question
    * Corresponding answer
    * Usage report at the end

---

## 🛡️ Error Handling

* Missing API key → error prompt
* Uninitialized client → guided fix
* Empty/no response → flagged in preview
* Batch failures → retry option


---

## 🧑‍💻 Contributing

1. Fork this repo
2. Create a branch: `git checkout -b feature/xyz`
3. Push changes and open PR 🚀

---

## 📄 License

MIT License © 2025 Vansh Kumar Singh

---

## 🔗 Useful Links

* [Gemini API Docs](https://ai.google.dev/docs)
* [Gemini Studio](https://aistudio.google.com/)
* [DeepCache Project (related work)](https://github.com/vanshksingh/Gemini_DeepCache)

---
