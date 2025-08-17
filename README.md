# 📦 Gemini Batch Runner Pro

🚀 Streamlit-powered UI for batch prediction with Gemini 2.5 API  
🔗 GSoC 2025 Project Extension — Adds Context Caching, Batch Modes, and Job Management

---

## 📖 Overview

**Gemini Batch Runner Pro** is an interactive Streamlit app that allows you to submit and manage **batch jobs** using the Gemini API.  

It supports **multiple batch input modes**, integrates **context caching (explicit & implicit)**, and provides a **visual dashboard** to track jobs, preview results, and optimize token usage.

### ✨ Key Features

- 🛠️ **Configuration Panel**
  - API key management (stored in cookies securely)
  - Model selection (`gemini-2.5-flash`, `gemini-2.0-pro`, etc.)
  - Batch mode selector: Inline | File (JSONL)

- 📦 **Batch Job Submission**
  - Inline text queries
  - JSONL file upload for large batch input
  - Add **context** via explicit or implicit cache for queries

- ⏳ **Monitor Jobs**
  - Live status updates (e.g., `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`)
  - Job history log with re-run options

- 📊 **Results Dashboard**
  - Preview responses inline
  - Export results (JSON/CSV)
  - View error/refusal breakdown

- 💾 **Context Caching Integration**
  - Explicit cache: One-time upload, guaranteed reuse (`@use_cache {name}`)
  - Implicit cache: Automatic overlap detection, less strict but cost-saving
  - No cache: Always send full context

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/vanshksingh/Gemini_BatchRunnerPro.git
cd Gemini_BatchRunnerPro
````

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

Copy it and keep it safe.

### 5. Configure `.env`

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📂 Repository Structure

```
Gemini_BatchRunnerPro/
├── main.py               # Streamlit app entrypoint
├── cache_utils.py        # Explicit/implicit cache helpers
├── gem_cache.py          # Cache-aware planning logic
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
```

---

## 🧠 Batch Modes Explained

| Mode             | Usage                                                            |
| ---------------- | ---------------------------------------------------------------- |
| **Inline**       | Manually enter queries in a text area; best for quick testing.   |
| **File**         | Upload a `.jsonl` file containing queries in bulk.               |
| **With Context** | Inject explicit/implicit cached context so queries reuse tokens. |

ℹ️ A small helper tooltip (`ⓘ`) is available in the app next to the **Batch Mode selector** explaining these options.

---

## 🧩 Core Components

### `main.py`

* Streamlit UI
* Job creation + monitoring
* Context caching integration
* Results preview/export

### `cache_utils.py`

* Create, fetch, and delete explicit caches
* Wrap queries with `@use_cache`

### `gem_cache.py`

* Plans optimal batch execution
* Chooses between explicit, implicit, or no cache

---

## 🔍 Example Usage

### Run the App

```bash
streamlit run main.py
```

### Use Case Example

1. Select **Model**: `gemini-2.5-flash`
2. Choose **Batch Mode**: `File (JSONL)`
3. Upload `queries.jsonl`
4. Optionally add **context**:

   * Explicit: Upload document once, reuse across queries
   * Implicit: Let system auto-reuse overlapping tokens
5. Submit Job 🚀

---

## 📊 Results Summary

* ✅ Total responses vs. failures
* ⚠️ Content refusals detected
* 📜 Full JSON output preview
* 💾 Export answers for downstream use

---

## 📦 Context Caching Options

| Type     | Benefit                        | Tradeoff                   |
| -------- | ------------------------------ | -------------------------- |
| Explicit | Guaranteed reuse (75% cheaper) | Immutable, one-time upload |
| Implicit | Auto-reuse overlaps            | Not guaranteed             |
| None     | Always send full context       | Most expensive             |

---

## 🛡️ Error Handling

* Missing API key → error prompt
* Failed batch → retry option
* Incomplete responses → flagged in results table

---

## 📸 Screenshots

(Suggest adding screenshots of: Config panel, Batch Mode selection with helper icon, Job Monitoring panel, Results preview)

---

## 🧑‍💻 Contributing

1. Fork this repo
2. Create a feature branch: `git checkout -b feature/xyz`
3. Push changes and open PR 🚀

---

## 📄 License

MIT License © 2025 Vansh Kumar Singh

---

## 🔗 Useful Links

* [Gemini API Docs](https://ai.google.dev/docs)
* [Gemini Studio](https://aistudio.google.com/)
* [DeepCache Project (related work)](https://github.com/vanshksingh/Gemini_DeepCache)

