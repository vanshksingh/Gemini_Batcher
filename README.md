# 🌌 Gemini Batch Mode Streamlit Demo


🚀 GSoC 2025 Project with Google DeepMind


This project provides a fully functional **Streamlit frontend** for Google's Gemini API **Batch Mode**, using the official Python SDK.

## 🚀 Features

- Submit batch jobs using **Inline** or **JSONL file** mode
- Supports models like `gemini-2.5-flash`, `gemini-1.5-pro`
- Poll job status and view real-time results
- Cancel and delete jobs from the UI
- Built on top of modular backend (`batch_handler.py`)

---

## 📁 File Structure

```
.
├── main.py                # Streamlit frontend
├── batch_handler.py      # Modular Gemini Batch API wrapper
├── .env                  # Contains your GEMINI_API_KEY
```

---

## 🧪 Requirements

- Python 3.9+
- `streamlit`
- `google-generativeai`
- `python-dotenv`

Install with:

```bash
pip install streamlit google-generativeai python-dotenv
```

---

## 🛠 .env Format

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL provided in your browser (usually http://localhost:8501).

---

## 🧼 Cleanup

Batch jobs remain in the system unless deleted. Use the UI to cancel or delete jobs as needed.

---

## 📌 Notes

- Batch jobs are **asynchronous** and can take time depending on load.
- Batch Mode offers **50% token cost** compared to standard API calls.
- Use File mode for large-scale jobs (up to 2GB JSONL input).

---

## 🧠 Learn More

- [Gemini Batch Mode Docs](https://ai.google.dev/gemini-api/docs/batch)
- [Gemini API Reference](https://ai.google.dev/api/python/google/generativeai)

---

## 📄 License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
