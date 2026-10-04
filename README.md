# Contract Clause Summarizer & Risk Flagger

**NLP project — LLM-powered contract/TOS analysis via API.**

Reads a contract, terms-of-service, or privacy policy (plain text or PDF) and uses a large language model accessed through a provider API to:

1. Summarize each key clause in plain English
2. Flag risky or one-sided clauses, classified by risk category (auto-renewal, liability cap, arbitration, data privacy, etc.)
3. Provide a plain-English rewrite for each flagged clause
4. Assign an overall risk rating (low / medium / high)

The LLM call is the core of the project — a structured prompt drives the model to return valid JSON, which the tool parses, validates, and renders into both a human-readable report and a machine-readable `report.json`.

This project was built and tested **live against the Groq free-tier API** (model `qwen/qwen3.8-27b`) and produces real analysis output, not stubbed results.

---

## What it does

1. **Loads a document** — plain text (`.txt`, `.md`) or PDF (text-layer PDFs only).
2. **Calls an LLM via API** — the document and a structured system prompt are sent to the configured provider; the model returns JSON.
3. **Parses and validates** the JSON output — tolerant of markdown fences or stray commentary around the JSON.
4. **Renders a report** — prints a readable summary to the terminal and optionally saves a structured `report.json`.

Supported file formats: `.txt`, `.md`, `.pdf` (text-layer PDFs only — scanned/image PDFs without a text layer are not supported).

---

## Project structure

```
contract-analyzer/
├── contract_analyzer.py      # Main script: load doc → call LLM → parse JSON → render report
├── config.yaml               # Provider, model, API key env var, risk rubric, output settings
├── prompts/
│   └── contract_prompt.txt   # The LLM system + user prompt (structured output spec)
├── requirements.txt          # Python deps: requests, PyYAML, pypdf
├── sample_contract.txt       # Fictitious TOS sample for testing
├── sample_contract.pdf       # Same sample rendered as a PDF (for PDF-path testing)
├── README.md                 # This file
├── .gitignore                # Excludes .env (API keys), report.json, __pycache__, venvs
└── .env                      # Your API key (never committed — gitignored)
```

---

## Setup

Set your API key. Two options — pick one:

**Option A — environment variable (current shell session only):**

```bash
# Git Bash / WSL / macOS
export GROQ_API_KEY=***your key***

# PowerShell
$env:GROQ_API_KEY="***your key***"

# Windows CMD
set GROQ_API_KEY=***your key***
```

**Option B — `.env` file (recommended, works from any shell):**

Create a file named `.env` in the project folder containing exactly:

```
GROQ_API_KEY=***your key***
```

No quotes, no spaces around the `=`. The script reads `.env` automatically. Never commit `.env` (it's in `.gitignore`).

**Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

On Windows, if the `python` on your PATH doesn't have `pip`, use the Python that does. For example:

```bash
C:\Users\Akmal\AppData\Local\Python\bin\python.exe -m pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml` to switch providers, models, and output settings:

| Key | Meaning | Working example |
|---|---|---|
| `api.provider` | `openai`, `groq`, or `anthropic` | `groq` |
| `api.model` | Model name for the provider | `qwen/qwen3.8-27b` (Groq) — see API notes below |
| `api.api_key_env` | Env var holding the key (auto-set per provider) | `GROQ_API_KEY` for Groq |
| `api.base_url` | API endpoint base URL | `https://api.groq.com/openai/v1` for Groq |
| `risk_rubric` | Categories the LLM may flag | auto_renewal, liability_cap, arbitration, data_privacy, etc. |
| `output.*` | Report format, save path, rewrite toggle | defaults work; change `report_path` to rename the JSON output |

---

## Usage

Analyze a text file:

```bash
python contract_analyzer.py sample_contract.txt
python contract_analyzer.py path/to/contract.txt
```

Analyze a PDF (text-layer PDF only):

```bash
python contract_analyzer.py path/to/contract.pdf
```

Save the JSON report to a custom path:

```bash
python contract_analyzer.py path/to/contract.pdf --output my_report.json
```

Skip saving the JSON report (print-only):

```bash
python contract_analyzer.py path/to/contract.pdf --no-save
```

Force offline demo mode (no API key; uses built-in sample output — useful for verifying the pipeline without an API call):

```bash
python contract_analyzer.py sample_contract.txt --demo
```

The script auto-detects the file type by extension (`.pdf` → PDF extraction, everything else → plain text).

---

## PDF support notes

- PDFs are read with `pypdf` (installed via `requirements.txt`).
- Only text-layer PDFs are supported. Scanned or image PDFs (no extractable text) will error with a clear message — OCR is out of scope for this tool.
- Large PDFs are truncated to the configured `max_doc_chars` (default: 20,000 characters) with a warning printed in the report. Raise `max_doc_chars` in `config.yaml` for longer documents (the Groq model used has a large context window).

---

## Example output

Running against the included `sample_contract.txt` (or its PDF equivalent) produces output like:

```
Analyzing: sample_contract.txt  (provider=groq, model=qwen/qwen3.8-27b)
Document length: 3,051 chars
Calling LLM ...
LLM response parsed.

========================================================================
CONTRACT / TOS ANALYSIS REPORT
========================================================================
Document type: tos
Overall risk rating: HIGH

------------------------------------------------------------------------
KEY CLAUSE SUMMARIES (plain language)
------------------------------------------------------------------------

[Acceptance of Terms]
  Using the service means you agree to all the rules listed in this document.

[Auto-Renewal Subscription]
  The premium plan costs $14.99 per month and renews automatically unless you
  cancel at least 24 hours in advance via account settings. No refunds are
  given for partial months.
...
```

The full output includes up to 10 flagged clauses, each with a risk category, risk level (low / medium / high), an explanation of why it matters, and a plain-English rewrite. The report also saves a structured `report.json` with metadata (provider, model, timestamp, source document) and the full result.

---

## API provider notes

**Groq (free tier) — tested and working:**

```
provider: groq
model: qwen/qwen3.8-27b
base_url: https://api.groq.com/openai/v1
api_key_env: GROQ_API_KEY
```

- Set `GROQ_API_KEY` (env var or `.env`).
- `qwen/qwen3.8-27b` is the working model.
- Groq uses an OpenAI-compatible endpoint, so the same code path handles both OpenAI and Groq — only `config.yaml` differs.
- Free tier is sufficient for this project's document sizes. Check the Groq console for the current model list if you want to try others.

**OpenAI:**

```
provider: openai
model: gpt-4o-mini        # or gpt-4o, etc.
base_url: https://api.openai.com/v1
api_key_env: OPENAI_API_KEY
```

**Anthropic:**

```
provider: anthropic
model: claude-sonnet-4-20250514   # or another Claude model
api_key_env: ANTHROPIC_API_KEY
```

---

## Prompt design (evaluation-relevant)

The prompt in `prompts/contract_prompt.txt` is the main prompt-engineering artifact. It specifies:

- A clear **role** (careful legal-document analyst for a non-lawyer).
- A **structured output schema** (JSON only, with exact field names: `document_type`, `summary`, `flagged`, `overall_risk_rating`, `notes`).
- A **risk category rubric** (10 categories: auto_renewal, liability_cap, termination, arbitration, data_privacy, unilateral_change, indemnification, governing_law, hidden_fees, non_compete).
- **Strict rules**: valid JSON only, no surrounding text, no markdown fences in the raw output, be factual and neutral, don't give legal advice, leave uncertain items out rather than guessing.
- A **temperature of 0.1** and `response_format: {type: "json_object"}` to favour deterministic, structured output.

The parser tolerates minor deviations (markdown fences stripped, brace-scanning fallback for responses that wrap the JSON), but the prompt is written to push the model toward clean JSON on the first try.

---

## Disclaimer

This tool produces informational summaries only. It does not constitute legal advice. Always consult a qualified attorney for legal decisions.
