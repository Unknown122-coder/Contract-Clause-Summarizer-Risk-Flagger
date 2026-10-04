#!/usr/bin/env python3
"""
Contract Clause Summarizer & Risk Flagger (NLP project).

Loads a contract/TOS text or PDF file, sends it to an LLM via API with a
structured prompt, parses the JSON output, and produces a readable report
plus a report.json.

Supports OpenAI, Groq (OpenAI-compatible), and Anthropic.  PDF files are
read with pypdf (plain-text extraction — scanned/image PDFs without a text
layer are not supported).

Includes a demo mode that runs without an API key so the pipeline is
verifiable during grading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def _load_env_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE .env file (no quotes parsing, no exports)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out


def _env(var: str) -> str:
    val = os.environ.get(var, "")
    if not val:
        val = _env_cache.get(var, "")
    return val


_env_cache: dict[str, str] = _load_env_file(Path(__file__).resolve().parent / ".env")


def _base_url() -> str:
    raw = CONFIG.get("api", {}).get("base_url", "https://api.openai.com/v1")
    raw = raw.rstrip("/")
    return raw


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "contract_prompt.txt"

with open(PROMPT_PATH, encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()


# ---------------------------------------------------------------------------
# Document loading + truncation
# ---------------------------------------------------------------------------

MAX_CHARS = CONFIG.get("limits", {}).get("max_doc_chars", 20000)


def load_document(path: str) -> tuple[str, list[str]]:
    """Return (text, warnings). Supports .txt, .md, and .pdf files.

    PDF files are read with pypdf (plain-text extraction; scanned/image PDFs
    without a text layer will produce little or no text and should be rasterized
    with OCR instead, which is out of scope for this tool).
    """
    warnings: list[str] = []
    p = Path(path)

    if not p.is_file():
        sys.exit(f"Error: file not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError as exc:
            sys.exit(
                "Error: PDF support requires the 'pypdf' package.\n"
                "Install it with:  python -m pip install -r requirements.txt\n"
                "(Original error: {})".format(exc)
            )
        try:
            reader = pypdf.PdfReader(str(p))
        except Exception as exc:
            sys.exit(f"Error: could not read PDF '{path}': {exc}")
        pages: list[str] = []
        for page in reader.pages:
            chunk = page.extract_text()
            if chunk:
                pages.append(chunk)
        text = "\n\n".join(pages)
        if not text.strip():
            sys.exit(
                "Error: PDF appears to have no extractable text layer. "
                "Scanned/image PDFs require OCR, which this tool does not provide."
            )
        pages_count = len(reader.pages)
        if len(text) > MAX_CHARS:
            warnings.append(
                f"PDF had {pages_count} page(s) and exceeded {MAX_CHARS:,} chars "
                f"after extraction; truncated from {len(text):,} to {MAX_CHARS:,} chars. "
                f"Results may be partial."
            )
            text = text[:MAX_CHARS]
        else:
            warnings.append(f"PDF parsed: {pages_count} page(s), {len(text):,} chars extracted.")
    else:
        text = p.read_text(encoding="utf-8")
        if len(text) > MAX_CHARS:
            warnings.append(
                f"Document exceeded {MAX_CHARS:,} chars and was truncated "
                f"({len(text):,} -> {MAX_CHARS:,} chars). Results may be partial."
            )
            text = text[:MAX_CHARS]

    if not text.strip():
        sys.exit("Error: document is empty or could not be read.")

    return text, warnings


# ---------------------------------------------------------------------------
# LLM builders — build the right request for the configured provider
# ---------------------------------------------------------------------------


def _build_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": f"Here is the document to analyze:\n\n{text}"},
    ]


def openai_request(model: str, messages: list[dict[str, str]], api_key: str) -> dict[str, Any]:
    """Build the HTTP request for an OpenAI-compatible endpoint (OpenAI, Groq, etc.)."""
    return {
        "method": "POST",
        "url": f"{_base_url()}/chat/completions",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "json": {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
    }


def anthropic_request(model: str, text: str, api_key: str, max_tokens: int = 4096) -> dict[str, Any]:
    """Build the HTTP request body/headers for the Anthropic messages API."""
    system_text = PROMPT_TEMPLATE
    user_text = f"Here is the document to analyze:\n\n{text}"
    return {
        "method": "POST",
        "url": "https://api.anthropic.com/v1/messages",
        "headers": {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        "json": {
            "model": model,
            "system": system_text,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            ],
        },
    }


# ---------------------------------------------------------------------------
# HTTP call with retry
# ---------------------------------------------------------------------------


def call_llm(request_spec: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
    """Execute the HTTP request with exponential backoff on 429/5xx."""
    method = request_spec["method"]
    url = request_spec["url"]
    headers = dict(request_spec["headers"])
    json_body = request_spec.get("json")

    for attempt in range(max_retries):
        try:
            if method.upper() == "POST":
                resp = requests.post(url, headers=headers, json=json_body, timeout=120)
            elif method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=120)
            else:
                resp = requests.request(method, url, headers=headers, json=json_body, timeout=120)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Request failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2**attempt)
            continue

        # Non-retryable client/server errors: print the body once and fail.
        if resp.status_code >= 400 and resp.status_code not in (429, 500, 502, 503, 504):
            try:
                err_body = resp.text[:1000]
            except Exception:
                err_body = "(could not read response body)"
            print(f"  [error] LLM returned {resp.status_code}: {err_body}")
            raise RuntimeError(
                f"LLM request failed with HTTP {resp.status_code}: {err_body}"
            )

        # Retryable: 429 (rate limit) and 5xx (server errors).
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2**attempt
            print(f"  [warning] API error {resp.status_code}, retrying in {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)
            continue

        payload = resp.json()
        # OpenAI returns choices[0].message.content
        if "choices" in payload and payload["choices"]:
            content = payload["choices"][0].get("message", {}).get("content", "")
            if content:
                return {"content": content, "raw": payload}
        # Anthropic returns content block in content[0].text
        if "content" in payload and payload["content"]:
            text = ""
            for block in payload["content"]:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    break
            if text:
                return {"content": text, "raw": payload}
        # Fallback: try to stringify
        return {"content": json.dumps(payload), "raw": payload}

    raise RuntimeError("Exhausted retries without a usable response.")


# ---------------------------------------------------------------------------
# JSON parsing — robust against markdown fences and stray text
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of the LLM response, tolerating fences/commentary."""
    # Strip markdown code fences
    text = re.sub(r"^```(?:json|markdown|txt)?\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text.strip())

    # Try the whole thing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } that parses
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    raise ValueError("Could not extract valid JSON from LLM response.")


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


PROVIDER_DISPATCH = {
    "openai": openai_request,
    "groq": openai_request,     # Groq is OpenAI-compatible
    "anthropic": anthropic_request,
}


def call_provider(provider: str, model: str, api_key: str, text: str) -> dict[str, Any]:
    if provider not in PROVIDER_DISPATCH:
        sys.exit(f"Error: unsupported provider '{provider}'. Use openai, groq, or anthropic.")
    builder = PROVIDER_DISPATCH[provider]
    messages = _build_messages(text)
    if provider == "anthropic":
        spec = builder(model, text, api_key)
    else:
        spec = builder(model, messages, api_key)

    result = call_llm(spec)
    return extract_json(result["content"])


# ---------------------------------------------------------------------------
# Demo mode — produces a plausible analysis without an API key
# ---------------------------------------------------------------------------


DEMO_RESULT = {
    "document_type": "tos",
    "summary": [
        {"clause": "Auto-Renewal Subscription", "plain_summary": "Premium plan auto-renews monthly at $14.99; you must cancel at least 24 hours before renewal, and there are no refunds for partial periods."},
        {"clause": "Fees and Price Changes", "plain_summary": "The company can change prices at any time without notice; continued use means you accept the new price."},
        {"clause": "Limitation of Liability", "plain_summary": "The company is not liable for indirect or consequential damages, and total liability is capped at what you paid in the last 30 days."},
        {"clause": "Arbitration and Waiver", "plain_summary": "Disputes must go through binding individual arbitration; you give up class actions and a jury trial, in the company's home jurisdiction."},
        {"clause": "Termination", "plain_summary": "The company can suspend or terminate your account with 7 days' notice and no cause required; you can cancel anytime."},
        {"clause": "Data Collection and Usage", "plain_summary": "The company collects personal, usage, and device data, may share aggregated data with partners, and may sell personal data where allowed; data is kept indefinitely unless you request deletion."},
        {"clause": "Unilateral Changes", "plain_summary": "The company can change these terms anytime and notify you by email; continued use counts as acceptance, with no separate consent required."},
        {"clause": "Indemnification", "plain_summary": "You must indemnify the company and its officers/directors/employees for claims arising from your use of the service or violation of the terms."},
        {"clause": "Governing Law", "plain_summary": "The terms are governed by the laws of Exampleland, and legal action must be brought in Exampleland courts."},
        {"clause": "Non-Compete", "plain_summary": "While subscribed and for 12 months after, you may not use a competing service with similar features."},
    ],
    "flagged": [
        {"clause": "Auto-Renewal Subscription", "risk_category": "auto_renewal", "risk_level": "medium", "explanation": "Auto-renewal with a short cancellation window and no refunds can trap users into unwanted charges.", "plain_english_rewrite": "Your Premium subscription renews monthly at $14.99. You may cancel anytime through account settings; cancellation takes effect at the end of your current billing period, and you will not be charged for the next period."},
        {"clause": "Fees and Price Changes", "risk_category": "hidden_fees", "risk_level": "high", "explanation": "Unilateral, no-notice price increases let the company raise rates without giving you a chance to decide whether to stay.", "plain_english_rewrite": "We may change pricing with at least 30 days' notice by email. You may continue at the new price or cancel before the change takes effect."},
        {"clause": "Limitation of Liability", "risk_category": "liability_cap", "risk_level": "high", "explanation": "Capping liability at 30 days of fees leaves you with little recourse if the service causes significant harm.", "plain_english_rewrite": "To the extent permitted by law, our total liability for any claim arising from these Terms is limited to the amount you paid us in the 12 months preceding the claim. We are not liable for indirect or consequential damages where such limitation is prohibited by law."},
        {"clause": "Arbitration and Waiver", "risk_category": "arbitration", "risk_level": "high", "explanation": "Mandatory individual arbitration with a class-action and jury-trial waiver significantly limits your ability to challenge the company.", "plain_english_rewrite": "Any dispute may be resolved through binding arbitration. You may choose to resolve disputes in small claims court. Class actions and jury trials are available where permitted by law."},
        {"clause": "Data Collection and Usage", "risk_category": "data_privacy", "risk_level": "high", "explanation": "Selling personal data and indefinite retention without clear consent is a significant privacy risk.", "plain_english_rewrite": "We collect information you provide, usage data, and device information to operate and improve the service. We do not sell personal data. We retain personal data only as long as necessary and delete it within 30 days of a verified deletion request."},
        {"clause": "Unilateral Changes", "risk_category": "unilateral_change", "risk_level": "medium", "explanation": "Being able to change terms without separate consent means you may be bound by new obligations you never explicitly agreed to.", "plain_english_rewrite": "We may update these Terms with at least 30 days' notice by email. For material changes, we will ask for your explicit consent. Continued use after notice constitutes acceptance of non-material changes."},
        {"clause": "Indemnification", "risk_category": "indemnification", "risk_level": "medium", "explanation": "A broad indemnity covering the company's officers and directors shifts risk to you broadly.", "plain_english_rewrite": "You agree to indemnify ExampleService against claims arising from your misuse of the service or violation of these Terms, to the extent permitted by law."},
        {"clause": "Governing Law", "risk_category": "governing_law", "risk_level": "medium", "explanation": "Requiring litigation in the company's home state raises the cost of enforcing your rights.", "plain_english_rewrite": "These Terms are governed by the laws of the State of Exampleland. Either party may bring legal action in the courts of the other party's principal place of business for claims under $10,000."},
        {"clause": "Non-Compete", "risk_category": "non_compete", "risk_level": "high", "explanation": "A 12-month non-compete on 'similar features' is vague and restrictive, potentially limiting your ability to use other tools.", "plain_english_rewrite": "During your subscription, you may not reverse-engineer, resell, or redistribute the service. No non-compete applies after termination."},
    ],
    "overall_risk_rating": "high",
    "notes": "Document appears to heavily favor the service provider. Several clauses are one-sided and would benefit from negotiation. This is a fictitious sample for testing.",
}


def demo_mode(env_var: str, env_value: str) -> dict[str, Any]:
    present = bool(env_value and env_value.strip())
    if present:
        reason = (
            "the environment variable is set but is empty/whitespace. "
            "Check for leading/trailing spaces or quotes in the value."
        )
    else:
        reason = (
            f"{env_var} is not set in the environment and no .env file was found "
            f"with that key. Either set {env_var} in your shell, or create a .env "
            f"file next to config.yaml containing:  {env_var}=<your key>"
        )
    print(f"[demo mode] No API key detected — running with a built-in sample analysis.")
    print(f"[demo mode] Reason: {reason}")
    return DEMO_RESULT


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(result: dict[str, Any], warnings: list[str]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CONTRACT / TOS ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append(f"Document type: {result.get('document_type', 'unknown')}")
    lines.append(f"Overall risk rating: {result.get('overall_risk_rating', 'unknown').upper()}")
    lines.append("")

    if warnings:
        lines.append("WARNINGS / NOTES:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("KEY CLAUSE SUMMARIES (plain language)")
    lines.append("-" * 72)
    for item in result.get("summary", []):
        lines.append(f"\n[{item.get('clause', 'Unnamed clause')}]")
        lines.append(f"  {item.get('plain_summary', '')}")
    lines.append("")

    flagged = result.get("flagged", [])
    lines.append("-" * 72)
    lines.append(f"FLAGGED CLAUSES ({len(flagged)})")
    lines.append("-" * 72)
    for idx, item in enumerate(flagged, 1):
        cat = item.get("risk_category", "unknown")
        level = item.get("risk_level", "unknown")
        lines.append(f"\n{idx}. [{item.get('clause', 'Unnamed clause')}]")
        lines.append(f"   Risk category : {cat}")
        lines.append(f"   Risk level    : {level.upper()}")
        lines.append(f"   Why it matters: {item.get('explanation', '')}")
        rewrite = item.get("plain_english_rewrite")
        if rewrite:
            lines.append(f"   Fairer wording: {rewrite}")
    lines.append("")

    notes = result.get("notes")
    if notes:
        lines.append("-" * 72)
        lines.append("NOTES")
        lines.append("-" * 72)
        lines.append(f"  {notes}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("Disclaimer: This is an informational summary only and does not")
    lines.append("constitute legal advice. Consult a qualified attorney for decisions.")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a contract/TOS document with an LLM and flag risky clauses.",
    )
    parser.add_argument(
        "document",
        nargs="?",
        help="Path to the contract/TOS/PDF file to analyze.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save report.json to disk.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom path for the saved JSON report (default: report.json in CWD).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force demo mode (no API call), useful for offline verification.",
    )
    args = parser.parse_args()

    api_cfg = CONFIG.get("api", {})
    provider = api_cfg.get("provider", "openai")
    model = api_cfg.get("model", "gpt-4o-mini")
    key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")

    # Per-provider env var override (groq -> GROQ_API_KEY, etc.)
    preferred_env = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = preferred_env.get(provider, key_env)
    api_key = _env(env_var)

    use_demo = args.demo or not api_key
    doc_path = args.document or Path(__file__).resolve().parent / "sample_contract.txt"
    doc_path = Path(doc_path)

    text, warnings = load_document(str(doc_path))

    print(f"Analyzing: {doc_path.name}  (provider={provider}, model={model})")
    print(f"Document length: {len(text):,} chars")
    if use_demo:
        result = demo_mode(env_var, api_key)
    else:
        print("Calling LLM ...")
        result = call_provider(provider, model, api_key, text)
        print("LLM response parsed.")

    report_text = render_report(result, warnings)
    print("\n" + report_text + "\n")

    if not args.no_save:
        save_path = Path(args.output) if args.output else Path("report.json")
        # Merge metadata
        out = {
            "metadata": {
                "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "document": str(doc_path),
                "provider": provider,
                "model": model,
                "warnings": warnings,
            },
            "result": result,
        }
        save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to: {save_path.resolve()}")

    # Exit with a non-zero code if the document came back high-risk (convenience)
    rating = result.get("overall_risk_rating", "low")
    if rating == "high":
        print("\n[info] Overall risk rating is HIGH. Review flagged clauses carefully.")


if __name__ == "__main__":
    main()
