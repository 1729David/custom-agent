import json
import logging
import os
import re
import warnings
from pathlib import Path

from pypdf import PdfReader, PdfWriter

logging.getLogger("pypdf").setLevel(logging.ERROR)

NAME = "sec-10k"
DESCRIPTION = "Split a 10-K PDF into per-item files, then summarize Part I, II, and III"

SYSTEM_PROMPT_ADDITION = (
    "You are analyzing an SEC 10-K annual report. The filing is structured as:\n"
    "  Part I  — Item 1 (Business), Item 1A (Risk Factors)\n"
    "  Part II — Item 7 (MD&A), Item 7A (Market Risk), Item 8 (Financial Statements)\n"
    "  Part III— Item 10 (Directors & Governance), Item 11 (Executive Compensation)\n"
    "Use split_10k to split the PDF, then read_10k_item to read each section. "
    "Write concise, investor-focused summaries for each Part."
)

WORKFLOW_PROMPT = """\
Analyze the 10-K filing at: {query}

The query above may contain one or two paths separated by a space:
  - First path (required): the 10-K PDF file to analyze.
  - Second path (optional): the output directory for the split item PDFs.
    If not provided, split_10k will default to a folder next to the source PDF.

Step 1 — Split: call split_10k with the PDF path and, if a second path was given, pass it as output_dir. Note the output directory and file paths returned.

Step 2 — Part I Summary (Items 1 & 1A):
  Read item_1.pdf and item_1a.pdf with read_10k_item.
  Summarize: business overview, main products/services, markets, subsidiaries, and the top risk factors.

Step 3 — Part II Summary (Items 7, 7A & 8):
  Read item_7.pdf, item_7a.pdf, and item_8.pdf with read_10k_item.
  Summarize: financial performance highlights, MD&A key points, market risk exposures, and headline financial figures (revenue, net income, cash).

Step 4 — Part III Summary (Items 10 & 11):
  Read item_10.pdf and item_11.pdf with read_10k_item.
  Summarize: key executives and directors, governance highlights, and executive compensation structure and amounts.

Step 5 — Save each Part summary to memory with tags "10k, part-I" / "10k, part-II" / "10k, part-III" and the PDF path as source.

Present all three summaries clearly labeled Part I, Part II, and Part III.\
"""

# Matches "ITEM 1A. Risk Factors" or "Item 7." as a complete line (no trailing digits = not a TOC entry)
_ITEM_RE = re.compile(
    r"(?m)^[ \t]*(?:ITEM|Item)[ \t]+(\d{1,2}[A-Za-z]?)(?:[ \t]*[.\-–][ \t]*[A-Z][^\n]*|[ \t]*)$"
)
_ITEM_TEXT_LIMIT = 30_000


def split_10k(path: str, output_dir: str = "") -> str:
    try:
        src = Path(os.path.expanduser(path))
        out = Path(os.path.expanduser(output_dir)) if output_dir else src.parent / src.stem
        out.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(str(src))

        n = len(reader.pages)

        # Pass 1: identify TOC pages (≥4 "Item \d" references on one page)
        toc_pages: set[int] = set()
        for i in range(min(20, n)):
            text = reader.pages[i].extract_text() or ""
            if len(re.findall(r"(?i)item\s+\d", text)) >= 4:
                toc_pages.add(i)

        # Pass 2: search full page text for item section headers, skipping TOC pages
        item_starts: dict[str, int] = {}
        for i in range(n):
            if i in toc_pages:
                continue
            text = reader.pages[i].extract_text() or ""
            for m in _ITEM_RE.finditer(text):
                key = m.group(1).upper()
                if key not in item_starts:
                    item_starts[key] = i
                    break

        if not item_starts:
            return "No item boundaries detected. Verify this is a text-based 10-K PDF."

        sorted_items = sorted(item_starts.items(), key=lambda x: x[1])
        created: dict[str, str] = {}

        for idx, (key, start) in enumerate(sorted_items):
            end = sorted_items[idx + 1][1] if idx + 1 < len(sorted_items) else n
            writer = PdfWriter()
            for p in range(start, end):
                writer.add_page(reader.pages[p])
            out_path = out / f"item_{key.lower()}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)
            created[f"Item {key}"] = str(out_path)

        return json.dumps({"output_dir": str(out), "items": created}, indent=2)
    except Exception as e:
        return f"Error: {e}"


def read_10k_item(path: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(os.path.expanduser(path))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        if len(text) > _ITEM_TEXT_LIMIT:
            text = text[:_ITEM_TEXT_LIMIT] + f"\n\n[truncated — {len(text)} total chars]"
        return text
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCTIONS = {
    "split_10k": split_10k,
    "read_10k_item": read_10k_item,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "split_10k",
            "description": (
                "Split a 10-K PDF into separate PDFs, one per Item section. "
                "Returns JSON with the output directory and a mapping of item names to file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the 10-K PDF file."},
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to write item PDFs (default: <pdf_stem>/ next to the source file).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_10k_item",
            "description": "Extract text from a 10-K item PDF for summarization (30,000-char limit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the item PDF file."}
                },
                "required": ["path"],
            },
        },
    },
]
