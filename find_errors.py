"""AEC Hackathon agent: page-wise item labeling, then conflict resolution.

PDF-only (no data pack). Schedules → one item per Tag row. Drawings → full page text.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

DATASET_DIR = os.environ.get("DATASET_DIR", "assets/datasets")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "./output.json")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

CATEGORIES = (
    "cross-document-conflict",
    "code-violation",
    "unit-error",
    "missing-item",
)

ItemMap = dict[str, list[dict[str, Any]]]

# Finish / fixture / equipment tags that can be slash-grouped on drawings (CPT-2/3).
TAG_CHUNK_RE = re.compile(
    r"\b((?:CPT|P|WB|CONC|ESD|MWP|ACT|TWB|T|SS|PL|FRP|WP|CG|GLF|RSH|WD|UPH|WV|"
    r"DF|WC|L|U|MS|DWH|EWH|GD|KS|FCU)-\d+[A-Za-z]?(?:/\d+[A-Za-z]?)*)\b",
    re.IGNORECASE,
)
ROOM_KEY_RE = re.compile(r"^[NSEW]?\d+[A-Z]?$", re.IGNORECASE)
MEP_KEY_RE = re.compile(
    r"^(?:WC|L|U|DF|MS|DWH|EWH|GD|KS|FCU|FD)-\d+[A-Z]?$",
    re.IGNORECASE,
)


def list_pdfs(dataset_dir: str) -> list[str]:
    names = []
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".pdf"):
            names.append(name)
    return names


def is_schedule_pdf(name: str) -> bool:
    return "schedule" in name.lower()


def is_drawing_pdf(name: str) -> bool:
    lower = name.lower()
    return "draw" in lower or "plan" in lower


def iter_pdf_pages(path: str) -> list[str]:
    from pypdf import PdfReader

    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(text)
        print(f"  extracted page {i}: {len(text)} chars")
    return pages


def call_llm(prompt: str, *, timeout: int = 180) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_key(key: str) -> str:
    return re.sub(r"\s+", " ", (key or "").strip())


def expand_tag_token(token: str) -> list[str]:
    """CPT-2/3 → [CPT-2, CPT-3]; P-1/4 → [P-1, P-4]."""
    token = token.strip()
    m = re.match(
        r"^([A-Za-z]+)-(\d+[A-Za-z]?)((?:/\d+[A-Za-z]?)*)$",
        token,
    )
    if not m:
        return [normalize_key(token)]
    prefix, first, rest = m.group(1).upper(), m.group(2), m.group(3)
    # Preserve letter suffixes case as in source for digits-only first part
    nums = [first] + [p for p in rest.split("/") if p]
    return [normalize_key(f"{prefix}-{n}") for n in nums]


def tags_in_text(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in TAG_CHUNK_RE.finditer(text or ""):
        for tag in expand_tag_token(match.group(1)):
            if tag not in seen:
                seen.add(tag)
                found.append(tag)
    return found


def summarize_map(item_map: ItemMap, *, max_keys: int = 120) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key in sorted(item_map.keys())[:max_keys]:
        texts = [c["text"] for c in item_map[key][:3]]
        summary[key] = " | ".join(texts)[:400]
    return summary


def _parse_items(raw: str) -> list[dict[str, str]]:
    data = parse_json_object(raw)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = normalize_key(str(item.get("key", "")))
        text = str(item.get("text", "")).strip()
        if key and text:
            out.append({"key": key, "text": text})
    return out


def label_schedule_page(*, document: str, page: int, page_text: str) -> list[dict[str, str]]:
    """One item per schedule/table row (Tag column)."""
    if not page_text.strip():
        return []
    prompt = f"""This page is a product / fixture SCHEDULE (spreadsheet-like table).

Document: {document}
Page: {page}

Page text:
---
{page_text[:16000]}
---

Emit EXACTLY ONE item per table row that has a Tag (e.g. CPT-1, L-1, WC-1, CG-1).
Do not merge rows. Do not skip rows that have a Tag.
If the same Tag appears on two rows, emit TWO items with that same key (different text).
The "text" must include the important columns for that row (Manufacturer, Product,
Masterformat, Category, Remarks/notes, colors, flow rates, etc.).

Respond with ONLY JSON:
{{"items": [{{"key": "<Tag>", "text": "<row fields as one sentence or semicolon-separated>"}}]}}
"""
    return _parse_items(call_llm(prompt))


def label_drawing_page(
    *,
    document: str,
    page: int,
    page_text: str,
    existing_summary: dict[str, str],
    allow_chunk_retry: bool = True,
) -> list[dict[str, str]]:
    if len(page_text.strip()) < 40:
        print(f"  warning: nearly empty text for {document} p{page} ({len(page_text)} chars)")
        return []

    keys_blob = json.dumps(existing_summary, ensure_ascii=False, indent=2)
    chunk = page_text[:18000]
    prompt = f"""You extract items from a construction DRAWING sheet (floor plan, diagram, notes).

The page text below is real extracted PDF content — use it. Do NOT return an empty list if
tags, rooms, equipment marks, finish codes, panels, or measurable notes appear.

A KEY is a stable id: finish tag (CPT-1, P-1), fixture (L-1, WC-2), equipment (DWH-1,
MSB-NORTH), room number (N106), panel, etc.

Reuse existing schedule keys when the drawing references the same tag.

Existing schedule/items:
{keys_blob}

Document: {document}
Page: {page}
Page text length: {len(page_text)} characters

Page text:
---
{chunk}
---

Respond with ONLY JSON:
{{"items": [{{"key": "<id>", "text": "<fact from this sheet>"}}]}}

Rules:
- Emit a claim for each distinct tag/mark/room with useful info on this page.
- For rooms: include finish codes shown (floor/base/wall), e.g. "Room N106 floor CPT-2/3 base WB-2 wall P-1".
- ALWAYS emit fixture/MEP marks when present on the sheet (WC-1, WC-2, L-1, L-2, U-1,
  DF-1, DF-2, MS-1, DWH-1, EWH-1, GD-1, KS-1, FCU-1, FD-1) — reuse schedule keys exactly.
- Also emit each finish tag (CPT-*, P-*, WB-*, CONC-*, MWP-*) when the legend or plan uses it.
- For notes with slopes, ratings, sizes, voltages — attach to the relevant key.
- Prefer many specific items over omitting content.
"""
    items = _parse_items(call_llm(prompt))
    if not items and allow_chunk_retry and len(page_text) > 500:
        print(f"  retry labeling {document} p{page} in chunks…")
        for start in range(0, min(len(page_text), 24000), 8000):
            part = page_text[start : start + 8000]
            items.extend(
                label_drawing_page(
                    document=document,
                    page=page,
                    page_text=part,
                    existing_summary=existing_summary,
                    allow_chunk_retry=False,
                )
            )
        seen: set[tuple[str, str]] = set()
        uniq = []
        for it in items:
            sig = (it["key"], it["text"])
            if sig not in seen:
                seen.add(sig)
                uniq.append(it)
        items = uniq
    return items


def label_generic_page(
    *,
    document: str,
    page: int,
    page_text: str,
    existing_summary: dict[str, str],
) -> list[dict[str, str]]:
    if not page_text.strip():
        return []
    keys_blob = json.dumps(existing_summary, ensure_ascii=False, indent=2)
    prompt = f"""Extract facts about unique construction items from this page.
Reuse existing keys when a requirement applies to them.

Existing items:
{keys_blob}

Document: {document}
Page: {page}

Page text:
---
{page_text[:12000]}
---

Respond with ONLY JSON:
{{"items": [{{"key": "<item id>", "text": "<claim>"}}]}}
"""
    return _parse_items(call_llm(prompt))


def add_claims(item_map: ItemMap, document: str, claims: list[dict[str, Any]]) -> int:
    n = 0
    for claim in claims:
        key = normalize_key(str(claim["key"]))
        text = str(claim["text"]).strip()
        page = int(claim.get("page") or 1)
        if not key or not text:
            continue
        item_map.setdefault(key, []).append(
            {"document": document, "page": page, "text": text}
        )
        n += 1
    return n


def expand_room_finish_links(item_map: ItemMap) -> int:
    """Propagate finish tags from room/drawing claims onto CPT-*/P-*/… keys."""
    added = 0
    snapshot = {k: list(v) for k, v in item_map.items()}
    for key, claims in snapshot.items():
        for claim in claims:
            if not is_drawing_pdf(claim["document"]):
                continue
            tags = tags_in_text(claim["text"])
            for tag in tags:
                if tag.upper() == key.upper():
                    continue
                if ROOM_KEY_RE.match(key):
                    text = f"Room {key} uses {tag}: {claim['text']}"
                else:
                    text = f"Drawing co-occurrence with {key}: {claim['text']}"
                existing = item_map.get(tag, [])
                if any(
                    e["document"] == claim["document"]
                    and e["page"] == claim["page"]
                    and e["text"] == text
                    for e in existing
                ):
                    continue
                item_map.setdefault(tag, []).append(
                    {
                        "document": claim["document"],
                        "page": claim["page"],
                        "text": text,
                    }
                )
                added += 1
    print(f"Room→finish link expansion: +{added} claim(s)")
    return added


def link_schedule_marks_on_drawings(
    item_map: ItemMap, drawing_pages: dict[str, list[str]]
) -> int:
    """If a schedule mark appears in drawing PDF text, attach a drawing claim (MEP orphans)."""
    schedule_keys = {
        k
        for k, claims in item_map.items()
        if any(is_schedule_pdf(c["document"]) for c in claims)
    }
    # Prefer MEP-like keys, but also any short schedule tag
    priority = [k for k in schedule_keys if MEP_KEY_RE.match(k)]
    others = [k for k in schedule_keys if k not in priority]
    watch = priority + others

    added = 0
    for doc, pages in drawing_pages.items():
        for page_num, page_text in enumerate(pages, start=1):
            # Normalize text for matching: collapse whitespace
            hay = page_text.upper()
            for key in watch:
                # Word-ish match for mark (WC-1, L-1, MSB-NORTH handled separately)
                pattern = re.compile(
                    r"(?<![A-Z0-9])" + re.escape(key.upper()) + r"(?![A-Z0-9])"
                )
                if not pattern.search(hay):
                    continue
                text = f"Mark {key} appears on drawing sheet page {page_num}."
                existing = item_map.get(key, [])
                if any(
                    e["document"] == doc and e["page"] == page_num and key in e["text"]
                    for e in existing
                    if e["document"] == doc
                ):
                    continue
                item_map.setdefault(key, []).append(
                    {"document": doc, "page": page_num, "text": text}
                )
                added += 1
    print(f"Drawing mark linking: +{added} claim(s) for schedule tags found in drawings")
    return added


def build_item_map(dataset_dir: str) -> ItemMap:
    item_map: ItemMap = {}
    drawing_pages: dict[str, list[str]] = {}
    pdfs = list_pdfs(dataset_dir)
    if not pdfs:
        print(f"No PDFs found in {dataset_dir}")
        return item_map

    def sort_key(name: str) -> tuple[int, str]:
        if is_schedule_pdf(name):
            return (0, name)
        if "spec" in name.lower():
            return (2, name)
        return (1, name)

    for name in sorted(pdfs, key=sort_key):
        path = os.path.join(dataset_dir, name)
        try:
            print(f"Reading PDF: {name}")
            pages = iter_pdf_pages(path)
        except Exception as exc:  # noqa: BLE001
            print(f"could not read {name}: {exc}")
            continue

        if is_schedule_pdf(name):
            print(f"Parsing schedule rows (1 item per Tag): {name}")
            for idx, page_text in enumerate(pages, start=1):
                try:
                    labeled = label_schedule_page(
                        document=name, page=idx, page_text=page_text
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  label failed: {exc}")
                    continue
                print(f"  page {idx}: {len(labeled)} row-item(s)")
                add_claims(
                    item_map,
                    name,
                    [{"key": c["key"], "text": c["text"], "page": idx} for c in labeled],
                )
            continue

        kind = "drawing" if is_drawing_pdf(name) else "doc"
        if is_drawing_pdf(name):
            drawing_pages[name] = pages
        print(f"Labeling {name} as {kind} ({len(pages)} page(s) via pypdf)…")
        for idx, page_text in enumerate(pages, start=1):
            try:
                if is_drawing_pdf(name):
                    labeled = label_drawing_page(
                        document=name,
                        page=idx,
                        page_text=page_text,
                        existing_summary=summarize_map(item_map),
                    )
                    # Deterministic: also emit tags/MEP marks regex-found on this page
                    for tag in tags_in_text(page_text):
                        labeled.append(
                            {
                                "key": tag,
                                "text": f"Tag {tag} present in drawing page text.",
                            }
                        )
                else:
                    labeled = label_generic_page(
                        document=name,
                        page=idx,
                        page_text=page_text,
                        existing_summary=summarize_map(item_map),
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  label failed {name} p{idx}: {exc}")
                continue
            # de-dupe labeled
            seen: set[tuple[str, str]] = set()
            uniq = []
            for c in labeled:
                sig = (c["key"], c["text"])
                if sig not in seen:
                    seen.add(sig)
                    uniq.append(c)
            print(f"  page {idx}: {len(uniq)} claim(s)")
            add_claims(
                item_map,
                name,
                [{"key": c["key"], "text": c["text"], "page": idx} for c in uniq],
            )

    expand_room_finish_links(item_map)
    link_schedule_marks_on_drawings(item_map, drawing_pages)
    return item_map


def multi_document_keys(item_map: ItemMap) -> list[str]:
    return sorted(
        key
        for key, claims in item_map.items()
        if len({c["document"] for c in claims}) >= 2
    )


def intra_document_conflict_keys(item_map: ItemMap) -> list[str]:
    """Keys with ≥2 distinct claim texts inside the same document (e.g. duplicate Tag rows)."""
    keys: list[str] = []
    for key, claims in item_map.items():
        if ROOM_KEY_RE.match(key):
            continue  # room paraphrases are not schedule conflicts
        by_doc: dict[str, set[str]] = {}
        for c in claims:
            # Focus intra-doc checks on schedule rows (duplicate Tags) and
            # product-like tags, not every drawing note pair.
            if not is_schedule_pdf(c["document"]) and not re.match(
                r"^[A-Za-z]{1,5}-\d+", key
            ):
                continue
            by_doc.setdefault(c["document"], set()).add(c["text"].strip().lower())
        if any(len(texts) >= 2 for texts in by_doc.values()):
            keys.append(key)
    return sorted(keys)


def conflict_candidate_keys(item_map: ItemMap) -> list[str]:
    return sorted(set(multi_document_keys(item_map)) | set(intra_document_conflict_keys(item_map)))


def classify_category(description: str, suggested: str) -> str:
    if suggested in CATEGORIES and suggested != "cross-document-conflict":
        return suggested
    d = description.lower()
    if re.search(r"\b\d+(\.\d+)?\s*(gpm|gpf|cfm|kw|kva|amp|psi)\b", d):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", d)]
        if len(nums) >= 2 and any(
            abs(a - b) > 1e-9 and (max(a, b) / max(min(a, b), 1e-9) >= 5)
            for a in nums
            for b in nums
        ):
            return "unit-error"
    if suggested in CATEGORIES:
        return suggested
    return "cross-document-conflict"


def resolve_conflict(key: str, claims: list[dict[str, Any]]) -> dict[str, Any] | None:
    claims_blob = json.dumps(claims, ensure_ascii=False, indent=2)
    docs = {c["document"] for c in claims}
    scope = (
        "across different documents"
        if len(docs) >= 2
        else "within the SAME document (duplicate/contradictory rows or notes)"
    )
    prompt = f"""You compare claims about the same construction item {scope}.

Item key: {key}
Claims:
{claims_blob}

A CONFLICT means contradictory measurable facts (e.g. 45 min vs 90 min fire rating,
5.0 gpm vs 0.5 gpm, different manufacturers for the same exclusive tag, conflicting
colors/models, two schedule rows for the same Tag with different products).
NOT a conflict: synonyms, restating the same value, compatible extras, or "tag appears on drawing"
vs a full schedule row (those are compatible unless product/color/rating disagrees).

If conflict: which document has the INCORRECT info?
- Cross-doc: schedule/drawing usually wrong when it violates a written requirement;
  if schedule vs drawing product/color disagree, prefer citing the drawing OR schedule
  that looks like the injected error (state both values clearly).
- Same-doc duplicate Tag rows: cite that schedule file; describe both manufacturers/products.

Respond with ONLY JSON:
{{
  "conflict": true|false,
  "document": "<file with incorrect info, required if conflict>",
  "category": "cross-document-conflict|unit-error|code-violation|missing-item",
  "location": "<page/section/table hint; include item key and distinctive values>",
  "description": "<one sentence: include the item key and quote wrong value and correct value>"
}}

Category: "unit-error" for magnitude/decimal slips; else usually "cross-document-conflict".
If no conflict: {{"conflict": false}}
"""
    raw = call_llm(prompt)
    data = parse_json_object(raw)
    if not data.get("conflict"):
        return None

    document = str(data.get("document", "")).strip()
    description = str(data.get("description", "")).strip()
    location = str(data.get("location", "")).strip()
    category = classify_category(description, str(data.get("category", "")).strip())
    if not document or not description:
        return None
    if key not in description:
        description = f"{key}: {description}"

    claim_docs = {c["document"] for c in claims}
    if document not in claim_docs:
        lowered = {d.lower(): d for d in claim_docs}
        document = lowered.get(document.lower(), document)
        if document not in claim_docs:
            for d in claim_docs:
                if "schedule" in d.lower():
                    document = d
                    break
            else:
                document = next(iter(claim_docs))

    error: dict[str, Any] = {
        "document": document,
        "category": category,
        "description": description,
    }
    if location:
        error["location"] = location
    else:
        pages = sorted({c["page"] for c in claims if c["document"] == document})
        if pages:
            error["location"] = f"page {pages[0]}, {key}"
    return error


def find_conflicts(item_map: ItemMap) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    keys = conflict_candidate_keys(item_map)
    multi = set(multi_document_keys(item_map))
    intra = set(intra_document_conflict_keys(item_map))
    print(
        f"Conflict check on {len(keys)} key(s) "
        f"(multi-doc={len(multi)}, intra-doc={len(intra)})…"
    )
    for key in keys:
        try:
            err = resolve_conflict(key, item_map[key])
        except Exception as exc:  # noqa: BLE001
            print(f"  conflict failed {key}: {exc}")
            continue
        if err:
            print(f"  CONFLICT {key}: {err['description']}")
            errors.append(err)
        else:
            print(f"  ok {key}")
    return errors


def main() -> None:
    print(f"DATASET_DIR={DATASET_DIR}")
    print(f"MODEL={MODEL}")
    item_map: ItemMap = {}
    errors: list[dict[str, Any]] = []
    try:
        item_map = build_item_map(DATASET_DIR)
        print(f"Map has {len(item_map)} item key(s)")
        errors = find_conflicts(item_map)
        print(f"Reported {len(errors)} error(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}")

    debug_path = os.environ.get("ITEM_MAP_PATH")
    if debug_path:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(item_map, f, indent=2)
        print(f"Wrote item map to {debug_path}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"errors": errors}, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
