#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Builder (Agents-SDK version)
---------------------------------
Extracts structured rules JSON from a Hebrew regulatory PDF using the OpenAI
Agents SDK. Designed for Windows-friendly paths and rich debug outputs so you
can see exactly what the LLM returned every N pages.

Dependencies (suggested versions):
    pip install -U openai-agents pydantic PyPDF2 tqdm python-dotenv

Usage (PowerShell):
    $env:OPENAI_API_KEY="<your_key>"
    python scripts/data_builder_agents_sdk.py `
      --pdf "data\raw\18-07-2022_4.2A.pdf" `
      --out_dir "data\curated" `
      --model "gpt-4o-mini" `
      --window_pages 2 `
      --stride 1 `
      --max_pages 10 `
      --debug_every_pages 5 `
      --debug_dir debug

Outputs:
  - rules_flat.json : flat list of rules
  - rules_tree.json : hierarchical tree reconstructed from numbering
  - debug/*.prompt.txt : prompt window sent to the LLM (every N pages)
  - debug/*.raw.txt    : raw JSON text the LLM returned (every N pages)

Notes:
- This version uses two Agents:
  1) text_agent → returns RAW JSON text (no output_type) so we can save the exact
     model output for debugging.
  2) parse_agent → returns structured output (Pydantic) of the same JSON schema.
- If the structured parse fails, we attempt a manual Pydantic validation of the
  raw JSON (with helpful coercions like generating missing ids and fixing
  citations to objects).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PyPDF2 import PdfReader
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError

# Agents SDK
from agents import Agent, Runner

# ------------------------------
# Models / Schemas
# ------------------------------
class Bullet(BaseModel):
    idx: Optional[int] = Field(None, description="Bullet index like ')1(' → 1")
    text: str

class Citation(BaseModel):
    page: int

class ConditionCtx(BaseModel):
    text: str
    value: float
    hint: Optional[str] = Field(None, description="'min' if ≥/ומעלה; 'max' if ≤/עד; else null")

class Conditions(BaseModel):
    area_ctx: List[ConditionCtx] = Field(default_factory=list)
    seats_ctx: List[ConditionCtx] = Field(default_factory=list)
    temps_c_ctx: List[ConditionCtx] = Field(default_factory=list)
    min_area_m2: Optional[float] = None
    max_area_m2: Optional[float] = None
    min_seats: Optional[int] = None
    max_seats: Optional[int] = None

class Rule(BaseModel):
    id: str
    number: str
    title: str
    category: str
    text: str
    bullets: List[Bullet] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    conditions: Conditions = Field(default_factory=Conditions)
    features: List[str] = Field(default_factory=list)
    children: List["Rule"] = Field(default_factory=list)

class RulesBatch(BaseModel):
    rules: List[Rule] = Field(default_factory=list)

# ------------------------------
# Utilities
# ------------------------------
CHAPTER_RE = re.compile(r"^\s*פרק\s*(\d+)\s*[-–]\s*(.+?)\s*$")

CATEGORY_MAP = [
    (re.compile(r"משטרת\s*ישראל"), "משטרה"),
    (re.compile(r"משרד\s*הבריאות"), "בריאות"),
    (re.compile(r"הרשות\s*הארצית\s*לכבאות\s*והצלה"), "כבאות"),
    (re.compile(r"תנאים\s*רוחביים"), "תנאים רוחביים"),
    (re.compile(r"הגדרות\s*כלליות|כלליות|הגדרות"), "הגדרות"),
]

def detect_category(txt: str) -> Optional[str]:
    for rx, lab in CATEGORY_MAP:
        if rx.search(txt):
            return lab
    return None
def brace(s: str) -> str:
    # escape literal braces so .format() won't treat them as placeholders
    return s.replace("{", "{{").replace("}", "}}")



def pages_to_windows(pages: List[Dict[str, Any]], window: int = 2, stride: int = 1) -> List[Tuple[List[int], str]]:
    out: List[Tuple[List[int], str]] = []
    N = len(pages)
    i = 0
    while i < N:
        j = min(i + window, N)
        idxs = list(range(i, j))
        combined = []
        for k in idxs:
            pg_no = pages[k]["page"]
            combined.append(f"\n===== [PAGE {pg_no}] =====\n" + pages[k]["text"])
        out.append((idxs, "\n".join(combined)))
        i += stride
    return out


def read_pdf_per_page(path: Path, max_pages: int | None = None) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    pages: List[Dict[str, Any]] = []
    for i, p in enumerate(reader.pages, start=1):
        if max_pages and i > max_pages:
            break
        txt = p.extract_text() or ""
        pages.append({"page": i, "text": txt})
    return pages


def build_page_catalog(pages: List[Dict[str, Any]]):
    page_cats: Dict[int, str] = {}
    for it in pages:
        page = it["page"]
        txt = it["text"]
        head = "\n".join(txt.splitlines()[:12])
        m = CHAPTER_RE.search(head)
        if m:
            ch_title = m.group(2)
            cat = detect_category(ch_title) or ch_title
            page_cats[page] = cat
    last_cat = None
    for p in [x["page"] for x in pages]:
        if p in page_cats:
            last_cat = page_cats[p]
        elif last_cat:
            page_cats[p] = last_cat
        else:
            page_cats[p] = "לא ידוע"
    return page_cats

# ------------------------------
# Prompts (few-shot with explicit schema)
# ------------------------------
SYSTEM_PROMPT = (
    "You are a meticulous Hebrew regulatory parser. Read the Hebrew text and extract ONLY numbered regulations "
    "and subitems. Output STRICT JSON that conforms to the schema. Do not invent facts. Keep Hebrew as-is."
)

FEWSHOT_EXAMPLE = r'''
{
  "rules": [
    {
      "id": "r_4_1_1",
      "number": "4.1.1",
      "title": "כותרת קצרה",
      "category": "כבאות",
      "text": "טקסט הסעיף…",
      "bullets": [
        { "idx": 1, "text": "דרישה א" },
        { "idx": 2, "text": "דרישה ב" }
      ],
      "citations": [{ "page": 10 }],
      "conditions": {
        "area_ctx": [{ "text": "שטח של 100 מ\"ר ומעלה", "value": 100, "hint": "min" }],
        "seats_ctx": [{ "text": "תפוסה מעל 200 איש", "value": 200, "hint": "min" }],
        "temps_c_ctx": [],
        "min_area_m2": 100,
        "max_area_m2": null,
        "min_seats": 200,
        "max_seats": null
      },
      "features": ["uses_gas","frying"]
    }
  ]
}
'''.strip()

HUMAN_PROMPT = """
מסמך: מפרט רישוי עסקים לבתי אוכל (עברית). לפניך חלון עמודים עם טקסט.

הנחיות קצרות:
• חלץ רק סעיפים ממוספרים (למשל "3.6.1", "4.10.3", "1.1", תתי־פרק וכו').
• מלא את כל השדות כנדרש; אם אין כותרת – גזור מהמשפט הראשון.
• bullets הם אובייקטים מסוג { "idx": <int>, "text": <str> }.
• citations הם אובייקטים מסוג { "page": <int> }.
• conditions: דלל אזכורי מ״ר/תפוסה/טמפ׳ לתוך *_ctx עם hint="min"/"max" לפי ניסוח ("ומעלה"/"עד" וכו'), וגם קבע min/max_*.

דוגמת פלט JSON קשיח (שמור על המבנה והסוגים):
{fewshot}

חלון העמודים:
{window_text}

מספרי העמודים בחלון: {pages_list}

החזר אך ורק JSON תקין עם המפתח "rules".
""".strip()


# ------------------------------
# Merge / Reconcile
# ------------------------------
FEATURES_ORDER = [
    "alcohol","uses_gas","serves_meat","delivery","frying","cctv","smoking_sign","water","sewage","food_cold","food_hot"
]

@dataclass
class AccRule:
    number: str
    title: str = ""
    category: str = "לא ידוע"
    text: str = ""
    bullets: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    area_ctx: List[Dict[str, Any]] = field(default_factory=list)
    seats_ctx: List[Dict[str, Any]] = field(default_factory=list)
    temps_ctx: List[Dict[str, Any]] = field(default_factory=list)

    def to_rule(self) -> Dict[str, Any]:
        def infer_bounds(ctx_list, is_int=False):
            vmin = None
            vmax = None
            vals = []
            for it in ctx_list:
                if "value" not in it:
                    continue
                v = int(round(float(it.get("value", 0)))) if is_int else float(it.get("value", 0))
                hint = it.get("hint")
                if hint == "min":
                    vmin = v if vmin is None else min(vmin, v)
                elif hint == "max":
                    vmax = v if vmax is None else max(vmax, v)
                vals.append(v)
            if (vmin is None and vmax is None) and vals:
                vmin = min(vals)
                vmax = max(vals)
            return vmin, vmax

        min_area, max_area = infer_bounds(self.area_ctx, is_int=False)
        min_seats, max_seats = infer_bounds(self.seats_ctx, is_int=True)
        feats = sorted(list({f for f in self.features if f in FEATURES_ORDER}), key=lambda x: FEATURES_ORDER.index(x))

        return {
            "id": f"r_{self.number.replace('.', '_')}",
            "number": self.number,
            "title": self.title or self.text[:90],
            "category": self.category,
            "text": self.text.strip(),
            "bullets": self.bullets,
            "citations": self.citations,
            "conditions": {
                "area_ctx": self.area_ctx,
                "seats_ctx": self.seats_ctx,
                "temps_c_ctx": self.temps_ctx,
                "min_area_m2": min_area,
                "max_area_m2": max_area,
                "min_seats": min_seats,
                "max_seats": max_seats,
            },
            "features": feats,
            "children": [],
        }


def merge_batches(batches: List[RulesBatch]) -> Dict[str, AccRule]:
    acc: Dict[str, AccRule] = {}
    for b in batches:
        for r in b.rules:
            num = r.number.strip()
            if not num:
                continue
            if num not in acc:
                acc[num] = AccRule(number=num)
            ar = acc[num]
            if r.title and len(r.title) > len(ar.title):
                ar.title = r.title
            if r.category and (ar.category == "לא ידוע" or ar.category == ""):
                ar.category = r.category
            if r.text and len(r.text) > len(ar.text):
                ar.text = r.text
            ar.bullets.extend([b.dict() for b in r.bullets])
            ar.citations.extend([c.dict() for c in r.citations])
            ar.features.extend(r.features)
            ar.area_ctx.extend([ac.dict() for ac in r.conditions.area_ctx])
            ar.seats_ctx.extend([sc.dict() for sc in r.conditions.seats_ctx])
            ar.temps_ctx.extend([tc.dict() for tc in r.conditions.temps_c_ctx])
    for ar in acc.values():
        seen = set()
        uniq = []
        for c in ar.citations:
            p = c.get("page")
            if p not in seen:
                uniq.append({"page": p})
                seen.add(p)
        ar.citations = sorted(uniq, key=lambda x: x["page"])  # type: ignore
    return acc


def build_tree(flat_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {r["number"]: r for r in flat_rules}
    roots: List[Dict[str, Any]] = []

    def parent_number(num: str) -> Optional[str]:
        if "." not in num:
            return None
        return ".".join(num.split(".")[:-1])

    for r in nodes.values():
        r["children"] = []

    for num, r in nodes.items():
        p = parent_number(num)
        if p and p in nodes:
            nodes[p]["children"].append(r)
        else:
            roots.append(r)

    def sort_key(n):
        try:
            return [int(x) for x in n["number"].split(".")]
        except Exception:
            return [math.inf]

    def sort_rec(arr):
        arr.sort(key=sort_key)
        for n in arr:
            sort_rec(n.get("children", []))

    sort_rec(roots)
    return roots

# ------------------------------
# Coercion helpers for flaky outputs
# ------------------------------

def coerce_rules_payload(payload: dict, pages_list: List[int]) -> RulesBatch:
    """Fix common LLM mistakes: missing id, numeric citations, missing keys."""
    rules = []
    items = payload.get("rules", []) if isinstance(payload, dict) else []
    for it in items:
        if not isinstance(it, dict):
            continue
        # id
        num = (it.get("number") or "").strip()
        if not it.get("id") and num:
            it["id"] = f"r_{num.replace('.', '_')}"
        # title/category/text defaults
        it.setdefault("title", "")
        it.setdefault("category", "לא ידוע")
        it.setdefault("text", "")
        # bullets normalize
        bl = it.get("bullets", [])
        it["bullets"] = [b if isinstance(b, dict) else {"idx": None, "text": str(b)} for b in bl]
        # citations normalize
        cits = it.get("citations", [])
        norm_cits = []
        if isinstance(cits, list):
            for c in cits:
                if isinstance(c, dict) and "page" in c:
                    norm_cits.append({"page": int(c["page"])})
                elif isinstance(c, int):
                    norm_cits.append({"page": c})
                elif isinstance(c, str) and c.isdigit():
                    norm_cits.append({"page": int(c)})
        if not norm_cits:
            # if empty, at least attribute to visible pages
            norm_cits = [{"page": p} for p in pages_list]
        it["citations"] = norm_cits
        # conditions defaults
        cond = it.get("conditions") or {}
        cond.setdefault("area_ctx", [])
        cond.setdefault("seats_ctx", [])
        cond.setdefault("temps_c_ctx", [])
        cond.setdefault("min_area_m2", None)
        cond.setdefault("max_area_m2", None)
        cond.setdefault("min_seats", None)
        cond.setdefault("max_seats", None)
        it["conditions"] = cond
        # features default
        it.setdefault("features", [])
        it.setdefault("children", [])
        rules.append(it)
    return RulesBatch.model_validate({"rules": rules})

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--window_pages", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pages", type=int, default=0, help="If >0, only read first N pages of the PDF")
    ap.add_argument("--debug_every_pages", type=int, default=10)
    ap.add_argument("--debug_dir", default="debug")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = Path(args.debug_dir)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load PDF as plain text per page (Windows-safe)
    pages = read_pdf_per_page(pdf_path, max_pages=args.max_pages if args.max_pages > 0 else None)
    page_cats = build_page_catalog(pages)

    # 2) Make sliding windows
    windows = pages_to_windows(pages, window=args.window_pages, stride=args.stride)

    # 3) Define two agents (text first for raw, parse second for structured)
    text_agent = Agent(
        name="raw-json-extractor",
        instructions=SYSTEM_PROMPT,
        model=args.model,
    )
    parse_agent = Agent(
        name="rules-parser",
        instructions="Return JSON ONLY that matches the declared output schema.",
        model=args.model,
        output_type=RulesBatch,
    )

    batches: List[RulesBatch] = []

    for idx, (idxs, combined_text) in enumerate(tqdm(windows, desc="Extracting windows"), start=1):
        pages_list = [pages[i]["page"] for i in idxs]
        cats = [page_cats.get(p, "לא ידוע") for p in pages_list]
        cat_hint = "\n[קטגוריות משוערות לעמודים]: " + ", ".join(f"p{p}:{c}" for p, c in zip(pages_list, cats))
        window_text = combined_text + cat_hint

        # Build the user prompt
        prompt_text = f"""{HUMAN_PROMPT}

          חלון העמודים:
           {window_text}

          מספרי העמודים בחלון: {pages_list}

           דוגמה:
          {FEWSHOT_EXAMPLE}
            """
        



        # Debug: write prompt every N pages (based on last page number in window)
        last_p = pages_list[-1]
        if args.debug_every_pages > 0 and (last_p % args.debug_every_pages == 0):
            (dbg_dir / f"win_{pages_list[0]}_{last_p}.prompt.txt").write_text(prompt_text, encoding="utf-8")

        # Step A: get raw JSON text
        raw_result = Runner.run_sync(text_agent, prompt_text)
        raw_text = str(raw_result.final_output or "").strip()
        # If the model tried to chat instead of JSON, try to locate a JSON block
        if not raw_text.startswith("{"):
            s = raw_text.find("{")
            e = raw_text.rfind("}")
            if s != -1 and e != -1 and e > s:
                raw_text = raw_text[s:e+1]
        if args.debug_every_pages > 0 and (last_p % args.debug_every_pages == 0):
            (dbg_dir / f"win_{pages_list[0]}_{last_p}.raw.txt").write_text(raw_text, encoding="utf-8")

        # Try JSON load
        payload: dict = {"rules": []}
        try:
            payload = json.loads(raw_text)
        except Exception:
            # If it failed, ask parse_agent directly (it will enforce schema and may ignore invalid raw)
            pass

        # Step B: parse to Pydantic either via parse_agent or manual coercion
        parsed: Optional[RulesBatch] = None
        if payload and isinstance(payload, dict) and "rules" in payload:
            try:
                parsed = coerce_rules_payload(payload, pages_list)
            except ValidationError as ve:
                # Leave parsed=None to try parse_agent
                (dbg_dir / f"win_{pages_list[0]}_{last_p}.coerce_error.txt").write_text(str(ve), encoding="utf-8")

        if parsed is None:
            try:
                parsed = Runner.run_sync(parse_agent, prompt_text).final_output  # type: ignore
            except Exception as e:
                # Last resort: empty batch
                (dbg_dir / f"win_{pages_list[0]}_{last_p}.parse_error.txt").write_text(str(e), encoding="utf-8")
                parsed = RulesBatch(rules=[])

        batches.append(parsed)

    # 4) Merge across windows
    merged = merge_batches(batches)

    # 5) Build flat list
    flat_rules = [r.to_rule() for r in merged.values()]

    # 6) Build tree
    tree = build_tree(flat_rules)

    # 7) Save
    (out_dir / "rules_flat.json").write_text(json.dumps(flat_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rules_tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✔ Done. Rules: flat={len(flat_rules)}; tree roots={len(tree)} → {out_dir}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY is not set. Export it before running.")
    main()
