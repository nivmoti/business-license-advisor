#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Builder (Agents SDK + OpenAI) — extract structured rules JSON from Hebrew PDF
---------------------------------------------------------------------------------
- Input: regulatory PDF (e.g., data/raw/18-07-2022_4.2A.pdf)
- Output:
    * <out_dir>/rules_flat.json  — flat list of rules
    * <out_dir>/rules_tree.json  — hierarchical tree reconstructed from numbering

Key points:
- Uses only OpenAI Responses API (via Agents SDK/openai) — no LangChain pipeline.
- Sliding windows over pages to reduce boundary errors.
- Strong, example-driven prompt (Hebrew) with strict JSON schema.
- Debug: save the raw LLM output per window to files for inspection.
- Resilient JSON parsing & normalization (fills missing id/citations/conditions).

Usage:
  python scripts/data_builder_agents_sdk.py \
      --pdf "data/raw/18-07-2022_4.2A.pdf" \
      --out_dir "data/curated" \
      --model "gpt-4o-mini" \
      --window_pages 2 \
      --stride 1 \
      --max_pages 0 \
      --debug_every_pages 5 \
      --debug_dir debug

Requires:
  pip install openai tqdm langchain-community pydantic==2.*
  # (langchain-community only for PyPDFLoader to read per-page text)
  # Or replace PyPDFLoader with any PDF reader you prefer.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader

# ------------------------------
# Minimal Pydantic models (v2)
# ------------------------------
class Bullet(BaseModel):
    idx: Optional[int] = Field(None)
    text: str

class Citation(BaseModel):
    page: int

class ConditionCtx(BaseModel):
    text: str
    value: float
    hint: Optional[str] = Field(None, description="'min'/'max'/null")

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
# Prompt (Hebrew) — with example, JSON schema & instructions
# ------------------------------
SYSTEM_PROMPT = (
    "You are a meticulous Hebrew regulatory parser. Read the Hebrew text and extract ONLY numbered regulations "
    "and subitems. Output STRICT JSON that conforms to the schema. Do not invent facts. Keep Hebrew as-is."
)

# Use doubled braces to escape literal braces for .format()
HUMAN_PROMPT = (
    """
אתה סוכן מומחה בחילוץ רגולציה בעברית מתוך קובץ PDF.
קבל טקסט של חלון עמודים והחזר JSON תקין לפי הסכמה.

הוראות חילוץ:

1) כלול אך ורק סעיפים שמתחילים במספור ברור ("3.8.1", "4.10.3", "1.1" וכו').
   - אל תמציא סעיפים חדשים.
   - אל תכלול טקסט מבוא או כותרות שאינן ממוספרות.

2) לכל סעיף:
   - id: "r_" + number כשהנקודות מוחלפות בקו תחתון. לדוגמה: "3.8.1" → "r_3_8_1".
   - number: המספור המדויק (מחרוזת).
   - title: אם יש כותרת, רשום אותה; אחרת גזור כותרת קצרה ממשפט הפתיחה.
   - category: קטגוריה (משטרה/בריאות/כבאות/תנאים רוחביים/הגדרות/לא ידוע).
   - text: הטקסט המלא של הסעיף ללא קישוטים.

3) bullets:
   - אם יש תתי־סעיפים בסוגריים ")1(", ")2(" וכו', החזר במבנה: {{ "idx": 1, "text": "..." }}.

4) citations:
   - רשימת אובייקטים {{ "page": <int> }} עם מספרי העמודים בהם הופיע הסעיף.

5) conditions:
   - אתר אזכורים של שטח/מ"ר, תפוסה/מקומות ישיבה/סועדים/קיבולת, וטמפרטורה (°C).
   - לכל אזכור צור אובייקט {{ "text": "...", "value": מספר, "hint": "min"/"max"/null }}.
     * "ומעלה", "לפחות", "מינימום" → hint="min"
     * "עד", "לכל היותר", "מקסימום" → hint="max"
   - חשב min/max_* אם אפשר.

   דוגמה:
   מהסעיף "עסק עד 200 מקומות ישיבה…" →
   conditions.seats_ctx = [ {{ "text": "עד 200 מקומות ישיבה", "value": 200, "hint": "max" }} ]
   conditions.max_seats = 200

6) features:
   - הוסף תגיות מתאימות אם מופיעות במפורש:
     ["alcohol","uses_gas","serves_meat","delivery","frying","cctv","smoking_sign","water","sewage","food_cold","food_hot"].
   דוגמה: "ללא מכירה, הגשה וצריכה של משקאות משכרים" → feature = "alcohol" (הטקסט מציין "ללא").

7) children: תמיד החזר [] בשלב זה.

פלט JSON סופי:
{{
  "rules": [
    {{
      "id": "r_<digits_with_underscores>",
      "number": "<x.y.z>",
      "title": "<short title>",
      "category": "<קטגוריה או 'לא ידוע'>",
      "text": "<full body>",
      "bullets": [ {{ "idx": 1, "text": "..." }} ],
      "citations": [ {{ "page": 10 }} ],
      "conditions": {{
        "area_ctx": [ {{ "text": "עד 100 מ""ר", "value": 100, "hint": "max" }} ],
        "seats_ctx": [ {{ "text": "עד 200 מקומות ישיבה", "value": 200, "hint": "max" }} ],
        "temps_c_ctx": [ {{ "text": "לפחות 4°C", "value": 4, "hint": "min" }} ],
        "min_area_m2": null,
        "max_area_m2": 100,
        "min_seats": null,
        "max_seats": 200
      }},
      "features": ["alcohol"],
      "children": []
    }}
  ]
}}

דוגמת פלט מלאה לסעיף אמיתי:
{{
  "rules": [
    {{
      "id": "r_3_8_1",
      "number": "3.8.1",
      "title": "הקלות – פטור לעסק עד 200 מקומות ללא אלכוהול",
      "category": "תנאים רוחביים",
      "text": "עסק עד 200 מקומות ישיבה ללא מכירה, הגשה וצריכה של משקאות משכרים פטור מהדרישות המופיעות בפריט זה.",
      "bullets": [],
      "citations": [ {{ "page": 25 }} ],
      "conditions": {{
        "area_ctx": [],
        "seats_ctx": [ {{ "text": "עד 200 מקומות ישיבה", "value": 200, "hint": "max" }} ],
        "temps_c_ctx": [],
        "min_area_m2": null,
        "max_area_m2": null,
        "min_seats": null,
        "max_seats": 200
      }},
      "features": ["alcohol"],
      "children": []
    }}
  ]
}}

---
חלון העמודים:
{window_text}

מספרי העמודים בחלון: {pages_list}

החזר אך ורק JSON תקין (אובייקט אחד עם המפתח "rules").
    """
)

# ------------------------------
# Helpers
# ------------------------------

def load_pdf_pages(pdf_path: Path, max_pages: int = 0) -> List[Dict[str, Any]]:
    loader = PyPDFLoader(pdf_path.as_posix())
    docs = loader.load()
    pages: List[Dict[str, Any]] = []
    for d in docs:
        pno = int(d.metadata.get("page", 0)) + 1
        pages.append({"page": pno, "text": d.page_content})
    pages.sort(key=lambda x: x["page"])  # ensure order
    if max_pages and len(pages) > max_pages:
        pages = pages[:max_pages]
    return pages


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
        def infer_bounds(ctx_list: List[Dict[str, Any]], is_int: bool = False):
            vmin = None
            vmax = None
            vals: List[float] = []
            for it in ctx_list:
                try:
                    v = float(it.get("value", 0))
                except Exception:
                    continue
                hint = it.get("hint")
                if hint == "min":
                    vmin = v if vmin is None else min(vmin, v)
                elif hint == "max":
                    vmax = v if vmax is None else max(vmax, v)
                vals.append(v)
            if (vmin is None and vmax is None) and vals:
                vmin = min(vals)
                vmax = max(vals)
            if is_int:
                vmin = int(vmin) if vmin is not None else None
                vmax = int(vmax) if vmax is not None else None
            return vmin, vmax

        min_area, max_area = infer_bounds(self.area_ctx, is_int=False)
        min_seats, max_seats = infer_bounds(self.seats_ctx, is_int=True)

        return {
            "id": f"r_{self.number.replace('.', '_')}",
            "number": self.number,
            "title": self.title or self.text[:80],
            "category": self.category,
            "text": self.text.strip(),
            "bullets": self.bullets,
            "citations": sorted(self.citations, key=lambda x: x.get("page", 0)),
            "conditions": {
                "area_ctx": self.area_ctx,
                "seats_ctx": self.seats_ctx,
                "temps_c_ctx": self.temps_ctx,
                "min_area_m2": min_area,
                "max_area_m2": max_area,
                "min_seats": min_seats,
                "max_seats": max_seats,
            },
            "features": sorted(list({f for f in self.features})),
            "children": [],
        }


# ------------------------------
# Normalization & Merge
# ------------------------------

def ensure_rule_defaults(r: Dict[str, Any], pages_list: List[int]) -> Dict[str, Any]:
    # id
    number = str(r.get("number", "")).strip()
    rid = r.get("id") or ("r_" + number.replace(".", "_")) if number else "r_missing"

    # citations → list of dicts {page:int}
    cits = r.get("citations", [])
    norm_cits: List[Dict[str, int]] = []
    if isinstance(cits, list):
        for c in cits:
            if isinstance(c, dict) and "page" in c:
                try:
                    norm_cits.append({"page": int(c["page"])})
                except Exception:
                    pass
            elif isinstance(c, int):
                norm_cits.append({"page": c})
    if not norm_cits:  # fallback: attach all current pages
        norm_cits = [{"page": int(p)} for p in pages_list]

    # bullets
    bl = []
    for b in r.get("bullets", []) or []:
        if isinstance(b, dict):
            idx = b.get("idx")
            txt = b.get("text", "")
            bl.append({"idx": int(idx) if idx is not None and str(idx).isdigit() else None, "text": str(txt)})
        else:
            bl.append({"idx": None, "text": str(b)})

    # conditions
    cond = r.get("conditions", {}) or {}
    def norm_ctx(lst):
        out = []
        for it in lst or []:
            try:
                out.append({
                    "text": str(it.get("text", "")),
                    "value": float(it.get("value", 0)),
                    "hint": it.get("hint") if it.get("hint") in ("min", "max") else None,
                })
            except Exception:
                pass
        return out
    area_ctx = norm_ctx(cond.get("area_ctx"))
    seats_ctx = norm_ctx(cond.get("seats_ctx"))
    temps_ctx = norm_ctx(cond.get("temps_c_ctx"))

    res = {
        "id": rid,
        "number": number,
        "title": str(r.get("title", "")),
        "category": str(r.get("category", "לא ידוע")) or "לא ידוע",
        "text": str(r.get("text", "")),
        "bullets": bl,
        "citations": norm_cits,
        "conditions": {
            "area_ctx": area_ctx,
            "seats_ctx": seats_ctx,
            "temps_c_ctx": temps_ctx,
            "min_area_m2": r.get("conditions", {}).get("min_area_m2"),
            "max_area_m2": r.get("conditions", {}).get("max_area_m2"),
            "min_seats": r.get("conditions", {}).get("min_seats"),
            "max_seats": r.get("conditions", {}).get("max_seats"),
        },
        "features": [str(x) for x in (r.get("features", []) or [])],
        "children": [],
    }
    return res


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
            if r.category and (ar.category == "לא ידוע" or not ar.category):
                ar.category = r.category
            if r.text and len(r.text) > len(ar.text):
                ar.text = r.text
            ar.bullets.extend([b.model_dump() for b in r.bullets])
            ar.citations.extend([c.model_dump() for c in r.citations])
            ar.features.extend(r.features)
            ar.area_ctx.extend([ac.model_dump() for ac in r.conditions.area_ctx])
            ar.seats_ctx.extend([sc.model_dump() for sc in r.conditions.seats_ctx])
            ar.temps_ctx.extend([tc.model_dump() for tc in r.conditions.temps_c_ctx])
    # dedupe citations by page
    for ar in acc.values():
        seen = set()
        uniq = []
        for c in ar.citations:
            p = c.get("page")
            if p not in seen:
                uniq.append({"page": p})
                seen.add(p)
        ar.citations = uniq
    return acc


def build_tree(flat_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {r["number"]: r for r in flat_rules if r.get("number")}
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

    def sort_key(n: Dict[str, Any]):
        try:
            return [int(x) for x in n["number"].split(".")]
        except Exception:
            return [math.inf]

    def sort_rec(arr: List[Dict[str, Any]]):
        arr.sort(key=sort_key)
        for n in arr:
            sort_rec(n.get("children", []))

    sort_rec(roots)
    return roots


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the PDF (Hebrew)")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON files")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    ap.add_argument("--window_pages", type=int, default=2, help="Pages per window")
    ap.add_argument("--stride", type=int, default=1, help="Stride between windows")
    ap.add_argument("--max_pages", type=int, default=0, help="Limit pages for quick runs (0=no limit)")
    ap.add_argument("--debug_every_pages", type=int, default=5, help="Save raw LLM output every N pages (0=off)")
    ap.add_argument("--debug_dir", default="debug", help="Debug directory")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir)
    if args.debug_every_pages:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load pages
    pages = load_pdf_pages(pdf_path, max_pages=args.max_pages)

    # 2) Windows
    windows = pages_to_windows(pages, window=args.window_pages, stride=args.stride)

    # 3) OpenAI client
    if not os.environ.get("OPENAI_API_KEY"):
        print("[WARN] OPENAI_API_KEY is not set.")
    client = OpenAI()

    batches: List[RulesBatch] = []

    for w_i, (idxs, text) in enumerate(tqdm(windows, desc="Extracting windows")):
        pages_list = [pages[i]["page"] for i in idxs]
        window_text = text

        user_prompt = HUMAN_PROMPT.format(window_text=window_text, pages_list=pages_list)

        # call model
        resp = client.chat.completions.create(
            model=args.model,
            temperature=1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content if resp.choices else ""

        # Debug raw dump
        if args.debug_every_pages:
            start_page = pages_list[0]
            if (start_page - 1) % args.debug_every_pages == 0:
                (debug_dir / f"window_p{start_page}_raw.txt").write_text(raw or "", encoding="utf-8")

        # Parse JSON from raw
        payload: Dict[str, Any]
        try:
            start = raw.find("{") if raw else -1
            end = raw.rfind("}") if raw else -1
            payload = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {"rules": []}
        except Exception:
            payload = {"rules": []}

        # Normalize and validate batch
        norm_rules: List[Dict[str, Any]] = []
        for r in payload.get("rules", []) or []:
            try:
                norm = ensure_rule_defaults(r, pages_list)
                # Pydantic validation (best-effort)
                rb = Rule.model_validate(norm)
                norm_rules.append(rb.model_dump())
            except Exception:
                # as a last resort, keep minimally filled rule if number exists
                if r.get("number"):
                    norm_rules.append(ensure_rule_defaults(r, pages_list))
        batches.append(RulesBatch.model_validate({"rules": norm_rules}))

    # 4) Merge
    merged = merge_batches(batches)

    # 5) Flat list
    flat_rules = [r.to_rule() for r in merged.values()]

    # 6) Tree
    tree = build_tree(flat_rules)

    # 7) Save
    (out_dir / "rules_flat.json").write_text(json.dumps(flat_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rules_tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✔ Done. Rules: flat={len(flat_rules)}; tree roots={len(tree)} → {out_dir}")


if __name__ == "__main__":
    main()
