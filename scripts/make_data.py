#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract hierarchical rules from 18-07-2022_4.2A.pdf (Hebrew, RTL quirks)
Outputs:
  - data/curated/rules_tree.json
  - data/curated/rules_flat.json

Run:
  python scripts/extract_rules_hierarchical.py \
    --pdf data/raw/18-07-2022_4.2A.pdf \
    --out_dir data/curated
"""

from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise SystemExit("Missing dependency: pip install pymupdf") from e


# ---------- Regex patterns ----------
# Chapter header: e.g. "פרק3 - משטרת ישראל"
CHAPTER_RE = re.compile(r"^\s*פרק\s*(\d+)\s*[-–]\s*(.+?)\s*$")

# Clause numbers: "1.1" , "1.1.1" , "3.6.2" etc. Sometimes a trailing dot appears.
CLAUSE_HEAD_RE = re.compile(r"^\s*(?P<num>(?:\d+\.)+\d+)\s*\.?\s*(?P<title>.*\S)?\s*$")

# Some PDFs may break the number inline; fallback searches anywhere in a line:
INLINE_NUM_RE = re.compile(r"(?<!\d)((?:\d+\.)+\d+)(?!\d)")

# Bullets in Hebrew often appear visually as ")1(" (due to RTL) but text-wise it’s the same sequence:
BULLET_RE = re.compile(r".*?\)(\d+)\( *(.+)$")  # captures ")1(" …text

# Simple paragraph boundary
EMPTY_RE = re.compile(r"^\s*$")

# Numeric capture (with comma/point)
NUM_RE = r"(\d+(?:[.,]\d+)?)"

# Conditions (short-window contexts)
AREA_RE = re.compile(rf"(?P<ctx>.{{0,40}}?{NUM_RE}\s*מ\"?ר.{{0,40}})")
# Seats: match patterns like "200 מקומות", "200 איש", "תפוסתו200 איש ומעלה", "200 אנשים"
SEATS_RE = re.compile(rf"(?P<ctx>.{{0,60}}?{NUM_RE}\s*(?:איש|אנשים|מקומ(?:ות|י)?|מקום|תפוס(?:ה|תו))(?:\s*(?:ו?מעל(?:ה)?|ומעלה|לפחות|עד))?.{{0,60}})")
TEMP_RE = re.compile(rf"(?P<ctx>.{{0,40}}?{NUM_RE}\s*מעלות?\s*צלזיוס.{{0,40}})")

# Features
FEATURE_PATTERNS = {
    "alcohol": re.compile(r"משקאות\s*משכרים|אלכוהול"),
    "uses_gas": re.compile(r"\bגז\b|בלוני\s*גז|מערכת\s*גז"),
    "serves_meat": re.compile(r"\bבשר\b|מוצרי\s*בשר"),
    "delivery": re.compile(r"משלוח(?:ים)?"),
    "frying": re.compile(r"טיגון|מטגן"),
    "cctv": re.compile(r"טמ\"?ס|מצלמות.*מעגל\s*סגור|CCTV"),
    "smoking_sign": re.compile(r"איסור\s*העישון|מניעת\s*העישון|שילוט.*עישון"),
    "water": re.compile(r"מי\s*שתייה|אספקת\s*מים"),
    "sewage": re.compile(r"שפכים|ביוב|מפריד\s*שומן"),
    "food_cold": re.compile(r"\+?\s*5\s*מעלות?\s*צלזיוס"),
    "food_hot": re.compile(r"65\s*מעלות?\s*צלזיוס"),
}

# Category detection by chapter title
CATEGORY_KEYS = [
    (re.compile(r"משטרת\s*ישראל"), "משטרה"),
    (re.compile(r"משרד\s*הבריאות"), "בריאות"),
    (re.compile(r"הרשות\s*הארצית\s*לכבאות\s*והצלה"), "כבאות"),
    (re.compile(r"תנאים\s*רוחביים"), "תנאים רוחביים"),
    (re.compile(r"הגדרות\s*כלליות|כלליות|הגדרות"), "הגדרות"),
]


# ---------- Utilities ----------
def normalize_line(s: str) -> str:
    # Decompose Unicode and strip control chars; keep Hebrew/RTL characters.
    s = unicodedata.normalize("NFKC", s).replace("\u200f", "").replace("\u202b", "").replace("\u202c", "")
    return s.rstrip()

def detect_category(chapter_title: str) -> str:
    for rx, label in CATEGORY_KEYS:
        if rx.search(chapter_title):
            return label
    return chapter_title or "לא ידוע"

def short_context_list(regex: re.Pattern, text: str) -> List[Dict[str, Any]]:
    out = []
    for m in regex.finditer(text):
        ctx = m.group("ctx").strip()
        # number extraction: first numeric in ctx
        nm = re.search(NUM_RE, ctx)
        if not nm: 
            continue
        val = float(nm.group(1).replace(",", "."))
        # heuristic: negative if "מינוס" appears nearby
        qual = None
        # qualifier detection: 'מעלה' / 'מעל' / 'ומעלה' / 'לפחות' => minimum
        if re.search(r"\b(ומעלה|מעלה|מעל|לפחות|לפחות)\b", ctx):
            qual = "min"
        # 'עד' nearby often indicates a maximum
        if re.search(r"\bעד\b", ctx):
            qual = "max"
        # 'מינימום' also indicates minimum
        if re.search(r"\bמינימום\b", ctx):
            qual = "min"
        # normalize negative mention
        if re.search(r"מינוס", ctx):
            val = -abs(val)
        out.append({"text": ctx, "value": val, "qual": qual})
    return out

def extract_conditions(text: str) -> Dict[str, Any]:
    # windowed contexts (cleaner than grabbing whole paragraphs)
    areas = short_context_list(AREA_RE, text)
    seats = short_context_list(SEATS_RE, text)
    temps = short_context_list(TEMP_RE, text)

    cond = {
        "area_ctx": areas,       # list of {"text","value"} (m2)
        "seats_ctx": seats,      # list of {"text","value"} (count)
        "temps_c_ctx": temps,    # list of {"text","value"} (°C)
        # convenience min/max if clearly inferable:
        "min_area_m2": None,
        "max_area_m2": None,
        "min_seats": None,
        "max_seats": None,
    }

    def update_minmax(lst, is_seats=False):
        if not lst:
            return (None, None)
        vals = []
        mins = []
        maxs = []
        for it in lst:
            try:
                v = int(it["value"]) if is_seats else float(it["value"])
                vals.append(v)
                if it.get("qual") == "min":
                    mins.append(v)
                if it.get("qual") == "max":
                    maxs.append(v)
            except:
                pass
        if not vals:
            return (None, None)
        # Prefer explicit qualifiers
        if mins:
            return (min(mins), max(vals))
        if maxs:
            return (min(vals), max(maxs))
        return (min(vals), max(vals))

    cond["min_area_m2"], cond["max_area_m2"] = update_minmax(areas, is_seats=False)
    cond["min_seats"],  cond["max_seats"]  = update_minmax(seats,  is_seats=True)
    return cond

def extract_features(text: str) -> List[str]:
    feats = []
    for name, rx in FEATURE_PATTERNS.items():
        if rx.search(text):
            feats.append(name)
    return feats

def title_from_text(txt: str, fallback: str) -> str:
    s = txt.strip()
    if not s: 
        return fallback
    # Take first sentence-ish or up to 90 chars
    s = re.split(r"[.:;]\s+", s, maxsplit=1)[0]
    return s[:90]

def mk_id(prefix: str, number: str) -> str:
    safe = number.replace(".", "_")
    return f"{prefix}_{safe}"


# ---------- Data structures ----------
class Node:
    def __init__(self, number: str, title: str, chapter: str, page_start: int):
        self.number = number
        self.title = title
        self.chapter = chapter
        self.page_start = page_start
        self.page_end = page_start
        self.body_lines: List[str] = []
        self.bullets: List[Dict[str, Any]] = []  # {idx:int, text:str}
        self.children: List["Node"] = []

    def as_dict(self) -> Dict[str, Any]:
        full_text = self.full_text()
        return {
            "id": mk_id("r", self.number),
            "number": self.number,
            "title": title_from_text(self.title or full_text, f"סעיף {self.number}"),
            "category": self.chapter,
            "text": full_text,
            "bullets": self.bullets,
            "citations": [{"page_start": self.page_start, "page_end": self.page_end}],
            "conditions": extract_conditions(full_text),
            "features": extract_features(full_text),
            "children": [c.as_dict() for c in self.children],
        }

    def full_text(self) -> str:
        return " ".join([l.strip() for l in self.body_lines if l.strip()])


# ---------- Parser ----------
def parse_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = fitz.open(pdf_path.as_posix())
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        lines = [normalize_line(ln) for ln in text.splitlines()]
        pages.append({"page": i+1, "lines": lines})

    nodes_root: List[Node] = []
    flat: List[Node] = []

    chapter_title = "לא ידוע"
    stack: List[Tuple[int, Node]] = []  # (depth, node)

    def flush_page_end(n: Optional[Node], page_num: int):
        if n:
            n.page_end = page_num

    for p in pages:
        page_num = p["page"]
        # Detect chapter on this page (if present)
        for ln in p["lines"][:12]:
            m_ch = CHAPTER_RE.match(ln)
            if m_ch:
                chapter_idx = m_ch.group(1)
                chapter_title = detect_category(m_ch.group(2))
                break  # chapter for the rest of this page (and until next)

        for raw in p["lines"]:
            ln = raw

            # 1) Clause head
            m = CLAUSE_HEAD_RE.match(ln)
            if m:
                num = m.group("num")
                title = (m.group("title") or "").strip()

                # Close last open node (update end page)
                if stack:
                    flush_page_end(stack[-1][1], page_num)

                depth = num.count(".") + 1  # "1.2.3" -> depth 3
                node = Node(number=num, title=title, chapter=chapter_title, page_start=page_num)

                # Re-stack
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                if stack:
                    stack[-1][1].children.append(node)
                else:
                    nodes_root.append(node)

                stack.append((depth, node))
                flat.append(node)
                continue

            # 2) Bullet like ")1(" ...
            m_b = BULLET_RE.match(ln)
            if m_b and stack:
                try:
                    idx = int(m_b.group(1))
                except:
                    idx = None
                text_b = m_b.group(2).strip()
                stack[-1][1].bullets.append({"idx": idx, "text": text_b})
                continue

            # 3) Continuation/body lines
            if stack and not EMPTY_RE.match(ln):
                stack[-1][1].body_lines.append(ln)

        # page end: update last open node
        if stack:
            stack[-1][1].page_end = page_num

    doc.close()

    # Convert to dicts
    tree = [n.as_dict() for n in nodes_root]
    flat_list = [n.as_dict() for n in flat]
    return tree, flat_list


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to 18-07-2022_4.2A.pdf")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON files")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree, flat_list = parse_pdf(pdf_path)

    (out_dir / "rules_tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rules_flat.json").write_text(json.dumps(flat_list, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✔ Parsed clauses: tree={len(tree)} roots, flat={len(flat_list)} items")
    print(f"→ {out_dir/'rules_tree.json'}")
    print(f"→ {out_dir/'rules_flat.json'}")


if __name__ == "__main__":
    main()
