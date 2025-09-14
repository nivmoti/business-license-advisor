#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# server/server.py

from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
from sklearn.preprocessing import normalize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware


# ---------- נתיבים וקונפיג ----------
ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data" / "curated" / "rules_flat.json"
INDEX_DIR = ROOT / "data" / "index"
TERMS_PATH = ROOT / "data" / "index" / "terms.json"  # מיפוי term -> rule_ids (לא חובה, אך מועיל)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")  # אפשר לשנות ב־env

# ---------- מצב גלובלי ----------
RULES: List[Dict[str, Any]] = []
TERMS: List[Dict[str, Any]] = []          # מהאינדקס הסמנטי data/index/terms.json (רשימת אובייקטים עם term, rule_ids)
TERM2IDX: Dict[str, int] = {}             # term -> row in X
X: Optional[np.ndarray] = None            # מטריצת אמבדינג של ה־terms
EMBED_CLIENT: Optional[OpenAI] = None
CHAT_CLIENT: Optional[OpenAI] = None

# ממפה מונחי פיצ'רים (עברית/אנגלית) לקנוני אחיד
FEATURE_SYNONYMS: Dict[str, str] = {
    # English canonical set
    "alcohol": "alcohol",
    "cctv": "cctv",
    "delivery": "delivery",
    "frying": "frying",
    "serves_meat": "serves_meat",
    "uses_gas": "uses_gas",
    "water": "water",
    "sewage": "sewage",
    "smoking_sign": "smoking_sign",
    "food_hot": "food_hot",
    "food_cold": "food_cold",

    # Hebrew → canonical
    "אלכוהול": "alcohol",
    "משקאות משכרים": "alcohol",
    "מצלמות אבטחה": "cctv",
    "מצלמות": "cctv",
    "משלוחים": "delivery",
    "שליחויות": "delivery",
    "טיגון": "frying",
    "צ׳יפס": "frying",
    "צ'יפס": "frying",
    "בשר": "serves_meat",
    "מגישה בשר": "serves_meat",
    "גז": "uses_gas",
    "בלוני גז": "uses_gas",
    "מים": "water",
    "ביוב": "sewage",
    "שפכים": "sewage",
    "שלטי עישון": "smoking_sign",
    "עישון": "smoking_sign",
    "מזון חם": "food_hot",
    "מזון קר": "food_cold",
}

# מילות שלילה בסיסיות (חלון מילים קצר סביב הפיצ׳ר)
NEGATION_TOKENS: Set[str] = {"ללא", "אין", "לא", "בלי", "אסור"}

# ---------- FastAPI ----------
app = FastAPI(title="Business License Advisor API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # לפיתוח. לפרודקשן הגבל דומיינים
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- מודלים ל־Pydantic ----------
class ReportRequest(BaseModel):
    size_m2: float = Field(0, description="Business area in m^2")
    seats: int = Field(0, description="Seating/occupancy")
    features_text: str = Field("", description="Free-text features (Hebrew)")
    top_n_rules: int = Field(60, description="Max number of rules for the LLM")
    debug: bool = Field(False, description="Return debug info in response")

class ReportResponse(BaseModel):
    selected_count: int
    matches: List[Dict[str, Any]] = []
    sample_rules: List[Dict[str, Any]] = []
    report: str
    debug_info: Optional[Dict[str, Any]] = Field(default=None)

# ---------- עזר ----------
ID_TRAILING_UNDER_RX = re.compile(r"_+$")

def normalize_rule_id(rid: str | None) -> str:
    """מסיר קווים תחתונים מסוף המזהה כדי לנרמל התאמות (r_3_6_1_ → r_3_6_1)."""
    if not rid:
        return ""
    return ID_TRAILING_UNDER_RX.sub("", rid.strip())

def _normalize_he(txt: str) -> str:
    """נרמול פשוט לעברית: החלפת גרשיים, הורדת תווים לא־אותיים, רווחים מיותרים, לאוור־קייס."""
    txt = txt.replace("״", '"').replace("”", '"').replace("„", '"').replace("’", "'")
    txt = re.sub(r"[^\w\u0590-\u05FF\s\'\"]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt.lower()

def load_rules(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rules file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("rules_flat.json must be a JSON array")
    # נרמול מזהים
    for r in data:
        if "id" in r and r["id"]:
            r["id"] = normalize_rule_id(r["id"])
    return data

def load_terms_map(path: Path) -> Dict[str, Set[str]]:
    """
    קורא את data/curated/terms.json (אם קיים): [{term, rule_ids, count}, ...]
    ומחזיר מיפוי term -> set(rule_id) כשהמזהים מנורמלים.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Set[str]] = {}
    for item in raw:
        term = item.get("term", "")
        rids = {normalize_rule_id(x) for x in item.get("rule_ids", [])}
        if term:
            out[term] = rids
    return out

def get_condition_bounds(r: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    c = r.get("conditions") or {}
    return (
        c.get("min_area_m2"), c.get("max_area_m2"),
        c.get("min_seats"), c.get("max_seats")
    )

def rule_matches_numeric(r: Dict[str, Any], size_m2: float, seats: int) -> bool:
    min_a, max_a, min_s, max_s = get_condition_bounds(r)
    ok_area = True
    ok_seats = True
    if min_a is not None and size_m2 < float(min_a): ok_area = False
    if max_a is not None and size_m2 > float(max_a): ok_area = False
    if min_s is not None and seats   < int(min_s):   ok_seats = False
    if max_s is not None and seats   > int(max_s):   ok_seats = False
    return ok_area and ok_seats

def filter_rules_by_numeric(rules: List[Dict[str, Any]], size_m2: float, seats: int) -> List[Dict[str, Any]]:
    return [r for r in rules if rule_matches_numeric(r, size_m2, seats)]

def detect_negations(text: str) -> Set[str]:
    """
    מאתר מאפיינים שמופיעים עם מילת שלילה בחלון קצר לפני המונח.
    לדוגמה: 'אין אלכוהול', 'בלי גז', 'ללא טיגון'.
    מחזיר סט של שמות קנוניים שנשללו.
    """
    t = _normalize_he(text)
    tokens = t.split()
    negated: Set[str] = set()
    window_before = 3

    # בנה רשימה של ווריאנטים מנורמלים → קנוני (שימושי לטוקניזציה)
    syn_norm = { _normalize_he(k): v for k, v in FEATURE_SYNONYMS.items() }

    for i, tok in enumerate(tokens):
        # אם הטוקן הוא אחד הווריאנטים – בדוק חלון לפניו
        if tok in syn_norm:
            canon = syn_norm[tok]
            start = max(0, i - window_before)
            if any(n in tokens[start:i] for n in NEGATION_TOKENS):
                negated.add(canon)
    return negated

def resolve_features(text: str) -> Dict[str, Any]:
    """
    מזהה מאפיינים לפי מילון סינונים (עברית/אנגלית) ומחזיר:
      resolved: [{input, canonical, via}]
      negative: [canonical,...] — מאפיינים שנשללו מהטקסט
    """
    t = _normalize_he(text)
    resolved = []
    seen = set()

    # התאמה לפי מילון
    for variant, canon in FEATURE_SYNONYMS.items():
        v = _normalize_he(variant)
        if v and v in t and (variant, canon) not in seen:
            resolved.append({"input": variant, "canonical": canon, "via": "synonym"})
            seen.add((variant, canon))

    # התאמה "ישירה" על פי טוקנים – לא הכרחי אבל לפעמים מועיל
    for raw in re.split(r"[,\s]+", t):
        raw = raw.strip()
        if not raw:
            continue
        if raw in FEATURE_SYNONYMS:
            canon = FEATURE_SYNONYMS[raw]
            if (raw, canon) not in seen:
                resolved.append({"input": raw, "canonical": canon, "via": "exact"})
                seen.add((raw, canon))

    negative = list(detect_negations(text))
    return {"resolved": resolved, "negative": negative}

# ---------- אמבדינג/התאמה סמנטית ----------
def embed_query(client: OpenAI, text: str) -> Optional[np.ndarray]:
    text = (text or "").strip()
    if not text:
        return None
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return normalize(v)

def semantic_match_features(text: str, top_k: int = 10, min_sim: float = 0.73):
    """
    התאמה סמנטית של טקסט חופשי ל־terms (אם נבנה אינדקס ב־data/index).
    מחזיר:
      {"matches": [{"term","score","rule_ids"}, ...], "rule_ids": set([...])}
    """
    if X is None or EMBED_CLIENT is None or not TERMS:
        return {"matches": [], "rule_ids": set()}
    v = embed_query(EMBED_CLIENT, text)
    if v is None:
        return {"matches": [], "rule_ids": set()}
    scores = (X @ v.T).ravel()
    idxs = np.argsort(-scores)[:top_k]
    out = []
    acc: Set[str] = set()
    for i in idxs:
        s = float(scores[int(i)])
        if s < min_sim:
            continue
        t = TERMS[int(i)]
        # נרמול מזהים בצד ה־terms למקרה שטרם נוקו
        rule_ids = [normalize_rule_id(x) for x in t.get("rule_ids", [])]
        out.append({"term": t["term"], "score": s, "rule_ids": rule_ids})
        acc.update(rule_ids)
    return {"matches": out, "rule_ids": acc}

# ---------- בחירת כללים לדוח ----------
def select_rules_for_report(payload: ReportRequest) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    1) סינון נומרי לפי שטח/תפוסה
    2) מיצוי מאפיינים: resolved + negative
    3) חיבור חוקים:
       - חוקים שעוברים נומרית
       - עדיפות לחוקים שהגיעו ממאפיינים (terms.json + התאמה סמנטית)
       - מדלגים על מאפיינים שנשללו (negative)
    4) דירוג לפי: הופעה במאפיינים, אורך טקסט, ומעט ענישה לכללים כלליים מאוד
    """
    # בסיס: תנאים נומריים
    base = filter_rules_by_numeric(RULES, size_m2=payload.size_m2, seats=payload.seats)

    # שלב מאפיינים
    feat = resolve_features(payload.features_text)
    resolved_canons = [x["canonical"] for x in feat["resolved"]]
    negative_canons = set(feat["negative"])

    # חוקים מ-terms.json הקלאסי (אם קיים)
    ids_from_terms: Set[str] = set()
    for canon in resolved_canons:
        if canon in negative_canons:
            continue  # אם נשלל — לא לאסוף חוקים ממנו
        if TERMS_MAP and canon in TERMS_MAP:
            ids_from_terms |= TERMS_MAP[canon]

    # התאמה סמנטית (אם יש אינדקס)
    sem = semantic_match_features(payload.features_text)
    sem_ids = set(sem["rule_ids"])

    # מועמדים: חיתוך עם הבסיס, עם הטיה לטובת מי שמופיע במאפיינים
    def pick_candidates() -> List[Dict[str, Any]]:
        prefer_ids = ids_from_terms | sem_ids
        if prefer_ids:
            on_sem = [
                r for r in base
                if normalize_rule_id(r.get("id") or r.get("number")) in prefer_ids
            ]
            return on_sem if on_sem else base
        return base

    candidates = pick_candidates()

    # דירוג
    def score(r: Dict[str, Any]) -> Tuple[float, int, int, float]:
        rid = normalize_rule_id(r.get("id") or r.get("number", ""))
        s_from_feat = 1.0 if (rid in ids_from_terms or rid in sem_ids) else 0.0
        feats_len = len(r.get("features") or [])
        text_len = len((r.get("text") or ""))
        generic_penalty = -0.2 if r.get("number", "").count(".") <= 1 else 0.0
        return (s_from_feat, feats_len, text_len, generic_penalty)

    selected = sorted(candidates, key=score, reverse=True)[:payload.top_n_rules]

    debug_info = {
        "features_resolved": feat["resolved"],
        "features_negative": list(negative_canons),
        "rule_ids_from_terms_count": len(ids_from_terms),
        "rule_ids_from_terms_preview": list(sorted(ids_from_terms))[:30],
        "semantic_matches_count": len(sem.get("matches", [])),
        "semantic_matches": sem.get("matches", [])[:10],
        "selected_ids_preview": [
            normalize_rule_id(r.get("id") or r.get("number")) for r in selected[:30]
        ],
    }

    return selected, debug_info

# ---------- קריאה ל־LLM ----------
def llm_report(client: OpenAI, model: str, payload: ReportRequest, rules: List[Dict[str, Any]]) -> str:
    # הקשר קומפקטי — מספר/כותרת/קטגוריה/טקסט קצר
    def short(r):
        return {
            "id": normalize_rule_id(r.get("id") or r.get("number")),
            "number": r.get("number"),
            "title": r.get("title") or "",
            "category": r.get("category") or "",
            "text": (r.get("text") or "")[:600]
        }

    rules_ctx = [short(r) for r in rules]

    system = (
        "You are a helpful assistant that writes plain Hebrew compliance summaries for restaurant licensing.\n"
        "Use only the provided rules context. Be concise, structured, and clear.\n"
        "Avoid legalese; explain in business language. Group by category and add actionable next steps."
    )
    user = (
        "נתוני העסק:\n"
        f"- גודל: {payload.size_m2} מ\"ר\n"
        f"- מקומות ישיבה: {payload.seats}\n"
        f"- מאפיינים חופשיים: {payload.features_text}\n\n"
        "להלן אוסף כללים רלוונטיים (מספר/כותרת/קטגוריה/טקסט):\n"
        f"{json.dumps(rules_ctx, ensure_ascii=False, indent=2)}\n\n"
        "משימה: הפק דו\"ח מותאם אישית בעברית ברורה:\n"
        "1) סיכום קצר של מצב העסק ביחס לכללים שנמצאו\n"
        "2) דרישות מחייבות — מחולקות לפי קטגוריות (בריאות/כבאות/משטרה/אחר)\n"
        "3) המלצות פעולה פרקטיות לפי סדר עדיפויות (מיידי/בינוני/נחמד שיהיה)\n"
        "4) ציין הפניות (מספר סעיף) היכן שאפשר.\n"
        "הימנע ממידע שלא נמצא בהקשר שניתן."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- מסלולים ----------
@app.get("/health")
def health():
    ok_index = (INDEX_DIR / "terms.json").exists() and (INDEX_DIR / "embeddings.npz").exists()
    return {"ok": True, "rules_loaded": len(RULES), "feature_index": ok_index}

@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    if not RULES:
        raise HTTPException(status_code=500, detail="rules_flat.json not loaded")
    if CHAT_CLIENT is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured (OPENAI_API_KEY)")

    selected, debug_info = select_rules_for_report(req)
    report_text = llm_report(CHAT_CLIENT, CHAT_MODEL, req, selected)

    sample = [{"id": normalize_rule_id(r.get("id") or r.get("number")),
               "number": r.get("number"),
               "title": r.get("title")} for r in selected[:10]]

    return ReportResponse(
        selected_count=len(selected),
        matches=debug_info["semantic_matches"] if req.debug else [],
        sample_rules=sample,
        report=report_text,
        debug_info=debug_info if req.debug else None
    )

# ---------- אתחול ----------
TERMS_MAP: Dict[str, Set[str]] = {}  # term (כמו "alcohol") -> set(rule_id)

@app.on_event("startup")
def on_startup():
    global RULES, TERMS, TERM2IDX, X, EMBED_CLIENT, CHAT_CLIENT, TERMS_MAP

    # rules
    RULES = load_rules(RULES_PATH)
    print(f"[startup] loaded {len(RULES)} rules from {RULES_PATH}")

    # terms map (לא חובה אך מוסיף recall גבוה במאפיינים)
    TERMS_MAP = load_terms_map(TERMS_PATH)
    print(f"[startup] loaded {len(TERMS_MAP)} canonical terms from {TERMS_PATH}")

    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[startup] WARNING: OPENAI_API_KEY not set — LLM/embeddings disabled")
    else:
        EMBED_CLIENT = OpenAI(api_key=api_key)
        CHAT_CLIENT  = EMBED_CLIENT

    # feature index (סמנטי)
    tpath = INDEX_DIR / "terms.json"
    epath = INDEX_DIR / "embeddings.npz"
    if tpath.exists() and epath.exists():
        TERMS = json.loads(tpath.read_text(encoding="utf-8"))
        # נרמול מזהים גם בצד הזה ליתר ביטחון
        for t in TERMS:
            t["rule_ids"] = [normalize_rule_id(x) for x in t.get("rule_ids", [])]

        data = np.load(epath)
        X = data["X"].astype(np.float32)
        X = normalize(X)
        TERM2IDX = {t["term"]: i for i, t in enumerate(TERMS)}
        print(f"[startup] feature index loaded: {len(TERMS)} terms, X={X.shape}")
    else:
        print("[startup] feature index not found — run scripts/build_feature_index.py first")
