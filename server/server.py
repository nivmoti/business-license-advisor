# server.py
from __future__ import annotations
import os, json, re, math
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from openai import OpenAI

# ============== קונפיג ==============
OPENAI_MODEL_EMB = os.getenv("EMB_MODEL", "text-embedding-3-large")
OPENAI_MODEL_CHAT = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RULES_PATH = Path(os.getenv("RULES_PATH", "data/curated/rules_flat.json"))
TAXO_PATH  = Path(os.getenv("TAXO_PATH",  "data/features_taxonomy.json"))

# ספים/משקלים
SIM_THRESHOLD_ACCEPT = float(os.getenv("SIM_THRESHOLD_ACCEPT", "0.82"))
SIM_THRESHOLD_SUGGEST = float(os.getenv("SIM_THRESHOLD_SUGGEST", "0.65"))
KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.10"))
POS_FEATURE_BOOST = float(os.getenv("POS_FEATURE_BOOST", "0.30"))
NEG_FEATURE_PENALTY = float(os.getenv("NEG_FEATURE_PENALTY", "0.20"))
MAX_RULES_TO_SUMMARIZE = int(os.getenv("MAX_RULES_TO_SUMMARIZE", "60"))

NEGATION_WORDS = ["ללא", "בלי", "אסור", "אין", "לא"]

client = OpenAI()

# ============== מודלים/DTOs ==============
class FeatureTaxoItem(BaseModel):
    id: str
    label_he: str
    synonyms_he: List[str] = Field(default_factory=list)

class ResolveRequest(BaseModel):
    text: str

class ResolvedFeature(BaseModel):
    id: str
    label_he: str
    score: float
    polarity: str  # "positive"|"negative"
    evidence: List[str] = Field(default_factory=list)

class ResolveResponse(BaseModel):
    resolved: List[ResolvedFeature]
    suggested: List[ResolvedFeature]
    unresolved: List[str] = Field(default_factory=list)

class MatchRequest(BaseModel):
    area_m2: Optional[float] = None
    seats: Optional[int] = None
    features_text: str
    language: str = "he"

class Rule(BaseModel):
    id: str
    number: str
    title: str
    category: str
    text: str
    features: List[str] = Field(default_factory=list)
    citations: List[Dict[str, int]] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)

class MatchResponse(BaseModel):
    selected_rule_ids: List[str]
    llm_report: str
    debug: Dict[str, Any]

# ============== עזר: נרמול עברית ==============
HEB_PUNCT_RE = re.compile(r"[^\w\s׳״\"'\-–—/:()]+", re.UNICODE)
SPACES_RE = re.compile(r"\s+")

def normalize_hebrew(s: str) -> str:
    s = s.replace("•", " ").replace("·", " ").replace("\u200f", " ")
    s = s.replace("־", "-")  # מקף עברי
    s = HEB_PUNCT_RE.sub(" ", s)
    s = SPACES_RE.sub(" ", s).strip()
    return s

def has_negation_near(term: str, text: str, window: int = 18) -> bool:
    # חיפוש שלילה קרובה: "ללא<עד ח׳ רווחים>אלכוהול"
    for neg in NEGATION_WORDS:
        pattern = rf"{neg}\s+(?:\S+\s+){{0,{max(1, window//6)}}}?{re.escape(term)}"
        if re.search(pattern, text):
            return True
    return False

# ============== טעינת כללים + טקסונומיה ==============
def load_rules(path: Path) -> List[Rule]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Rule(**r) for r in data]

def build_taxonomy(rules: List[Rule]) -> List[FeatureTaxoItem]:
    if TAXO_PATH.exists():
        return [FeatureTaxoItem(**x) for x in json.loads(TAXO_PATH.read_text(encoding="utf-8"))]
    # גזירת טקסונומיה בסיסית מהחוקים (fallback)
    seen = {}
    for r in rules:
        for f in r.features:
            if f not in seen:
                seen[f] = FeatureTaxoItem(id=f, label_he=f, synonyms_he=[f])
    return list(seen.values())

# ============== אינדקס Embeddings ==============
@dataclass
class SurfaceVec:
    feature_id: str
    surface: str
    vec: np.ndarray

class EmbIndex:
    def __init__(self):
        self.items: List[SurfaceVec] = []
        self._mat: Optional[np.ndarray] = None

    def build(self, taxo: List[FeatureTaxoItem]):
        surfaces = []
        meta: List[tuple[str,str]] = []
        for t in taxo:
            candidates = [t.label_he] + list(t.synonyms_he)
            for s in candidates:
                s_norm = normalize_hebrew(s)
                if not s_norm:
                    continue
                surfaces.append(s_norm)
                meta.append((t.id, s))
        if not surfaces:
            self._mat = np.zeros((0, 1536), dtype=np.float32)
            return

        embs = client.embeddings.create(model=OPENAI_MODEL_EMB, input=surfaces).data
        vecs = [np.array(e.embedding, dtype=np.float32) for e in embs]
        self.items = [SurfaceVec(feature_id=mid, surface=surf, vec=v) for (mid, surf), v in zip(meta, vecs)]
        self._mat = np.vstack([it.vec for it in self.items]) if self.items else np.zeros((0, len(vecs[0])), dtype=np.float32)

    def similarity(self, query_vec: np.ndarray) -> np.ndarray:
        if self._mat is None or self._mat.size == 0:
            return np.zeros((0,), dtype=np.float32)
        # cosine
        a = self._mat
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        return (a @ query_vec) / denom

# ============== פירוק טקסט משתמש למאפיינים ==============
def resolve_features_free_text(taxo: List[FeatureTaxoItem], emb_index: EmbIndex, text: str) -> ResolveResponse:
    base_text = normalize_hebrew(text)
    if not base_text:
        return ResolveResponse(resolved=[], suggested=[], unresolved=[])

    # 1) אמבדינג לטקסט כולו
    q_emb = client.embeddings.create(model=OPENAI_MODEL_EMB, input=[base_text]).data[0].embedding
    q_vec = np.array(q_emb, dtype=np.float32)
    sims = emb_index.similarity(q_vec)

    # 2) איתור מילות מפתח / נרדפים (עם fuzzy קטן)
    def keyword_signal(surface: str, text: str) -> float:
        s = surface.strip()
        if not s:
            return 0.0
        if s in text:
            return 1.0
        # fuzzy boost קטן (נניח מ-92 ומעלה)
        if fuzz.partial_ratio(s, text) >= 92:
            return 0.6
        return 0.0

    # 3) אגרגציה לפי feature
    agg: Dict[str, Dict[str, Any]] = {}
    for idx, it in enumerate(emb_index.items):
        emb_score = float(sims[idx])
        key_score = keyword_signal(normalize_hebrew(it.surface), base_text)
        score = max(emb_score, min(1.0, emb_score + (KEYWORD_BOOST if key_score > 0 else 0.0)))
        neg = has_negation_near(normalize_hebrew(it.surface), base_text)
        if it.feature_id not in agg or score > agg[it.feature_id]["score"]:
            agg[it.feature_id] = {"score": score, "label_he": it.surface, "neg": neg, "evidence": [it.surface]}

    resolved: List[ResolvedFeature] = []
    suggested: List[ResolvedFeature] = []
    for fid, item in agg.items():
        label = next((t.label_he for t in taxo if t.id == fid), fid)
        rf = ResolvedFeature(
            id=fid,
            label_he=label,
            score=round(item["score"], 4),
            polarity="negative" if item["neg"] else "positive",
            evidence=list(set(item["evidence"]))
        )
        if rf.score >= SIM_THRESHOLD_ACCEPT:
            resolved.append(rf)
        elif rf.score >= SIM_THRESHOLD_SUGGEST:
            suggested.append(rf)

    return ResolveResponse(resolved=sorted(resolved, key=lambda x: -x.score),
                           suggested=sorted(suggested, key=lambda x: -x.score),
                           unresolved=[])

# ============== סינון חוקים + דירוג ==============
def rule_passes_bounds(rule: Rule, area_m2: Optional[float], seats: Optional[int]) -> bool:
    cond = rule.conditions or {}
    # מרחב: אם יש מינימום/מקסימום – נבדוק; אם None – לא מגבילים.
    min_area = cond.get("min_area_m2")
    max_area = cond.get("max_area_m2")
    min_seats = cond.get("min_seats")
    max_seats = cond.get("max_seats")

    if area_m2 is not None:
        if min_area is not None and area_m2 < float(min_area): return False
        if max_area is not None and area_m2 > float(max_area): return False

    if seats is not None:
        if min_seats is not None and seats < int(min_seats): return False
        if max_seats is not None and seats > int(max_seats): return False

    return True

def score_rule(rule: Rule, resolved: List[ResolvedFeature]) -> float:
    score = 1.0  # בסיס
    feat_set = set(rule.features or [])
    for rf in resolved:
        if rf.id in feat_set:
            if rf.polarity == "positive":
                score += POS_FEATURE_BOOST
            else:
                score -= NEG_FEATURE_PENALTY
    # משקולת קטנה לאורך טקסט (חוקים ארוכים בדרך כלל מפורטים יותר)
    score += min(0.2, (len(rule.text) / 1000.0) * 0.05)
    return score

# ============== יצירת דוח LLM ==============
def make_report_llm(user: MatchRequest, rules: List[Rule], resolved: List[ResolvedFeature]) -> str:
    # חותכים אם יותר מדי חוקים
    rules = rules[:MAX_RULES_TO_SUMMARIZE]

    # תמצית קלט
    features_human = [f"{r.label_he} ({'ללא' if r.polarity=='negative' else 'כן'}) ~{r.score:.2f}" for r in resolved]
    # מקצרים טקסט לחיסכון בטוקנים
    def rule_brief(r: Rule) -> Dict[str, Any]:
        return {
            "id": r.id,
            "number": r.number,
            "category": r.category,
            "title": r.title[:80],
            "text": (r.text[:600] + "…") if len(r.text) > 650 else r.text,
            "citations": r.citations,
            "features": r.features
        }

    payload = {
        "user_profile": {
            "area_m2": user.area_m2,
            "seats": user.seats,
            "features": features_human
        },
        "rules": [rule_brief(r) for r in rules]
    }

    sys = (
        "את/ה עוזר/ת רגולטורי/ת בעברית פשוטה. קבל/י רשימת סעיפים רלוונטיים לעסק, "
        "והפק/י דוח תמציתי ומועיל לבעל העסק. "
        "הדוח צריך להיות ברור, לפעולה, ומחולק לקטגוריות/עדיפויות. "
        "אל תמציא/י עובדות. ציין/י מספר סעיף וציטוט קצר רלוונטי, ושמור/י על שפה ידידותית."
    )
    user_msg = (
        "פרטי העסק ותאימות ראשונית לחוקים מצורפים כ-JSON. "
        "הפק/י דוח מקוצר: סיכום כללי, דרישות קריטיות (חובה), דרישות מומלצות (רשות), והקלות/פטורים. "
        "אם יש סעיפי 'הקלה' (למשל 'ללא אלכוהול' עד 200 מקומות) – הדגש/י אותם. "
        "בסוף כל תת-סעיף, הוסף/י סימוכין בסגנון: [סעיף X.Y.Z, עמ' N].\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user_msg}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ============== FastAPI App ==============
app = FastAPI(title="Business License Advisor API", version="0.1.0")

# מצב זיכרון
RULES: List[Rule] = []
TAXO: List[FeatureTaxoItem] = []
EMB_INDEX = EmbIndex()

@app.on_event("startup")
def _startup():
    global RULES, TAXO, EMB_INDEX
    RULES = load_rules(RULES_PATH)
    TAXO = build_taxonomy(RULES)
    EMB_INDEX.build(TAXO)
    print(f"[startup] rules={len(RULES)} ; features={len(TAXO)} ; surfaces={len(EMB_INDEX.items)}")

@app.get("/health")
def health():
    return {"ok": True, "rules": len(RULES), "features": len(TAXO)}

@app.post("/features/resolve", response_model=ResolveResponse)
def resolve_endpoint(payload: ResolveRequest):
    return resolve_features_free_text(TAXO, EMB_INDEX, payload.text)

@app.post("/match", response_model=MatchResponse)
def match_endpoint(req: MatchRequest):
    # 1) נרמול מאפיינים
    resolved_resp = resolve_features_free_text(TAXO, EMB_INDEX, req.features_text)
    resolved = resolved_resp.resolved  # אפשר לצרף גם suggested אם תרצה

    # 2) סינון לפי תחום גודל/תפוסה
    filtered = [r for r in RULES if rule_passes_bounds(r, req.area_m2, req.seats)]

    # 3) דירוג לפי בוסט מאפיינים + עוד
    scored = [(r, score_rule(r, resolved)) for r in filtered]
    scored.sort(key=lambda x: -x[1])

    selected_rules = [r for r, s in scored if s > 0.5][:MAX_RULES_TO_SUMMARIZE]

    # 4) דוח LLM
    report = make_report_llm(req, selected_rules, resolved)

    debug = {
        "resolved_features": [rf.model_dump() for rf in resolved],
        "filtered_count": len(filtered),
        "selected_count": len(selected_rules),
        "top5": [{"id": r.id, "score": s} for r, s in scored[:5]]
    }

    return MatchResponse(
        selected_rule_ids=[r.id for r in selected_rules],
        llm_report=report,
        debug=debug
    )
