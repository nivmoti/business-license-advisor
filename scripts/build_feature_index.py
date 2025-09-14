#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a local vocab of feature terms from rules_flat.json + OpenAI embeddings.
Outputs to data/index/: terms.json, embeddings.npz, meta.json
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from sklearn.preprocessing import normalize
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data" / "curated" / "rules_flat.json"
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"

# --- Hebrew keyword harvesting (אפשר להרחיב בהמשך)
HE_PATTERNS = [
    r"\bמשקאות\s+משכרים\b", r"\bאלכוהול\b",
    r"\bגז\b", r"\bבלוני?\s*גז\b",
    r"\bבשר\b", r"\bעוף\b", r"\bדגים\b",
    r"\bמשלוח(ים)?\b", r"\bטייק\s*אווי\b",
    r"\bטיגון\b", r"\bמחבת(ות)?\b", r"\bצ\'?יפס\b",
    r"\bמצלמות\s*(אבטחה)?\b", r"\bcctv\b",
    r"\bאסור\s+לעשן\b", r"\bשילוט\s*עישון\b",
    r"\bמי\s+שת(י|י)ה\b", r"\bביוב\b", r"\bשפכים\b",
    r"\bמזון\s*קר\b", r"\bקירור\b", r"\bמקרר\b",
    r"\bמזון\s*חם\b", r"\bבישול\b", r"\bחימום\b"
]
RXES = [re.compile(p, re.I) for p in HE_PATTERNS]

def normalize_he(s: str) -> str:
    s = re.sub(r"[^\w\s׳״\"'’\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def harvest_terms(rules: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    term2rules = defaultdict(set)

    # 1) מהשדה features המובנה אם קיים
    for r in rules:
        rid = r.get("id") or r.get("number")
        for f in r.get("features") or []:
            if isinstance(f, str) and f.strip():
                term2rules[f.strip()].add(rid)

    # 2) מהטקסט/כותרת ע"י regex
    for r in rules:
        rid = r.get("id") or r.get("number")
        txt = f"{r.get('title','')} {r.get('text','')}"
        n = normalize_he(txt)
        for rx in RXES:
            for m in rx.findall(n):
                term2rules[m.strip()].add(rid)

    # אריזה
    out = {}
    for t, ids in term2rules.items():
        tt = t.strip()
        if not tt:
            continue
        out[tt] = {"term": tt, "rule_ids": sorted(ids), "count": len(ids)}
    return out

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    out = []
    B = 256
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        out.append(np.vstack(vecs))
    X = np.vstack(out)
    return normalize(X)  # נרמול לקוסינוס

def main():
    if not RULES_PATH.exists():
        raise FileNotFoundError(f"Missing {RULES_PATH}")

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    terms_map = harvest_terms(rules)
    if not terms_map:
        print("[!] No terms harvested. Check rules_flat.json.")
        return

    terms = sorted(terms_map.keys())
    print(f"[*] harvested {len(terms)} terms from {len(rules)} rules")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    X = embed_texts(client, terms)
    np.savez(INDEX_DIR / "embeddings.npz", X=X.astype(np.float32))
    (INDEX_DIR / "terms.json").write_text(
        json.dumps([terms_map[t] for t in terms], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(
            {"model": EMBED_MODEL, "dim": int(X.shape[1]), "count": len(terms), "ts": int(time.time())},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    print(f"[✔] wrote index to {INDEX_DIR}")

if __name__ == "__main__":
    main()
