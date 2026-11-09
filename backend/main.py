"""FairMatch FastAPI Backend.

Serves the Siamese Bi-Encoder for inclusive resume–JD matching
with demographic de-identification (UN SDG 10) and a lexical domain gate
to eliminate embedding-collapse false positives.

Run from project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
Or from within the backend directory:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend directory is in sys.path so modules resolve whether running from root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.preprocessor import TextSanitizer
from models.biencoder import (
    SiameseBiEncoder,
    load_model_checkpoint,
    text_to_sequence,
)

# ---------------------------------------------------------------------------
# Logging (never log raw PII)
# ---------------------------------------------------------------------------
logger = logging.getLogger("fairmatch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")

# ---------------------------------------------------------------------------
# Global state populated at startup
# ---------------------------------------------------------------------------
_model: SiameseBiEncoder | None = None
_metadata: Dict[str, Any] = {}
_sanitizer: TextSanitizer | None = None


# ---------------------------------------------------------------------------
# Lexical domain gate helper
# ---------------------------------------------------------------------------
def compute_token_overlap(tokens_a: list, tokens_b: list) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and metadata once at startup (CPU only)."""
    global _model, _metadata, _sanitizer

    logger.info("Loading TextSanitizer (spaCy en_core_web_sm) ...")
    _sanitizer = TextSanitizer()

    logger.info("Loading model checkpoint (CPU) ...")
    try:
        _model, _metadata = load_model_checkpoint()
        logger.info(
            "Model loaded  –  vocab=%d  max_len=%d  threshold=%.2f",
            len(_metadata["vocab"]),
            _metadata["max_len"],
            _metadata["threshold"],
        )
    except FileNotFoundError:
        logger.warning(
            "Model weights not found in backend/weights/.  "
            "Place fair_ats_biencoder.pt and ats_metadata.pkl there.  "
            "The /api/match endpoint will return an error until the files are present."
        )

    yield  # app runs

    # Cleanup (nothing to release for CPU inference)
    logger.info("Shutting down FairMatch backend.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FairMatch API",
    description="Inclusive Resume ↔ JD matching with demographic de-identification (SDG 10)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    resume_text: str | None = Field(None, description="Raw resume text (PII will be scrubbed)")
    job_description: str | None = Field(None, description="Raw job description text")
    resume: str | None = Field(None, description="Alias for resume_text")
    jd: str | None = Field(None, description="Alias for job_description")


class MatchResponse(BaseModel):
    compatibility_score: float = Field(..., description="Calibrated compatibility score percentage (0–100%)")
    score: float | None = Field(None, description="Alias for compatibility_score")
    is_match: bool = Field(..., description="Whether candidate qualifies for the role")
    cos_sim: float | None = Field(None, description="Cosine similarity between normalized embeddings")
    distance: float = Field(..., description="Euclidean pairwise distance between normalized embeddings")
    threshold: float = Field(0.72, description="Decision threshold used")
    anonymized_resume: str = Field("", description="De-identified resume text")
    anonymized_jd: str = Field("", description="De-identified JD text")

    # Optional / backward-compatible fields with defaults
    match_score: float | None = None
    raw_distance: float | None = None
    raw_cosine: float | None = None
    raw_cosine_similarity: float | None = None
    shared_keywords_count: int | None = 0
    lexical_overlap: float | None = 0.0
    domain_overlap: float | None = 0.0
    effective_similarity: float | None = None
    verdict: str | None = None
    redacted_resume_proxies: List[str] = Field(default_factory=list)
    redacted_jd_proxies: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "FairMatch API",
        "status": "running",
        "model_loaded": _model is not None,
    }


@app.post("/api/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    """Compute a bias-mitigated match score between a resume and a JD.

    Pipeline:
    1. Sanitize both inputs (strip emails, phones, names, locations, institutes).
    2. Convert sanitized text → padded token-id sequences via the training vocab.
    3. Forward through the shared Siamese Bi-Encoder tower (CPU) to extract raw cosine.
    4. Apply a lexical domain gate (Jaccard token overlap) to penalize vocabulary divergence.
    5. Determine match qualification strictly:
       is_qualified = bool(effective_sim >= threshold and jaccard >= 0.08)
    6. Calibrate final compatibility_score (0% to 100%).
    """
    if _model is None or not _metadata:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model weights not loaded.  "
                "Place fair_ats_biencoder.pt and ats_metadata.pkl in backend/weights/ "
                "and restart the server."
            ),
        )

    try:
        payload = req
        print(f"\n>>> RECEIVED REQUEST AT /api/match <<<")
        resume_val = payload.resume if hasattr(payload, "resume") and payload.resume else (payload.resume_text or "")
        jd_val = payload.jd if hasattr(payload, "jd") and payload.jd else (payload.job_description or "")
        print(f"Resume chars: {len(payload.resume) if hasattr(payload, 'resume') and payload.resume else len(resume_val)}")

        # --- 1. De-identify -----------------------------------------------
        res_result = _sanitizer.sanitize(resume_val)
        jd_result = _sanitizer.sanitize(jd_val)

        clean_resume = res_result["clean_text"]
        clean_jd = jd_result["clean_text"]
        model_resume = res_result.get("model_text") or clean_resume
        model_jd = jd_result.get("model_text") or clean_jd

        if not clean_resume.strip() or not clean_jd.strip():
            raise HTTPException(
                status_code=422,
                detail="After de-identification, one or both inputs are empty.",
            )

        # --- 2. Tokenise for PyTorch Bi-Encoder ----------------------------
        vocab = _metadata["vocab"]
        max_len = _metadata["max_len"]

        tensor_r = text_to_sequence(model_resume, vocab, max_len)
        tensor_j = text_to_sequence(model_jd, vocab, max_len)

        print(f"[TOKEN AUDIT] Resume non-zero: {(tensor_r != 0).sum().item()} | JD non-zero: {(tensor_j != 0).sum().item()}")

        # --- 3. Inference (CPU, no grad) -----------------------------------
        _model.eval()
        with torch.no_grad():
            u, v = _model(tensor_r, tensor_j)
            u_norm = torch.nn.functional.normalize(u, p=2, dim=1)
            v_norm = torch.nn.functional.normalize(v, p=2, dim=1)

            cos_sim = float(torch.sum(u_norm * v_norm, dim=1).item())
            dist = float(torch.norm(u_norm - v_norm, p=2, dim=1).item())

        # --- 4. Domain Lexical Contrast Guard (Hybrid Re-ranking) ---------
        # Extract words >= 4 letters from JD excluding basic filler stopwords
        STOPWORDS = {
            "with", "from", "that", "this", "will", "have", "been", "must", "team", 
            "work", "years", "experience", "role", "looking", "candidate", "responsibilities", 
            "requirements", "skills", "ability", "strong", "preferred", "build", "using"
        }

        jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', clean_jd.lower())) - STOPWORDS
        resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', clean_resume.lower()))

        # Calculate lexical domain overlap ratio
        domain_overlap = len(jd_words & resume_words) / max(len(jd_words), 1)

        # Calibrate qualification threshold to 0.72 (empirical matched vs mismatched boundary)
        MATCH_THRESHOLD = 0.72
        STRONG_MATCH_THRESHOLD = 0.78
        MIN_DOMAIN_OVERLAP = 0.10

        # Hybrid Re-ranking:
        # Cross-domain mismatches lacking domain skills (< 0.10) are penalized to prevent false matches.
        # Qualified domain matches (>= 0.10) have semantic similarity reinforced by skill keyword coverage.
        if domain_overlap < MIN_DOMAIN_OVERLAP:
            penalty = max(0.2, domain_overlap / MIN_DOMAIN_OVERLAP)
            effective_sim = cos_sim * penalty
            is_qualified = False
        else:
            skill_boost = min(0.06, (domain_overlap - MIN_DOMAIN_OVERLAP) * 0.20)
            effective_sim = cos_sim + skill_boost
            is_qualified = bool(effective_sim >= MATCH_THRESHOLD)

        # Continuous dual-band hybrid calibration curve:
        if is_qualified:
            if effective_sim >= STRONG_MATCH_THRESHOLD:
                # Strong match: [0.78, 1.0] -> [85%, 98%]
                norm = min(1.0, (effective_sim - STRONG_MATCH_THRESHOLD) / (1.0 - STRONG_MATCH_THRESHOLD))
                score = round(85.0 + norm * 13.0, 2)
            else:
                # Qualified match: [0.72, 0.78] -> [68%, 84%]
                norm = (effective_sim - MATCH_THRESHOLD) / (STRONG_MATCH_THRESHOLD - MATCH_THRESHOLD)
                score = round(68.0 + norm * 16.0, 2)
            score = round(float(min(98.0, max(68.0, score))), 2)
        else:
            # Disqualified candidates map to [10%, 54%]
            norm = max(0.0, effective_sim) / MATCH_THRESHOLD
            score = round(float(min(54.0, max(10.0, norm * 54.0))), 2)

        print(
            f"\n[SERVER MATCH AUDIT] Cos Sim: {cos_sim:.4f} | Euc Dist: {dist:.4f} | "
            f"Overlap: {domain_overlap:.4f} ({len(jd_words & resume_words)}/{len(jd_words)}) | "
            f"Eff Sim: {effective_sim:.4f} | Score: {score}% | Qualified: {is_qualified}"
        )

        shared_skills = list(jd_words & resume_words)
        verdict = "MATCH" if is_qualified else "NO MATCH"

        logger.info(
            "cos_sim=%.4f  euc_dist=%.4f  overlap=%.4f  score=%.2f%%  is_qualified=%s",
            cos_sim,
            dist,
            domain_overlap,
            score,
            is_qualified,
        )

        return {
            "compatibility_score": score,
            "score": score,
            "is_match": is_qualified,
            "cos_sim": round(cos_sim, 4),
            "distance": round(dist, 4),
            "threshold": MATCH_THRESHOLD,
            "anonymized_resume": clean_resume,
            "anonymized_jd": clean_jd,
            "match_score": score,
            "raw_distance": round(dist, 4),
            "raw_cosine": round(cos_sim, 4),
            "raw_cosine_similarity": round(cos_sim, 4),
            "shared_keywords_count": len(shared_skills),
            "lexical_overlap": round(domain_overlap, 4),
            "domain_overlap": round(domain_overlap, 4),
            "effective_similarity": round(effective_sim, 4),
            "verdict": verdict,
            "redacted_resume_proxies": res_result.get("redacted_proxies", []),
            "redacted_jd_proxies": jd_result.get("redacted_proxies", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /api/match: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during matching: {str(exc)}",
        )
