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
    allow_origins=["*"],
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
    distance: float = Field(..., description="Euclidean pairwise distance between normalized embeddings")
    threshold: float = Field(0.55, description="Decision threshold used")
    anonymized_resume: str = Field("", description="De-identified resume text")
    anonymized_jd: str = Field("", description="De-identified JD text")

    # Optional / backward-compatible fields with defaults
    match_score: float | None = None
    raw_distance: float | None = None
    raw_cosine: float | None = None
    raw_cosine_similarity: float | None = None
    shared_keywords_count: int | None = 0
    lexical_overlap: float | None = 0.0
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

        if not clean_resume.strip() or not clean_jd.strip():
            raise HTTPException(
                status_code=422,
                detail="After de-identification, one or both inputs are empty.",
            )

        # --- 2. Tokenise for PyTorch Bi-Encoder ----------------------------
        vocab = _metadata["vocab"]
        max_len = _metadata["max_len"]

        tensor_r = text_to_sequence(clean_resume, vocab, max_len)
        tensor_j = text_to_sequence(clean_jd, vocab, max_len)

        # --- 3. Inference (CPU, no grad) -----------------------------------
        with torch.no_grad():
            u, v = _model(tensor_r, tensor_j)
            dist = torch.norm(u - v, p=2, dim=1).item()
            sim = torch.sum(u * v, dim=1).item()

        # --- 4. Shared Keywords & Distance-Based Scoring ------------------
        common_words = set(clean_resume.lower().split()).intersection(set(clean_jd.lower().split()))
        meaningful_shared = [w for w in common_words if len(w) > 3]

        THRESHOLD = 0.55
        is_qualified = bool(dist <= THRESHOLD)

        if is_qualified:
            score = 98.0 - (dist / THRESHOLD) * 28.0
        else:
            excess = min(dist - THRESHOLD, 0.70)
            score = max(10.0, 50.0 - (excess / 0.70) * 40.0)

        score = round(float(score), 2)

        print(f"\n[INFERENCE CHECK]")
        print(f"Euclidean Distance: {dist:.4f}")
        print(f"Colab Threshold:    {THRESHOLD}")
        print(f"Final Score:        {score}%")
        print(f"Qualified:          {is_qualified}\n")

        tokens_resume = [t.lower().strip() for t in clean_resume.split() if t.strip()]
        tokens_jd = [t.lower().strip() for t in clean_jd.split() if t.strip()]
        jaccard = compute_token_overlap(tokens_resume, tokens_jd)
        jaccard_rounded = round(jaccard, 4)

        verdict = "MATCH" if is_qualified else "NO MATCH"

        logger.info(
            "dist=%.4f  sim=%.4f  score=%.2f%%  is_qualified=%s",
            dist,
            sim,
            score,
            is_qualified,
        )

        print(f"[API RESPONSE] Score: {score}% | is_match: {is_qualified} | dist: {dist:.4f}")

        return {
            "compatibility_score": score,
            "score": score,
            "is_match": is_qualified,
            "distance": round(dist, 4),
            "threshold": 0.55,
            "anonymized_resume": clean_resume,
            "anonymized_jd": clean_jd,
            "match_score": score,
            "raw_distance": round(dist, 4),
            "raw_cosine": round(sim, 4),
            "raw_cosine_similarity": round(sim, 4),
            "shared_keywords_count": len(meaningful_shared),
            "lexical_overlap": jaccard_rounded,
            "effective_similarity": round(sim, 4),
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
