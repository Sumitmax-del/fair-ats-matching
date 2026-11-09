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


def clean_text_for_model(text: str) -> str:
    """Normalize text by stripping Unicode bullets, smart quotes, em-dashes, and non-ASCII punctuation."""
    # Replace special bullets, smart quotes, em-dashes
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\u2014\u2013\u201c\u201d\u2018\u2019]', ' ', text)
    # Keep letters and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,;]', ' ', text)
    return text.lower().strip()


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
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "*",
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
    jd_text: str | None = Field(None, description="Alias for job_description")


class MatchResponse(BaseModel):
    compatibility_score: float = Field(..., description="Calibrated compatibility score percentage (0–100%)")
    score: float | None = Field(None, description="Alias for compatibility_score")
    is_match: bool = Field(..., description="Whether candidate qualifies for the role")
    cos_sim: float | None = Field(None, description="Cosine similarity between normalized embeddings")
    distance: float = Field(..., description="Euclidean pairwise distance between normalized embeddings")
    threshold: float = Field(0.70, description="Decision threshold used")
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
        resume_text = payload.resume_text or payload.resume or ""
        jd_text = payload.jd_text or payload.job_description or payload.jd or ""
        print(f"Extracted Resume chars: {len(resume_text)} | JD chars: {len(jd_text)}")

        if not resume_text.strip() or not jd_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Please provide both Resume and Job Description.",
            )

        # 1. Clean and lowercase inputs
        clean_r = re.sub(r'[^a-zA-Z0-9\s]', ' ', resume_text).lower()
        clean_j = re.sub(r'[^a-zA-Z0-9\s]', ' ', jd_text).lower()

        # 2. Calculate Vocabulary Overlap to detect domain mismatch
        words_r = set(clean_r.split())
        words_j = set(clean_j.split())
        common_words = words_r & words_j

        COMMON_FILLERS = {'and', 'the', 'to', 'of', 'in', 'for', 'with', 'a', 'an', 'is', 'as', 'at', 'by', 'on', 'from', 'or', 'team', 'work', 'years', 'experience', 'skills'}
        meaningful_overlap = common_words - COMMON_FILLERS
        overlap_ratio = len(meaningful_overlap) / max(len(words_j - COMMON_FILLERS), 1)

        # 3. De-identification & Preprocessing for Model
        res_result = _sanitizer.sanitize(resume_text)
        jd_result = _sanitizer.sanitize(jd_text)

        vocab = _metadata["vocab"]
        max_len = _metadata.get("max_len", 128)

        model_resume = res_result.get("model_text") or clean_r
        model_jd = jd_result.get("model_text") or clean_j

        tensor_r = text_to_sequence(model_resume, vocab, max_len)
        tensor_j = text_to_sequence(model_jd, vocab, max_len)

        # 4. Model Inference & Hard Guardrail
        _model.eval()
        with torch.no_grad():
            u, v = _model(tensor_r, tensor_j)
            u_norm = torch.nn.functional.normalize(u, p=2, dim=1)
            v_norm = torch.nn.functional.normalize(v, p=2, dim=1)
            cos_sim = float(torch.sum(u_norm * v_norm, dim=1).item())

        # GUARD: If two documents have less than 8% meaningful technical vocabulary in common, 
        # force unqualified regardless of embedding collapse:
        if overlap_ratio < 0.08:
            is_match = False
            score = round(max(14.0, overlap_ratio * 200.0), 1)  # Yields ~15% - 25%
        else:
            # Standard semantic match logic:
            is_match = bool(cos_sim >= 0.70)
            if is_match:
                score = round(68.0 + ((cos_sim - 0.70) / 0.30) * 28.0, 1)
            else:
                score = round(max(20.0, (cos_sim / 0.70) * 52.0), 1)

        print(f"[SUBMISSION AUDIT] Meaningful Overlap: {len(meaningful_overlap)} words | Cosine: {cos_sim:.4f} | Final Score: {score}% | Match: {is_match}")

        verdict = "MATCH" if is_match else "NO MATCH"
        dist = float(torch.norm(u_norm - v_norm, p=2, dim=1).item())

        return {
            "compatibility_score": score,
            "score": score,
            "is_match": is_match,
            "cos_sim": round(cos_sim, 4),
            "threshold": 0.70,
            "anonymized_resume": res_result.get("clean_text", clean_r),
            "anonymized_jd": jd_result.get("clean_text", clean_j),
            "distance": round(dist, 4),
            "redacted_resume_proxies": res_result.get("redacted_proxies", []),
            "redacted_jd_proxies": jd_result.get("redacted_proxies", []),
            "match_score": score,
            "raw_distance": round(dist, 4),
            "raw_cosine": round(cos_sim, 4),
            "raw_cosine_similarity": round(cos_sim, 4),
            "shared_keywords_count": len(meaningful_overlap),
            "lexical_overlap": round(overlap_ratio, 4),
            "domain_overlap": round(overlap_ratio, 4),
            "effective_similarity": round(cos_sim, 4),
            "verdict": verdict,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /api/match: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during matching: {str(exc)}",
        )
