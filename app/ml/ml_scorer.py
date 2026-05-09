"""
app/ml/ml_scorer.py

Classical ML fast pass — runs BEFORE the LLM to surface load-bearing signals.

Two models:
  1. TF-IDF cosine similarity  — fast, deterministic ATS keyword scorer
  2. XGBoost hire-probability  — trained on resume/JD pairs, gives P(hire)

Per mentor: "most tokens don't matter, a few are load-bearing.
Knowing which is the whole game." — classical ML does this cheaply.

If trained models are not found on disk, the scorer falls back to
TF-IDF-only mode with a clear log warning.
"""
from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ATS Keyword Scoring (TF-IDF cosine similarity)
# ---------------------------------------------------------------------------

def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    TF-IDF cosine similarity between resume and JD.

    Returns:
        {
            ats_score: float,          # 0.0–1.0
            matched_keywords: list,
            missing_keywords: list,
        }
    """
    # Extract meaningful keywords from JD (nouns, tech terms, proper nouns)
    jd_keywords = _extract_keywords(jd_text)
    resume_lower = resume_text.lower()

    matched = [kw for kw in jd_keywords if kw.lower() in resume_lower]
    missing = [kw for kw in jd_keywords if kw.lower() not in resume_lower]

    # TF-IDF vectoriser fit on both documents
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception as e:
        logger.warning("TF-IDF scoring failed: %s — defaulting to keyword ratio", e)
        score = len(matched) / max(len(jd_keywords), 1)

    logger.info(
        "ATS score=%.3f | matched=%d/%d keywords",
        score, len(matched), len(jd_keywords),
    )
    return {
        "ats_score": round(score, 4),
        "matched_keywords": matched[:30],   # top 30 for display
        "missing_keywords": missing[:30],
    }


def _extract_keywords(text: str) -> list[str]:
    """
    Heuristic keyword extractor — pulls tech terms, capitalized phrases,
    and common JD terms. Avoids NLTK dependency for portability.
    """
    # Patterns: CamelCase, ALL_CAPS, hyphenated terms, version strings
    patterns = [
        r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,}){0,2}\b",   # Title Case phrases
        r"\b[A-Z]{2,}\b",                                           # ACRONYMS
        r"\b[a-z]+(?:\.[a-z]+){1,3}\b",                            # dotted.names
        r"\b[A-Za-z]+[-][A-Za-z]+\b",                               # hyphen-terms
        r"\b(?:Python|Java|React|SQL|AWS|GCP|Azure|Docker|Kubernetes|"
        r"LangChain|LangGraph|FastAPI|PyTorch|TensorFlow|XGBoost|"
        r"Streamlit|Bedrock|SageMaker|FAISS|Langfuse|DVC|"
        r"Spring|Kafka|Redis|PostgreSQL|MongoDB|GraphQL|REST|SOAP|"
        r"CI/CD|MLOps|NLP|LLM|RAG|Transformer|BERT)\b",
    ]
    keywords: set[str] = set()
    for pat in patterns:
        keywords.update(re.findall(pat, text))

    # Filter stop words and very short tokens
    stopwords = {"The", "This", "That", "With", "For", "And", "Or", "But",
                 "In", "On", "At", "To", "A", "An", "Is", "Are", "We", "You"}
    return [kw for kw in keywords if kw not in stopwords and len(kw) > 2]


# ---------------------------------------------------------------------------
# XGBoost Hire-Probability Scoring
# ---------------------------------------------------------------------------

def compute_hire_probability(resume_text: str, jd_text: str) -> float:
    """
    XGBoost-based hire probability estimate.

    Falls back to a heuristic score if model files are not found.
    In production, train the model using scripts/train_xgb.py.

    Returns: float in [0.0, 1.0]
    """
    settings = get_settings()

    if not settings.ml_enabled:
        return _heuristic_hire_prob(resume_text, jd_text)

    model_path = Path(settings.xgb_model_path)
    features_path = Path(settings.xgb_features_path)
    tfidf_path = Path(settings.tfidf_model_path)

    if not model_path.exists():
        logger.warning(
            "XGBoost model not found at %s — using heuristic fallback. "
            "Run scripts/train_xgb.py to train.", model_path
        )
        return _heuristic_hire_prob(resume_text, jd_text)

    try:
        import joblib
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(model_path))
        vectorizer = joblib.load(tfidf_path)

        # Feature vector: TF-IDF of resume concat'd with JD
        combined = resume_text + " [SEP] " + jd_text
        features = vectorizer.transform([combined])
        prob = float(model.predict_proba(features)[0][1])
        logger.info("XGBoost hire probability: %.3f", prob)
        return round(prob, 4)

    except Exception as e:
        logger.error("XGBoost prediction failed: %s — using heuristic", e)
        return _heuristic_hire_prob(resume_text, jd_text)


def _heuristic_hire_prob(resume_text: str, jd_text: str) -> float:
    """
    Heuristic fallback when XGBoost model is not trained.
    Uses keyword overlap ratio as proxy.
    """
    jd_words = set(re.findall(r"\b\w{4,}\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b\w{4,}\b", resume_text.lower()))
    if not jd_words:
        return 0.5
    overlap = len(jd_words & resume_words) / len(jd_words)
    return round(min(overlap * 1.5, 1.0), 4)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def composite_score(tfidf_score: float, xgb_score: float) -> float:
    """Weighted composite: 60% TF-IDF ATS, 40% XGBoost."""
    return round(0.6 * tfidf_score + 0.4 * xgb_score, 4)
