"""
agent/rag.py
Simple keyword-based RAG over the AutoStream knowledge base.
Loads autostream_kb.json and returns the most relevant chunks
given a user query.
"""

import json
import os
from pathlib import Path


def _load_kb() -> dict:
    kb_path = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        return json.load(f)


def _build_chunks(kb: dict) -> list[dict]:
    """Convert the structured KB into flat text chunks for retrieval."""
    chunks = []

    # Company overview
    chunks.append({
        "id": "overview",
        "topic": "company overview",
        "text": (
            f"{kb['company']} — {kb['tagline']}. "
            f"{kb['description']} "
            f"Supported platforms: {', '.join(kb['supported_platforms'])}."
        ),
    })

    # Each plan as a chunk
    for plan in kb["plans"]:
        features_text = "; ".join(plan["features"])
        chunks.append({
            "id": f"plan_{plan['name'].lower().replace(' ', '_')}",
            "topic": "pricing plan",
            "text": (
                f"{plan['name']}: {plan['price']}. "
                f"Features: {features_text}. "
                f"Best for: {plan['best_for']}."
            ),
        })

    # All plans together as a pricing summary chunk
    pricing_lines = []
    for plan in kb["plans"]:
        pricing_lines.append(
            f"{plan['name']} ({plan['price']}): {'; '.join(plan['features'])}"
        )
    chunks.append({
        "id": "pricing_summary",
        "topic": "pricing comparison",
        "text": "AutoStream pricing: " + " | ".join(pricing_lines),
    })

    # Policies
    for policy in kb["policies"]:
        chunks.append({
            "id": f"policy_{policy['topic'].lower().replace(' ', '_')}",
            "topic": "policy",
            "text": f"{policy['topic']}: {policy['details']}",
        })

    # FAQs
    for faq in kb["faqs"]:
        chunks.append({
            "id": "faq",
            "topic": "faq",
            "text": f"Q: {faq['question']} A: {faq['answer']}",
        })

    return chunks


# ── Simple keyword-based retrieval ──────────────────────────────────────────

PRICING_KEYWORDS  = {"price", "pricing", "cost", "plan", "plans", "basic", "pro",
                     "how much", "subscription", "fee", "monthly", "per month", "upgrade"}
POLICY_KEYWORDS   = {"refund", "cancel", "cancellation", "support", "trial", "free"}
PLATFORM_KEYWORDS = {"youtube", "instagram", "tiktok", "twitch", "facebook", "platform"}
FEATURE_KEYWORDS  = {"4k", "720p", "captions", "subtitle", "video", "resolution",
                     "unlimited", "ai", "template", "editing", "feature"}


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Return the top_k most relevant knowledge-base chunks as a single
    formatted string, ready to be injected into the LLM prompt.
    """
    kb = _load_kb()
    chunks = _build_chunks(kb)

    q_lower = query.lower()
    q_words = set(q_lower.split())

    scored: list[tuple[int, dict]] = []
    for chunk in chunks:
        score = 0
        text_lower = chunk["text"].lower()

        # Direct word overlap
        for word in q_words:
            if word in text_lower:
                score += 2

        # Topic-based boosts
        if any(kw in q_lower for kw in PRICING_KEYWORDS) and "plan" in chunk["topic"]:
            score += 5
        if any(kw in q_lower for kw in POLICY_KEYWORDS) and "policy" in chunk["topic"]:
            score += 5
        if any(kw in q_lower for kw in FEATURE_KEYWORDS) and "plan" in chunk["topic"]:
            score += 3
        if "pricing" in q_lower and chunk["id"] == "pricing_summary":
            score += 8

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:top_k] if _ > 0]

    if not top_chunks:
        # Fallback: return the overview + pricing summary
        top_chunks = [c for c in chunks if c["id"] in ("overview", "pricing_summary")]

    result = "\n\n".join(
        f"[{c['topic'].upper()}]\n{c['text']}" for c in top_chunks
    )
    return result
