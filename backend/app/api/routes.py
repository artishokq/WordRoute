from __future__ import annotations
from collections import Counter

from fastapi import APIRouter, HTTPException

from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisStats,
    WordResult,
    GlottologInfo,
    DonorCandidate,
)
from app.core.classifier import get_classifier
from app.core.preprocessor import preprocess_text, preprocess_word
from app.core.explainer import generate_explanation
from app.core.enricher import get_glottolog_info, get_morphological_derivatives, get_word_card

router = APIRouter(prefix="/api")


def _analyze_single(word: str) -> WordResult:
    """Run the full analysis pipeline for a single Russian word."""
    clf = get_classifier()

    morph = preprocess_word(word)
    lemma = morph.get("lemma", word)

    prediction = clf.predict(word)

    glottolog_raw = get_glottolog_info(prediction.get("donor_language", ""))
    derivatives = get_morphological_derivatives(lemma)

    explanation = generate_explanation(
        word=word,
        lemma=lemma,
        feats=prediction.get("features", {}),
        loanword_prob=prediction.get("loanword_probability", 0.5),
        donor_language=prediction.get("donor_language", "Unknown"),
        source_word=prediction.get("source_word", ""),
        in_seed=prediction.get("in_seed", False),
    )

    card = get_word_card(
        word=word,
        lemma=lemma,
        morph_info=morph,
        prediction=prediction,
        glottolog=glottolog_raw,
        derivatives=derivatives,
        explanation=explanation,
    )

    glottolog = GlottologInfo(**glottolog_raw) if glottolog_raw else None
    top_donors = [
        DonorCandidate(language=d["language"], probability=d["probability"])
        for d in prediction.get("top_donors", [])
    ]

    return WordResult(
        word=card["word"],
        lemma=card["lemma"],
        pos=card["pos"],
        gender=card["gender"],
        is_declinable=card["is_declinable"],
        loanword_probability=card["loanword_probability"],
        is_loanword=card["is_loanword"],
        donor_language=card["donor_language"],
        donor_language_ru=card["donor_language_ru"],
        donor_family=card["donor_family"],
        donor_subfamily=card["donor_subfamily"],
        source_word=card["source_word"],
        semantic_domain=card["semantic_domain"],
        semantic_domain_ru=card["semantic_domain_ru"],
        explanation=card["explanation"],
        morphological_derivatives=card["morphological_derivatives"],
        top_donors=top_donors,
        glottolog=glottolog,
        in_seed=card["in_seed"],
    )


def _build_stats(words: list[WordResult], input_text: str) -> AnalysisStats:
    """Compute aggregate borrowing statistics over a list of analyzed words."""
    borrowings = [w for w in words if w.is_loanword]
    donor_counts: Counter = Counter(w.donor_language for w in borrowings)
    semantic_counts: Counter = Counter(
        w.semantic_domain for w in borrowings if w.semantic_domain
    )
    top_donor = donor_counts.most_common(1)[0][0] if donor_counts else None
    total = len(words)
    b_count = len(borrowings)

    return AnalysisStats(
        total_words=total,
        borrowings_found=b_count,
        native_words=total - b_count,
        borrowing_rate=round(b_count / max(total, 1), 3),
        top_donor=top_donor,
        donor_distribution=dict(donor_counts),
        semantic_distribution=dict(semantic_counts),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    clf = get_classifier()
    if not clf.trained:
        raise HTTPException(status_code=503, detail="Classifier not ready")

    text = body.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    if body.mode == "word":
        raw_words = [w.strip() for w in text.replace(",", " ").split() if w.strip()]
        words_to_analyze = raw_words[:100]
    else:
        tokens = preprocess_text(text, keep_all_pos=False)
        words_to_analyze = [t["token"] for t in tokens][:200]

    if not words_to_analyze:
        raise HTTPException(status_code=422, detail="No processable Russian words found in input")

    results = []
    for word in words_to_analyze:
        try:
            result = _analyze_single(word)
            results.append(result)
        except Exception:
            continue

    results.sort(key=lambda r: (-int(r.is_loanword), -r.loanword_probability))

    stats = _build_stats(results, text)

    return AnalyzeResponse(
        input_text=text[:200] + ("..." if len(text) > 200 else ""),
        mode=body.mode,
        words=results,
        stats=stats,
    )


@router.get("/word/{word}", response_model=WordResult)
async def analyze_word(word: str):
    clf = get_classifier()
    if not clf.trained:
        raise HTTPException(status_code=503, detail="Classifier not ready")
    return _analyze_single(word)


@router.get("/health")
async def health():
    clf = get_classifier()
    return {"status": "ok", "trained": clf.trained}
