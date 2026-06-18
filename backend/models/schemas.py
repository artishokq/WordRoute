"""Pydantic schemas for API request/response."""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    input: str = Field(..., description="Russian word, words list, or text", min_length=1)
    mode: Literal["word", "text", "batch"] = Field("text", description="Analysis mode")


class DonorCandidate(BaseModel):
    language: str
    probability: float


class GlottologInfo(BaseModel):
    glottocode: Optional[str] = None
    family: Optional[str] = None
    subfamily: Optional[str] = None
    macroarea: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    iso639: Optional[str] = None
    description: Optional[str] = None


class WordResult(BaseModel):
    word: str
    lemma: str
    pos: Optional[str] = None
    gender: Optional[str] = None
    is_declinable: bool = True
    loanword_probability: float
    is_loanword: bool
    donor_language: str
    donor_language_ru: Optional[str] = None
    donor_family: Optional[str] = None
    donor_subfamily: Optional[str] = None
    source_word: Optional[str] = None
    semantic_domain: Optional[str] = None
    semantic_domain_ru: Optional[str] = None
    explanation: list[str] = []
    morphological_derivatives: list[str] = []
    top_donors: list[DonorCandidate] = []
    glottolog: Optional[GlottologInfo] = None
    in_seed: bool = False


class AnalysisStats(BaseModel):
    total_words: int
    borrowings_found: int
    native_words: int
    borrowing_rate: float
    top_donor: Optional[str] = None
    donor_distribution: dict[str, int] = {}
    semantic_distribution: dict[str, int] = {}


class AnalyzeResponse(BaseModel):
    input_text: str
    mode: str
    words: list[WordResult]
    stats: AnalysisStats
