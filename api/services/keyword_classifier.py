from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from config import settings
from models import Dish

logger = logging.getLogger(__name__)


class KeywordSource(BaseModel):
    """Metadata entry for keyword datasets."""

    name: str
    url: Optional[str] = None


class KeywordMetadata(BaseModel):
    """Metadata block stored alongside keyword lists."""

    generated_at: Optional[str] = None
    sources: List[KeywordSource] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PositiveKeywords(BaseModel):
    """Positive vegetarian indicators."""

    ingredient_keywords: List[str] = Field(default_factory=list)
    dish_keywords: List[str] = Field(default_factory=list)
    contains_hints: List[str] = Field(default_factory=list)


class NegativeKeywords(BaseModel):
    """Negative (non-vegetarian) indicators."""

    meat_terms: List[str] = Field(default_factory=list)
    seafood_terms: List[str] = Field(default_factory=list)
    stock_keywords: List[str] = Field(default_factory=list)


class KeywordDataset(BaseModel):
    """Container for the keyword classifier dataset."""

    metadata: KeywordMetadata = Field(default_factory=KeywordMetadata)
    positive: PositiveKeywords = Field(default_factory=PositiveKeywords)
    negative: NegativeKeywords = Field(default_factory=NegativeKeywords)


@dataclass
class KeywordClassification:
    """Result bundle returned by the keyword classifier."""

    is_vegetarian: bool
    confidence: float
    reasoning: str
    signals: Dict[str, List[str]]
    is_uncertain: bool


class KeywordClassifierError(RuntimeError):
    """Raised when the keyword classifier cannot load or parse data."""


class KeywordClassifier:
    """Deterministic vegetarian classifier based on curated keyword datasets."""

    _normalize_pattern = re.compile(r"[^a-z0-9\s]")

    def __init__(
        self,
        *,
        dataset_path: Optional[Path | str] = None,
        dataset: Optional[KeywordDataset] = None,
        confidence_threshold: float = settings.confidence_threshold,
        fuzzy_threshold: float = settings.keyword_fuzzy_threshold,
        hint_bonus_cap: float = settings.keyword_max_hint_bonus,
    ) -> None:
        self.dataset_path = Path(dataset_path or settings.keyword_data_path)
        self._confidence_threshold = confidence_threshold
        self._fuzzy_threshold = fuzzy_threshold
        self._hint_bonus_cap = hint_bonus_cap

        if dataset is not None:
            self._dataset = dataset
        else:
            self._dataset = self._load_dataset(self.dataset_path)

        self._prepare_keyword_sets()

        logger.info(
            "KeywordClassifier initialised with %d dish keywords, %d ingredient keywords",
            len(self._dish_lookup),
            len(self._ingredient_keywords),
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def classify(self, dish: Dish) -> KeywordClassification:
        """
        Classify a dish using keyword heuristics.

        Args:
            dish: Parsed dish payload (name, price, raw_text)
        """
        full_text = f"{dish.name} {dish.raw_text or ''}".strip()
        normalized_name = self._normalize_text(dish.name)
        normalized_full_text = self._normalize_text(full_text)
        tokens = self._tokenise(full_text)
        token_set = set(tokens)

        signals: Dict[str, List[str]] = {}

        # Negative signals short-circuit to non-vegetarian
        negative_hits = self._match_keywords(self._negative_keywords, token_set, normalized_full_text)
        if negative_hits:
            signals["negative_matches"] = negative_hits
            confidence = min(1.0, 0.75 + 0.05 * len(negative_hits))
            reasoning = f"Detected non-vegetarian keywords: {', '.join(negative_hits)}"
            return KeywordClassification(
                is_vegetarian=False,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                signals=signals,
                is_uncertain=confidence < self._confidence_threshold,
            )

        # Positive signals
        exact_match = self._dish_lookup.get(normalized_name)
        if exact_match:
            signals["exact_matches"] = [exact_match]
            confidence = 0.92
            reasoning = f"Exact dish match for '{exact_match}'"
            hint_bonus = self._hint_bonus(normalized_full_text, token_set, signals)
            confidence = min(1.0, confidence + hint_bonus)
            return KeywordClassification(
                is_vegetarian=True,
                confidence=round(confidence, 3),
                reasoning=reasoning + self._format_hint_suffix(signals),
                signals=signals,
                is_uncertain=confidence < self._confidence_threshold,
            )

        ingredient_hits = self._match_keywords(self._ingredient_keywords, token_set, normalized_full_text)
        fuzzy_match, fuzzy_ratio = self._best_fuzzy_match(normalized_name)
        hint_bonus = self._hint_bonus(normalized_full_text, token_set, signals)

        if ingredient_hits:
            signals["ingredient_matches"] = ingredient_hits
            confidence = min(0.9, 0.65 + 0.05 * len(ingredient_hits) + hint_bonus)
            reasoning = f"Ingredient keywords matched: {', '.join(ingredient_hits)}"
            return KeywordClassification(
                is_vegetarian=True,
                confidence=round(confidence, 3),
                reasoning=reasoning + self._format_hint_suffix(signals),
                signals=signals,
                is_uncertain=confidence < self._confidence_threshold,
            )

        if fuzzy_match and fuzzy_ratio >= self._fuzzy_threshold:
            display_ratio = round(fuzzy_ratio, 2)
            signals["fuzzy_match"] = [f"{fuzzy_match} ({display_ratio})"]
            confidence = max(0.55, fuzzy_ratio * 0.9 + hint_bonus)
            reasoning = f"Fuzzy match to '{fuzzy_match}' ({display_ratio})"
            return KeywordClassification(
                is_vegetarian=True,
                confidence=round(min(confidence, 0.9), 3),
                reasoning=reasoning + self._format_hint_suffix(signals),
                signals=signals,
                is_uncertain=confidence < self._confidence_threshold,
            )

        # Hints alone yield weak positive; default to non-veg with low confidence
        hint_matches = signals.get("hint_matches")
        if hint_matches:
            confidence = max(0.4, 0.45 + 0.05 * len(hint_matches))
            reasoning = f"Vegetarian hints present: {', '.join(hint_matches)}"
            return KeywordClassification(
                is_vegetarian=True,
                confidence=round(min(confidence, 0.65), 3),
                reasoning=reasoning,
                signals=signals,
                is_uncertain=True,
            )

        confidence = 0.25
        reasoning = "No vegetarian indicators detected; defaulting to non-vegetarian"
        return KeywordClassification(
            is_vegetarian=False,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
            is_uncertain=True,
        )

    def classify_and_update(self, dish: Dish) -> KeywordClassification:
        """
        Classify the dish and apply the results directly onto the model.
        """
        result = self.classify(dish)
        dish.is_vegetarian = result.is_vegetarian
        dish.confidence = result.confidence
        dish.classification_method = "keyword"
        dish.reasoning = result.reasoning
        dish.signals = result.signals or None
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _hint_bonus(
        self,
        normalized_text: str,
        token_set: set[str],
        signals: Dict[str, List[str]],
    ) -> float:
        """Assign a small bonus for general vegetarian hints."""
        hint_matches = self._match_keywords(self._hint_keywords, token_set, normalized_text)
        if hint_matches:
            signals["hint_matches"] = hint_matches
            bonus = min(self._hint_bonus_cap, 0.05 * len(hint_matches))
            return bonus
        return 0.0

    def _match_keywords(
        self,
        keywords: Sequence[str],
        token_set: set[str],
        normalized_text: str,
    ) -> List[str]:
        hits: List[str] = []
        for keyword in keywords:
            cleaned = keyword.strip()
            if not cleaned:
                continue
            normalized_keyword = self._normalize_text(cleaned)
            if not normalized_keyword:
                continue
            if " " in normalized_keyword:
                if normalized_keyword in normalized_text:
                    hits.append(cleaned)
            else:
                if normalized_keyword in token_set:
                    hits.append(cleaned)
        return hits

    def _best_fuzzy_match(self, normalized_name: str) -> tuple[Optional[str], float]:
        """Find the closest dish keyword using sequence similarity."""
        if not normalized_name or not self._normalized_dish_keywords:
            return None, 0.0

        best_match = None
        best_ratio = 0.0

        for keyword in self._normalized_dish_keywords:
            ratio = SequenceMatcher(None, normalized_name, keyword).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = self._dish_lookup[keyword]

        return best_match, best_ratio

    def _prepare_keyword_sets(self) -> None:
        """Cache normalised keyword collections for quick lookups."""
        positive = self._dataset.positive
        negative = self._dataset.negative

        self._ingredient_keywords = self._dedupe_preserve_case(positive.ingredient_keywords)
        self._hint_keywords = self._dedupe_preserve_case(positive.contains_hints)

        # Dish lookup preserves original casing for reporting
        self._dish_lookup: Dict[str, str] = {}
        for dish in positive.dish_keywords:
            normalized = self._normalize_text(dish)
            if normalized:
                self._dish_lookup[normalized] = dish
        self._normalized_dish_keywords = list(self._dish_lookup.keys())

        negative_terms = (
            list(negative.meat_terms)
            + list(negative.seafood_terms)
            + list(negative.stock_keywords)
        )
        self._negative_keywords = self._dedupe_preserve_case(negative_terms)

    def _load_dataset(self, path: Path) -> KeywordDataset:
        """Load keyword dataset from disk."""
        if not path.exists():
            raise KeywordClassifierError(f"Keyword dataset not found at {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return KeywordDataset.parse_obj(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise KeywordClassifierError(f"Failed to load keyword dataset: {exc}") from exc

    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return re.sub(cls._normalize_pattern, " ", value.lower()).strip()

    @staticmethod
    def _tokenise(value: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", value.lower())

    @staticmethod
    def _format_hint_suffix(signals: Dict[str, List[str]]) -> str:
        hints = signals.get("hint_matches")
        if hints:
            return f"; vegetarian hints: {', '.join(hints)}"
        return ""

    @staticmethod
    def _dedupe_preserve_case(values: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        unique: List[str] = []
        for value in values:
            cleaned = value.strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            unique.append(cleaned)
        return unique
