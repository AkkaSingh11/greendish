import sys
from pathlib import Path

import pytest

# Ensure application modules are importable when running from repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
API_ROOT = ROOT_DIR / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from models import Dish  # noqa: E402
from services.keyword_classifier import (  # noqa: E402
    KeywordClassifier,
    KeywordDataset,
    KeywordMetadata,
    NegativeKeywords,
    PositiveKeywords,
)


@pytest.fixture()
def sample_dataset() -> KeywordDataset:
    return KeywordDataset(
        metadata=KeywordMetadata(notes=["test dataset"]),
        positive=PositiveKeywords(
            ingredient_keywords=[
                "paneer",
                "spinach",
                "chickpea",
                "lentil",
            ],
            dish_keywords=[
                "palak paneer",
                "chana masala",
                "masala dosa",
            ],
            contains_hints=[
                "veg",
                "vegetarian",
            ],
        ),
        negative=NegativeKeywords(
            meat_terms=["chicken", "beef", "lamb"],
            seafood_terms=["shrimp", "fish"],
            stock_keywords=["chicken stock"],
        ),
    )


@pytest.fixture()
def classifier(sample_dataset: KeywordDataset) -> KeywordClassifier:
    return KeywordClassifier(dataset=sample_dataset, confidence_threshold=0.7, fuzzy_threshold=0.8)


def test_exact_match_confers_high_confidence(classifier: KeywordClassifier) -> None:
    dish = Dish(name="Palak Paneer", raw_text="Palak Paneer ........ 12.50", price=12.5, confidence=0.5)

    result = classifier.classify_and_update(dish)

    assert result.is_vegetarian is True
    assert result.confidence >= 0.9
    assert dish.is_vegetarian is True
    assert dish.classification_method == "keyword"
    assert "palak paneer" in dish.reasoning.lower()
    assert dish.signals and "exact_matches" in dish.signals


def test_negative_keyword_overrides_positive(classifier: KeywordClassifier) -> None:
    dish = Dish(name="Chicken Paneer Surprise", raw_text="Chicken Paneer Surprise .... 11.00", price=11.0, confidence=0.5)

    result = classifier.classify_and_update(dish)

    assert result.is_vegetarian is False
    assert result.confidence >= 0.75
    assert dish.is_vegetarian is False
    assert dish.signals and "negative_matches" in dish.signals


def test_fuzzy_match_activates_for_close_names(classifier: KeywordClassifier) -> None:
    dish = Dish(name="Palak Paner", raw_text="Palak Paner ...... 10.00", price=10.0, confidence=0.5)

    result = classifier.classify_and_update(dish)

    assert result.is_vegetarian is True
    assert result.confidence >= 0.55
    assert dish.signals and "fuzzy_match" in dish.signals


def test_uncertain_when_no_signals(classifier: KeywordClassifier) -> None:
    dish = Dish(name="Chef Special", raw_text="Chef Special ...... 9.00", price=9.0, confidence=0.5)

    result = classifier.classify_and_update(dish)

    assert result.is_vegetarian is False
    assert result.is_uncertain is True
    assert result.confidence <= 0.3
    assert dish.signals is None
