import logging
import re
from collections import deque
from typing import List, Optional

from models.schemas import Dish

logger = logging.getLogger(__name__)


class ParserService:
    """Service for parsing OCR text into structured dish objects."""

    def __init__(self):
        """Initialize parser with regex patterns."""
        # Price patterns - match various formats
        self.price_patterns = [
            r'\$\s*(\d+\.?\d{0,2})',  # $12.99, $12, $ 12.99
            r'(\d+\.\d{2})\s*$',       # 12.99 at end of line
            r'(\d+\.\d{2})',           # 12.99 anywhere
            r'\$(\d+)',                 # $12 (no cents)
        ]

        # Compile price pattern (combined)
        self.price_regex = re.compile(
            r'(?:\$\s*)?(\d+(?:\.\d{2})?)',
            re.MULTILINE
        )

        # Dish name patterns - typically at start of line, may have numbers
        # Handles: "1. Greek Salad", "Greek Salad", "GREEK SALAD", etc.
        self.dish_line_pattern = re.compile(
            r'^[\s\d\.\-\*]*([A-Z][A-Za-z\s&\'\-]+?)(?:\s*[\.\s]+\s*|\s{2,}|\s*\$)',
            re.MULTILINE
        )

        logger.info("ParserService initialized with regex patterns")

    def parse_menu_text(self, raw_text: str) -> List[Dish]:
        """
        Parse OCR text into structured dish objects.

        Args:
            raw_text: Raw text extracted from OCR

        Returns:
            List of Dish objects with names and prices
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty or whitespace-only text provided to parser")
            return []

        dishes = []
        lines = raw_text.split('\n')

        # Try line-by-line parsing first
        dishes = self._parse_line_by_line(lines)

        # If that yields few results, try alternative parsing
        if len(dishes) < 3:
            logger.info("Line-by-line parsing yielded few results, trying alternative approach")
            alternative_dishes = self._parse_with_context(raw_text)
            if len(alternative_dishes) > len(dishes):
                dishes = alternative_dishes

        logger.info(f"Parsed {len(dishes)} dishes from text")
        return dishes

    def _parse_line_by_line(self, lines: List[str]) -> List[Dish]:
        """
        Parse text line by line, matching dish names with prices.

        Args:
            lines: List of text lines

        Returns:
            List of Dish objects
        """
        dishes: List[Dish] = []
        pending_dishes = deque()

        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            # Handle lines containing multiple inline prices (e.g., two columns merged)
            price_matches = [
                match for match in self.price_regex.finditer(line)
                if self._is_valid_price_token(match.group(0))
            ]
            if len(price_matches) > 1:
                segment_start = 0
                for match in price_matches:
                    segment_text = line[segment_start:match.start()]
                    segment_text = self._clean_candidate_text(segment_text)
                    price_value = self._extract_price(match.group(0))

                    if not segment_text or price_value is None:
                        segment_start = match.end()
                        continue

                    dish_name = self._extract_dish_name(segment_text, check_price=False)
                    if dish_name:
                        raw_segment = f"{segment_text.strip()} {match.group(0)}".strip()
                        dishes.append(Dish(
                            name=dish_name,
                            price=price_value,
                            raw_text=raw_segment,
                            confidence=self._calculate_confidence(dish_name, price_value, [raw_segment])
                        ))
                    segment_start = match.end()
                # Skip further processing since we've handled this line
                continue

            # Check if line contains a price
            price = self._extract_price(line)

            # Check if line looks like a dish name
            dish_name = self._extract_dish_name(line, price is not None)

            if dish_name and price:
                # Complete dish on single line
                raw_lines = [line]
                dishes.append(Dish(
                    name=dish_name,
                    price=price,
                    raw_text=' '.join(raw_lines),
                    confidence=self._calculate_confidence(dish_name, price, raw_lines)
                ))
                continue

            if dish_name and not price:
                # Dish name without price - queue it until we see a price
                pending_dishes.append({
                    'name': dish_name,
                    'raw_lines': [self._clean_candidate_text(line)],
                })
                continue

            if price and pending_dishes:
                # Assign price to the oldest pending dish
                dish_info = pending_dishes.popleft()
                raw_lines = dish_info['raw_lines'] + [self._clean_candidate_text(line)]
                dishes.append(Dish(
                    name=dish_info['name'],
                    price=price,
                    raw_text=' '.join(raw_lines),
                    confidence=self._calculate_confidence(dish_info['name'], price, raw_lines)
                ))
                continue

            if not dish_name and not price and pending_dishes:
                # Description line - attach to the most recent pending dish
                pending_dishes[-1]['raw_lines'].append(self._clean_candidate_text(line))

        # Flush any pending dishes without prices
        while pending_dishes:
            dish_info = pending_dishes.popleft()
            raw_lines = dish_info['raw_lines']
            dishes.append(Dish(
                name=dish_info['name'],
                price=None,
                raw_text=' '.join(raw_lines),
                confidence=self._calculate_confidence(dish_info['name'], None, raw_lines)
            ))

        return dishes

    def _parse_with_context(self, raw_text: str) -> List[Dish]:
        """
        Alternative parsing using regex patterns with more context.

        Args:
            raw_text: Complete raw text

        Returns:
            List of Dish objects
        """
        dishes = []

        # Split into potential dish blocks (separated by blank lines or clear separators)
        blocks = re.split(r'\n\s*\n', raw_text)

        for block in blocks:
            if not block.strip():
                continue

            # Try to find dish name and price in block
            lines = [l.strip() for l in block.split('\n') if l.strip()]

            if not lines:
                continue

            # First line is likely the dish name
            dish_name = self._extract_dish_name(lines[0], check_price=False)

            # Search for price in any line of the block
            price = None
            for line in lines:
                price = self._extract_price(line)
                if price:
                    break

            if dish_name:
                dishes.append(Dish(
                    name=dish_name,
                    price=price,
                    raw_text=block.strip(),
                    confidence=self._calculate_confidence(dish_name, price, lines)
                ))

        return dishes

    def _clean_candidate_text(self, text: str) -> str:
        """Normalize candidate text segments by stripping filler characters."""
        if not text:
            return ''

        text = text.replace('…', ' ')
        text = re.sub(r'[\u2022·•]+', ' ', text)
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'[_=]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_valid_price_token(self, token: str) -> bool:
        """Check if the matched token is likely representing a price."""
        if not token:
            return False

        token = token.strip()
        if '$' in token:
            return True

        # Require decimal point to avoid calorie counts like "990"
        return bool(re.search(r'\d+\.\d{2}', token))

    def _extract_dish_name(self, line: str, check_price: bool = True) -> Optional[str]:
        """
        Extract dish name from a line of text.

        Args:
            line: Text line
            check_price: If True, remove price from the name

        Returns:
            Dish name or None
        """
        if not line:
            return None

        line = self._clean_candidate_text(line)

        # Remove leading numbers, bullets, dashes
        line = re.sub(r'^[\s\d\.\-\*\)]+', '', line).strip()

        # Remove price if present and check_price is True
        if check_price:
            line = re.sub(r'\s*\$?\s*\d+\.\d{2}\s*$', '', line)
            line = re.sub(r'\s*\$\d+\s*$', '', line)

        # Clean up extra whitespace
        line = re.sub(r'\s+', ' ', line).strip()

        # Must have at least 3 chars and start with letter
        if len(line) < 3 or not line[0].isalpha():
            return None

        # Must contain at least one letter (not just numbers/symbols)
        if not any(c.isalpha() for c in line):
            return None

        # Remove trailing dots, commas, etc.
        line = re.sub(r'[\.,:;]+$', '', line).strip()

        # Check if it looks like a valid dish name (not headers like "APPETIZERS", "MENU", etc.)
        header_keywords = {
            'appetizer', 'appetizers', 'entree', 'entrees', 'dessert', 'desserts',
            'beverage', 'beverages', 'drink', 'drinks', 'menu', 'breakfast',
            'lunch', 'dinner', 'side', 'sides', 'page', 'continued', 'special',
            'specials', 'daily', 'salads', 'restaurant', 'starters', 'starter',
            'lunch specials'
        }
        normalized = re.sub(r'[^a-z\s]', '', line.lower()).strip()
        if 'www' in normalized:
            return None
        if normalized in header_keywords and len(normalized.split()) <= 2:
            return None

        return line if len(line) >= 3 else None

    def _extract_price(self, text: str) -> Optional[float]:
        """
        Extract price from text.

        Args:
            text: Text containing potential price

        Returns:
            Price as float or None
        """
        if not text:
            return None

        # Try to find price with dollar sign first
        match = re.search(r'\$\s*(\d+(?:\.\d{2})?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Try to find price pattern at end of line
        match = re.search(r'(\d+\.\d{2})\s*$', text)
        if match:
            try:
                price = float(match.group(1))
                # Sanity check: reasonable menu price range
                if 0.5 <= price <= 999.99:
                    return price
            except ValueError:
                pass

        # Try to find any price-like pattern
        match = re.search(r'\b(\d+\.\d{2})\b', text)
        if match:
            try:
                price = float(match.group(1))
                if 0.5 <= price <= 999.99:
                    return price
            except ValueError:
                pass

        return None

    def _calculate_confidence(
        self,
        dish_name: Optional[str],
        price: Optional[float],
        raw_lines: List[str]
    ) -> float:
        """
        Calculate parsing confidence score.

        Args:
            dish_name: Extracted dish name
            price: Extracted price
            raw_lines: Raw text lines used

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0

        # Base confidence for having a dish name
        if dish_name:
            confidence += 0.4

            # Bonus for reasonable length (not too short, not too long)
            if 5 <= len(dish_name) <= 50:
                confidence += 0.1

            # Bonus for proper capitalization
            if dish_name[0].isupper():
                confidence += 0.05

        # Confidence for having a price
        if price is not None:
            confidence += 0.3

            # Bonus for reasonable price range
            if 1.0 <= price <= 100.0:
                confidence += 0.1

        # Penalty for very long raw text (might be misparse)
        if raw_lines and len(raw_lines) > 5:
            confidence -= 0.1

        # Bonus for clean, single-line parse
        if raw_lines and len(raw_lines) == 1:
            confidence += 0.05

        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))

    def get_parsing_stats(self, dishes: List[Dish]) -> dict:
        """
        Calculate parsing statistics.

        Args:
            dishes: List of parsed dishes

        Returns:
            Dictionary with parsing statistics
        """
        if not dishes:
            return {
                'total_dishes': 0,
                'dishes_with_prices': 0,
                'dishes_without_prices': 0,
                'average_confidence': 0.0,
                'price_coverage': 0.0
            }

        dishes_with_prices = sum(1 for d in dishes if d.price is not None)
        avg_confidence = sum(d.confidence or 0.0 for d in dishes) / len(dishes)

        return {
            'total_dishes': len(dishes),
            'dishes_with_prices': dishes_with_prices,
            'dishes_without_prices': len(dishes) - dishes_with_prices,
            'average_confidence': round(avg_confidence, 3),
            'price_coverage': round(dishes_with_prices / len(dishes), 3)
        }
