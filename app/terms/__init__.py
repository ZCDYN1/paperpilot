"""F11「术语解释」：术语词表与文本标注。"""

from __future__ import annotations

from app.terms.glossary import annotate, extract_terms, lookup, unknown_terms

__all__ = ["annotate", "extract_terms", "lookup", "unknown_terms"]
