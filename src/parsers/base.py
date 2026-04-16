"""Abstract base parser interface.

All parsers inherit from this and produce the same ParsedPaper output.
"""

from abc import ABC, abstractmethod

from src.models.parsed_paper import ParsedPaper


class BaseParser(ABC):
    """Base class for all input format parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedPaper:
        """Parse input file and return normalized ParsedPaper."""
        ...

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        ...
