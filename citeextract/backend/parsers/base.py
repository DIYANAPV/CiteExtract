
from abc import ABC, abstractmethod

from citeextract.models.parsed_paper import ParsedPaper


class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> ParsedPaper:
        ...

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        ...
