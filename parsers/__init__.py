# parsers/__init__.py
from parsers.base_parser import BaseParser, SCHEMA_COLUMNS
from parsers.shanghaitech_parser import ShanghaiTechParser
from parsers.ucf_qnrf_parser import UCFQNRFParser
from parsers.pets2009_parser import PETS2009Parser

__all__ = [
    "BaseParser",
    "SCHEMA_COLUMNS",
    "ShanghaiTechParser",
    "UCFQNRFParser",
    "PETS2009Parser",
]
