"""Corpus provider base module."""

from .download import download_and_extract, download_file, extract_archive
from .lib import BaseProvider, DataType, StandardizedData

__all__ = [
    "BaseProvider",
    "DataType",
    "StandardizedData",
    "download_file",
    "extract_archive",
    "download_and_extract",
]
