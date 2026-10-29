from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    REPO_ROOT
    / "ebook-pdf"
    / "Site Reliability Engineering_ How Google Runs Production Systems ( PDFDrive ).pdf"
)


def pytest_collection_modifyitems(config, items):
    if PDF_PATH.exists():
        return
    skip = pytest.mark.skip(reason=f"sample PDF not found: {PDF_PATH}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
