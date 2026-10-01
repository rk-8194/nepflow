"""Shared constants and utilities for the train_nep sub-stages."""

import logging

logger = logging.getLogger("nepflow.train_nep")

# Try to import tqdm for progress bars; graceful fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        """Fallback: return iterable as-is if tqdm is not available."""
        return iterable
