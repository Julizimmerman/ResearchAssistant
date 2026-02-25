"""SQLite-based LLM response cache.

Caches identical prompt+model combinations so repeated calls (e.g. similar
research topics) are served from disk instead of making paid API requests.
"""

from __future__ import annotations

import logging
import os

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

logger = logging.getLogger(__name__)


def setup_cache(cache_path: str = ".llm_cache.db") -> None:
    """Initialise the global LLM response cache backed by SQLite.

    Creates parent directories if needed.  All subsequent ``ChatOpenAI``
    calls will automatically use this cache.
    """
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cache = SQLiteCache(database_path=cache_path)
    set_llm_cache(cache)
    logger.info("LLM cache initialised at %s", cache_path)
