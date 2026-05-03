"""OpenAI Moderation API helper shared across the server and eval layers."""

from __future__ import annotations

import os
from typing import Any, List

from openai import AsyncOpenAI

MODERATION_MODEL = "omni-moderation-latest"

REJECTION_PREFIX = "This query violates the site's policy for these categories."


def flagged_categories(moderation: Any) -> List[str]:
    """Return the sorted flagged category names from an OpenAI moderation result."""
    if not getattr(moderation, "results", None):
        return []
    result = moderation.results[0]
    if not bool(getattr(result, "flagged", False)):
        return []
    categories = getattr(result, "categories", None)
    if categories is None:
        return []
    raw = (
        categories.model_dump(by_alias=True)
        if hasattr(categories, "model_dump")
        else dict(categories)
    )
    return sorted(str(name) for name, flagged in raw.items() if flagged)


async def moderate_query(query: str) -> List[str]:
    """Run *query* through the OpenAI Moderation API.

    Returns:
        A sorted list of flagged category names, or an empty list when the
        query is clean.
    """
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    result = await client.moderations.create(model=MODERATION_MODEL, input=query)
    return flagged_categories(result)


def rejection_message(categories: List[str]) -> str:
    """Format the user-facing single-line rejection from a list of categories.

    Matches the ``FincentApiError`` message format produced by the API client
    so that eval expected_output values stay in sync with what the UI displays.
    """
    return f"{REJECTION_PREFIX} {', '.join(categories)}"
