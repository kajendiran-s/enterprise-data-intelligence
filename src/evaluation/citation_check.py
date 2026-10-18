
import re
from typing import Any


def validate_citations(
    citation_ids: list[Any],
    context: str,
) -> dict:
    """Validate generated citation IDs against source IDs in the context."""

    available_ids = {
        int(match)
        for match in re.findall(r"\[Source (\d+)\]", context or "")
    }

    valid_ids = []
    invalid_ids = []

    for citation_id in citation_ids or []:
        if (
            isinstance(citation_id, int)
            and not isinstance(citation_id, bool)
            and citation_id in available_ids
        ):
            valid_ids.append(citation_id)
        else:
            invalid_ids.append(citation_id)

    return {
        "valid": len(invalid_ids) == 0,
        "available_source_ids": sorted(available_ids),
        "valid_citation_ids": valid_ids,
        "invalid_citation_ids": invalid_ids,
    }
