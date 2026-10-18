
import re


def validate_final_citations(
    answer: str,
    source_map: list[dict],
) -> dict:
    """Validate RAG document citations in a final answer."""

    cited_ids = [
        int(value)
        for value in re.findall(r"\[Source (\d+)\]", answer)
    ]

    available_ids = {
        item["citation_id"]
        for item in source_map
        if isinstance(item.get("citation_id"), int)
        and not isinstance(item.get("citation_id"), bool)
    }

    invalid_ids = [
        citation_id
        for citation_id in cited_ids
        if citation_id not in available_ids
    ]

    return {
        "valid": bool(cited_ids) and not invalid_ids,
        "cited_ids": cited_ids,
        "available_ids": sorted(available_ids),
        "invalid_ids": invalid_ids,
        "has_citations": bool(cited_ids),
    }


def validate_sql_evidence(
    answer: str,
    tool_result: dict,
) -> dict:
    """Validate that a SQL answer references executed SQL evidence."""

    evidence_id = tool_result.get("evidence_id")
    evidence_type = tool_result.get("evidence_type")

    has_evidence = (
        evidence_type == "sql"
        and isinstance(evidence_id, str)
        and bool(evidence_id.strip())
        and "error" not in tool_result
        and isinstance(tool_result.get("rows"), list)
    )

    referenced_ids = (
        re.findall(r"\bSQL-\d+\b", answer)
        if has_evidence
        else []
    )

    valid = (
        has_evidence
        and evidence_id in referenced_ids
    )

    return {
        "valid": valid,
        "evidence_type": evidence_type,
        "evidence_id": evidence_id,
        "referenced_ids": referenced_ids,
        "has_evidence": has_evidence,
    }


def validate_sql_answer_consistency(
    answer: str,
    tool_result: dict,
) -> dict:
    """Check whether a SQL answer matches its returned result.

    Designed for a single-row sales result containing a region
    and a revenue value.
    """

    if not isinstance(answer, str):
        return {
            "valid": False,
            "error": "Answer must be a string.",
        }

    if not isinstance(tool_result, dict):
        return {
            "valid": False,
            "error": "SQL tool result must be a dictionary.",
        }

    if "error" in tool_result:
        return {
            "valid": False,
            "error": "SQL tool execution failed.",
        }

    rows = tool_result.get("rows", [])
    if not rows or not isinstance(rows[0], dict):
        return {
            "valid": False,
            "error": "SQL result contains no usable row.",
        }

    row = rows[0]

    # Find the expected region field.
    region = row.get("region")

    # Find a revenue field in the returned row.
    revenue_key = next(
        (
            key for key in row
            if "revenue" in key.lower()
        ),
        None,
    )
    revenue = row.get(revenue_key) if revenue_key else None

    if not isinstance(region, str) or revenue is None:
        return {
            "valid": False,
            "error": "Expected region and revenue fields were not found.",
        }

    # Check whether the answer mentions the returned region.
    region_matches = bool(
        re.search(
            rf"\b{re.escape(region)}\b",
            answer,
            flags=re.IGNORECASE,
        )
    )

    # Normalize formatted numbers, e.g. 954000, 954,000, 954000.00.
    answer_numbers = re.findall(
        r"(?<!\w)-?\d[\d,]*(?:\.\d+)?",
        answer,
    )

    parsed_numbers = []
    for value in answer_numbers:
        try:
            parsed_numbers.append(float(value.replace(",", "")))
        except ValueError:
            continue

    try:
        expected_revenue = float(revenue)
        revenue_matches = any(
            value == expected_revenue
            for value in parsed_numbers
        )
    except (TypeError, ValueError):
        revenue_matches = False

    checks = {
        "region_matches": region_matches,
        "revenue_matches": revenue_matches,
    }

    return {
        "valid": all(checks.values()),
        "checks": checks,
        "expected_region": region,
        "expected_revenue": revenue,
        "evidence_id": tool_result.get("evidence_id"),
    }


def validate_agent_answer(
    answer: str,
    evidence_type: str,
    sql_tool_result: dict | None = None,
    source_map: list[dict] | None = None,
) -> dict:
    """Run the appropriate validation checks for an agent answer.

    evidence_type:
        "sql" = answer based on SQL query results
        "rag" = answer based on knowledge-base retrieval
    """

    if not isinstance(answer, str):
        return {
            "valid": False,
            "evidence_type": evidence_type,
            "error": "Answer must be a string.",
        }

    if evidence_type == "sql":
        if not isinstance(sql_tool_result, dict):
            return {
                "valid": False,
                "evidence_type": "sql",
                "error": "SQL tool result is missing.",
            }

        evidence_check = validate_sql_evidence(
            answer,
            sql_tool_result,
        )

        consistency_check = validate_sql_answer_consistency(
            answer,
            sql_tool_result,
        )

        overall_valid = (
            evidence_check["valid"]
            and consistency_check["valid"]
        )

        return {
            "valid": overall_valid,
            "evidence_type": "sql",
            "checks": {
                "evidence_reference": evidence_check,
                "result_consistency": consistency_check,
            },
        }

    if evidence_type == "rag":
        citation_check = validate_final_citations(
            answer,
            source_map or [],
        )

        return {
            "valid": citation_check["valid"],
            "evidence_type": "rag",
            "checks": {
                "citation_validation": citation_check,
            },
        }

    return {
        "valid": False,
        "evidence_type": evidence_type,
        "error": "Unsupported evidence type.",
    }
