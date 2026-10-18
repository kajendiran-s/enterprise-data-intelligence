
import re
import sqlite3

from src.agent.sql_database import DB_PATH


ALLOWED_TABLES = {"sales"}
MAX_ROWS = 100


def run_readonly_sql(query: str) -> dict:
    """Execute a restricted, read-only SQL query against mock sales data."""

    normalized = query.strip()

    # Require a single SELECT statement.
    if not re.match(r"(?is)^SELECT\b", normalized):
        return {"error": "Only SELECT queries are permitted."}

    # Reject statement chaining and SQL comments.
    if ";" in normalized.rstrip(";"):
        return {"error": "Multiple SQL statements are not permitted."}

    if re.search(r"--|/\*|\*/", normalized):
        return {"error": "SQL comments are not permitted."}

    # Keep the tool scoped to the sales table.
    referenced_tables = {
        name.lower()
        for name in re.findall(
            r"(?i)\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            normalized,
        )
    }

    if not referenced_tables:
        return {"error": "Query must read from the sales table."}

    if not referenced_tables.issubset(ALLOWED_TABLES):
        return {
            "error": "Only the sales table is available to this tool."
        }

    try:
        # SQLite enforces read-only access at the connection level.
        database_uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"

        with sqlite3.connect(database_uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.set_authorizer(
                lambda action, arg1, arg2, db_name, trigger:
                    sqlite3.SQLITE_OK
                    if action in {
                        sqlite3.SQLITE_SELECT,
                        sqlite3.SQLITE_READ,
                        sqlite3.SQLITE_FUNCTION,
                    }
                    else sqlite3.SQLITE_DENY
            )

            cursor = connection.execute(normalized)
            columns = [item[0] for item in cursor.description or []]
            rows = [dict(row) for row in cursor.fetchmany(MAX_ROWS + 1)]

            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]

            return {
                    "evidence_type": "sql",
                    "evidence_id": "SQL-001",
                    "query": normalized,
                    "database": str(DB_PATH),
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": truncated,
                }

    except sqlite3.Error as error:
        return {"error": f"SQL execution failed: {error}"}
