
import sqlite3
from pathlib import Path


DB_PATH = Path("data/agent_analytics.db")


def initialize_database() -> None:
    """Create a small, repeatable sales dataset for agent experiments."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS sales (
                sale_id INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                region TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL
            );

            DELETE FROM sales;

            INSERT INTO sales
                (sale_id, product, region, quantity, unit_price)
            VALUES
                (1, 'Laptop', 'South', 10, 65000),
                (2, 'Monitor', 'South', 20, 12000),
                (3, 'Laptop', 'North', 8, 67000),
                (4, 'Keyboard', 'West', 50, 1500),
                (5, 'Monitor', 'North', 15, 12500),
                (6, 'Keyboard', 'South', 40, 1600),
                (7, 'Laptop', 'West', 5, 64000),
                (8, 'Monitor', 'West', 12, 11800);
        """)


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized mock analytics database: {DB_PATH}")
