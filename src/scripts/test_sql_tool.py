
from pprint import pprint

from src.agent.sql_tool import run_readonly_sql
from src.agent.sql_database import initialize_database


initialize_database()

query = """
SELECT
    region,
    SUM(quantity * unit_price) AS revenue
FROM sales
GROUP BY region
ORDER BY revenue DESC
"""

result = run_readonly_sql(query)
pprint(result)
