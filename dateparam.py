from google.cloud import bigquery
from datetime import datetime

# Initialize BigQuery client
client = bigquery.Client()

# Define your query with named parameters
query = """
    SELECT *
    FROM `your_project.your_dataset.your_table`
    WHERE DATE(your_date_column) BETWEEN @start_date AND @end_date
"""

# Define the parameters
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("start_date", "DATE", "2024-03-10"),
        bigquery.ScalarQueryParameter("end_date", "DATE", "2024-03-20"),
    ]
)

# Run the query
query_job = client.query(query, job_config=job_config)

# Fetch the results
results = query_job.result()

# Print the results (as an example)
for row in results:
    print(row)
