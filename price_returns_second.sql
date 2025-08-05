WITH mid_price_stream AS (
  SELECT
    TIMESTAMP_TRUNC(event_timestamp, SECOND) AS event_sec,
    (level_1_ask_price + level_1_bid_price) / 2 AS mid_price,
    event_timestamp
  FROM
    `your_project.your_dataset.clob_data`
),

latest_mid_per_second AS (
  SELECT
    event_sec,
    mid_price,
    ROW_NUMBER() OVER (PARTITION BY event_sec ORDER BY event_timestamp DESC) AS rn
  FROM mid_price_stream
),

per_second_mid_price AS (
  SELECT
    event_sec,
    mid_price
  FROM latest_mid_per_second
  WHERE rn = 1
),

-- Fill forward the last known mid_price to all missing seconds
all_seconds_with_fill AS (
  SELECT
    ts AS event_sec,
    LAST_VALUE(mid_price IGNORE NULLS) OVER (
      ORDER BY ts
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS mid_price
  FROM (
    SELECT
      ts
    FROM
      UNNEST(GENERATE_TIMESTAMP_ARRAY(
        (SELECT MIN(event_sec) FROM per_second_mid_price),
        (SELECT MAX(event_sec) FROM per_second_mid_price),
        INTERVAL 1 SECOND
      )) AS ts
    LEFT JOIN per_second_mid_price USING (event_sec)
  )
),

returns_calculated AS (
  SELECT
    event_sec,
    mid_price,
    LAG(mid_price) OVER (ORDER BY event_sec) AS prev_mid_price
  FROM all_seconds_with_fill
)

SELECT
  event_sec,
  mid_price,
  prev_mid_price,
  SAFE_DIVIDE(mid_price - prev_mid_price, prev_mid_price) AS pct_return
FROM returns_calculated
WHERE prev_mid_price IS NOT NULL
ORDER BY event_sec;
