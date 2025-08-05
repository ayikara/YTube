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

-- Generate all seconds and fill forward the last known mid_price
all_seconds_with_fill AS (
  SELECT
    ts AS event_sec,
    LAST_VALUE(mid_price IGNORE NULLS) OVER (
      ORDER BY ts
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS mid_price
  FROM (
    SELECT
      ts,
      mid_price
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
    LAG(mid_price, 5) OVER (ORDER BY event_sec) AS mid_price_5s_ago
  FROM all_seconds_with_fill
)

SELECT
  event_sec,
  mid_price,
  mid_price_5s_ago,
  SAFE_DIVIDE(mid_price - mid_price_5s_ago, mid_price_5s_ago) AS pct_return_5s
FROM returns_calculated
WHERE mid_price_5s_ago IS NOT NULL
ORDER BY event_sec;

