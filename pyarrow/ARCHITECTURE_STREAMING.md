# Architecture Addendum: Indexing, Sequence Gaps & Streaming Strategy

---

## 1. Do We Need to Index the CLOB Parquet File?

### Short answer: No. Parquet's built-in mechanisms replace traditional indexing.

### Why traditional indexing doesn't apply to Parquet:

Parquet is not a database — it has no B-tree, no hash index, no random-access
by key. But it has three features that serve the same purpose for our access
pattern:

| Parquet Feature | What It Does | Our Benefit |
|-----------------|-------------|-------------|
| **Partition pruning** | `date=.../product=ES/` directories | PyArrow never opens files for other products or dates. Zero I/O wasted. |
| **Row group statistics** | Min/max of seq_num and timestamp_ms stored in footer metadata | If we query a seq_num range, PyArrow skips entire row groups whose max < our min. |
| **Column pruning** | Only reads columns requested | If Phase A only needs seq_num + bid_px1 + ask_px1, the other 38 columns stay on disk. |

### What we SHOULD do instead of indexing:

**At write time (when creating the CLOB Parquet files):**

```
 Action                              Why
 ──────────────────────────────────  ─────────────────────────────────────
 Sort by seq_num within each file    Enables row group statistics to be
                                     tight (non-overlapping ranges).
                                     Makes sequential scan cache-friendly.

 Row group size = 500K–1M rows       Sweet spot: large enough for
                                     vectorized reads, small enough that
                                     row group min/max stats skip
                                     irrelevant groups.

 Write statistics = ON               Default in PyArrow, but verify.
                                     This is what enables "index-like"
                                     row group skipping.

 Compression = ZSTD level 3          Best decompression speed per byte
                                     for numeric columnar data.
```

**This means:** when we read the file sorted by seq_num with statistics
enabled, PyArrow already knows "row group 7 has seq_num range
[5,000,001 .. 6,000,000]" and can skip it if we don't need that range.

### When would we actually need an index?

Only if we needed **random point lookups** like "give me the book state at
seq_num = 4,823,117". But we don't — our access pattern is always a
**full sequential scan** of one product's entire day. Partition pruning
handles the filtering (right product, right date), and then we read
everything within that partition in order.

**Verdict: No indexing needed. Sort by seq_num at write time + partition
by date/product is sufficient.**

---

## 2. Sequence Number Gaps

### The concern:

seq_num is shared across order book updates AND transactions, but also
covers order cancels, revisions, and other message types that may not
appear in either file. So the sequence might look like:

```
 seq_num   Source          Event Type
 ───────   ──────────────  ──────────────────
 1001      CLOB            Book update (new order)
 1002      (neither)       Order cancel — not in our files
 1003      (neither)       Order revision — not in our files
 1004      CLOB            Book update (reflects cancel+revision)
 1005      Transaction     Trade
 1006      CLOB            Book update (post-trade)
 1007      (neither)       Heartbeat / admin
 1008      Transaction     Trade
```

### Impact on our pipeline:

**Interleave merge: NO PROBLEM.** Our merge-by-seq_num algorithm simply
interleaves whatever records exist in both files. Gaps are invisible —
we never look for a specific seq_num, we just maintain relative order.

```
 CLOB stream:          1001, 1004, 1006, ...
 Transaction stream:   1005, 1008, ...
 Interleaved:          1001(C), 1004(C), 1005(T), 1006(C), 1008(T), ...
                       ✓ Correct order preserved despite gaps
```

**Sweep detection: NO PROBLEM.** We walk the interleaved stream and
maintain a running best_bid / best_ask. When we hit trade 1005, the
running state reflects CLOB update 1004 (the last book state before
this trade). The gap at 1002–1003 is irrelevant because update 1004
already incorporates the cancel and revision.

**Key insight:** The CLOB file captures the *resulting book state* after
each update, not the individual order actions. So even though seq 1002
(cancel) and 1003 (revision) are missing, seq 1004 reflects the book
AFTER those events. The book state is self-contained.

### One edge case to handle:

**A trade arrives BEFORE the first CLOB update of the day.**
If the first record in the interleaved stream is a transaction, we have
no book state yet → running_best_bid and running_best_ask are NaN.

```
 Mitigation:
   - Initialize running_best_bid = NaN, running_best_ask = NaN
   - If a transaction fires with NaN book state:
       aggressor = UNKNOWN
       swept = False  (cannot determine without book state)
   - These are typically pre-market or opening-auction trades
     and are rare. Flag them for review.
```

### No code changes needed for gaps. The algorithm is inherently gap-tolerant.

---

## 3. PyArrow Streaming Strategy — Avoiding the Memory Bottleneck

### The current approach (v3 architecture) and its weakness:

```
 Current:  pq.read_table("clob/.../product=ES/")  → single Table in RAM
           .to_numpy() per column → 82.8M-element arrays
           Peak: ~17 GB for one liquid product's CLOB
```

This works on 128 GB, but it's not *streaming* — it materializes the
entire product's day into one PyArrow Table before processing. If a
product has an unusually large day (50M+ updates), or if we later want
to process multiple products in parallel, we'll hit the wall.

### The streaming alternative: Process in hourly chunks

The key insight is that **our pipeline's output granularity is hourly**
(Step 3 percentiles are per-hour). This means we can process one hour
at a time and only keep that hour's data in memory.

But there's a catch: **event windows cross hour boundaries.** An event
at ms 3,599,997 (last 3ms of an hour) has a +8 offset that reaches into
the next hour. We need a small overlap buffer.

### Proposed streaming architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STREAMING MODEL: Hourly Chunks with Overlap                       │
│                                                                     │
│  Instead of loading 82.8M rows, load ~3.6M rows at a time          │
│  (1 hour = 3,600,000 ms) with an overlap buffer of ±10ms           │
│  at boundaries.                                                     │
│                                                                     │
│  Memory per chunk: 3.6M × 43 cols × 8B ≈ 1.2 GB (vs 17 GB)       │
│                                                                     │
│  Timeline:                                                          │
│  Hour 17:  ms [0 .............. 3,599,999] + buffer [3,600,000..+9]│
│  Hour 18:  ms [3,599,991...-9] [3,600,000 ........... 7,199,999]  │
│  ...                                                                │
│                                                                     │
│  Each chunk:                                                        │
│    1. Stream CLOB rows for this hour's timestamp range              │
│    2. Stream transaction rows for same range                        │
│    3. Build ms timeline (3.6M slots)                                │
│    4. Forward-fill (carry last known state from previous chunk)     │
│    5. Compute metrics, extract windows, compute percentiles         │
│    6. Accumulate Step 2 windows to disk                             │
│    7. Free arrays, move to next hour                                │
└─────────────────────────────────────────────────────────────────────┘
```

### PyArrow streaming implementation:

```python
# APPROACH A: Row-group-level streaming with predicate pushdown
#
# PyArrow can push timestamp predicates down to row groups,
# reading only the row groups that overlap with our hour window.

parquet_file = pq.ParquetFile("clob/.../product=ES/part-00000.parquet")

for batch in parquet_file.iter_batches(
    batch_size=500_000,
    columns=["seq_num", "timestamp_ms", "bid_px1", "ask_px1", ...],
    # No native row-level filter in iter_batches, but row group
    # statistics will skip groups outside our range if file is
    # sorted by timestamp_ms
):
    # Process each 500K-row batch
    ts = batch.column("timestamp_ms").to_numpy()
    # Filter to current hour window
    mask = (ts >= hour_start_ms) & (ts < hour_end_ms + 10)
    # ... process
```

```python
# APPROACH B: PyArrow Dataset API with filter pushdown
#
# The Dataset API provides true predicate pushdown at the row-group
# level, which is cleaner for hour-based filtering.

import pyarrow.dataset as ds

dataset = ds.dataset(
    "clob/",
    format="parquet",
    partitioning=ds.partitioning(
        pa.schema([("date", pa.date32()), ("product", pa.string())]),
        flavor="hive"
    ),
)

# This pushes the filter to row-group statistics
hour_start_ms = ...
hour_end_ms = ...

scanner = dataset.scanner(
    columns=["seq_num", "timestamp_ms", "bid_px1", "ask_px1", ...],
    filter=(
        (ds.field("product") == "ES") &
        (ds.field("date") == "2024-01-15") &
        (ds.field("timestamp_ms") >= hour_start_ms) &
        (ds.field("timestamp_ms") < hour_end_ms + OVERLAP_MS)
    ),
)

for batch in scanner.to_batches():
    # Each batch is a RecordBatch, already filtered
    # Process directly — no full materialization
    ...
```

### Approach B is preferred because:

| Feature | Approach A (ParquetFile) | Approach B (Dataset API) |
|---------|--------------------------|--------------------------|
| Partition pruning | Manual path construction | Automatic from filter |
| Row-group skipping | Only via sorted stats | Explicit filter pushdown |
| Multi-file partitions | Must loop files manually | Handles transparently |
| Predicate on timestamp | Post-read filtering | Pre-read row-group skip |
| API complexity | Lower level | Higher level, cleaner |

### The hybrid strategy — two tiers:

Not all phases benefit equally from streaming. Here's the optimal split:

```
 Phase          Strategy             Reasoning
 ────────────── ──────────────────── ──────────────────────────────────────
 A3. Interleave FULL LOAD            Interleaving by seq_num needs the
 A4. Sweeps                          entire day's data in sequence order.
                                     Cannot chunk by hour because seq_num
                                     doesn't align with hour boundaries.
                                     BUT: we only need 3 columns
                                     (seq_num, bid_px1, ask_px1) + price
                                     for sweep detection.

                                     3 CLOB cols × 50M × 8B = 1.2 GB
                                     3 TXN cols  ×  2M × 8B = 0.05 GB
                                     Total: ~1.3 GB ← very manageable

 B. Timeline    STREAM BY HOUR       Forward-fill only needs carryover
 C. Metrics                          state from previous hour (a single
 D. Windows                          row of 40 book values).
                                     Load 3.6M rows per hour chunk.
                                     Peak: ~1.2 GB per chunk.

 E. Percentiles IN-MEMORY            Operates on Step 2 windows, which
                                     are already small.
```

### Revised memory profile with streaming:

```
 Component                                    Memory
 ─────────────────────────────────────────── ────────
 Phase A: seq_num + BBO arrays (full day)     1.3 GB
 Phase A output: aggressor + swept arrays     0.03 GB
 Phase B–D: one hour chunk (3.6M × all cols)  1.2 GB
 Phase B–D: metric arrays for one hour        0.4 GB
 Phase B–D: event windows accumulator         0.1 GB
 Forward-fill carryover state                 0.0003 GB
 Working buffers                              0.5 GB
 ────────────────────────────────────────────────────
 Peak per product:                            ~3.5 GB  (was 32 GB)
```

**This is a 9× reduction in peak memory.** With 3.5 GB per product,
we could even process 4–8 products in parallel on 128 GB if needed.

### The carryover mechanism for forward-fill across hours:

```
 End of hour H processing:
   carryover_state = {
       "bid_px1": last_known_bid_px1,    # scalar
       "ask_px1": last_known_ask_px1,    # scalar
       "bid_sz1": last_known_bid_sz1,
       ...all 40 book columns...         # 40 scalars = 320 bytes
   }

 Start of hour H+1:
   Initialize forward-fill with carryover_state
   If the first ms of hour H+1 has no CLOB update,
   it inherits the book state from end of hour H.
```

### How the Dataset scanner handles multi-file partitions:

If a product's daily CLOB data is split across multiple Parquet files
within the partition directory:

```
clob/date=2024-01-15/product=ES/
  part-00000.parquet    (rows with timestamp 17:00–20:00)
  part-00001.parquet    (rows with timestamp 20:00–00:00)
  part-00002.parquet    (rows with timestamp 00:00–08:00)
  part-00003.parquet    (rows with timestamp 08:00–16:00)
```

The Dataset API scanner with a timestamp filter like
`timestamp_ms >= hour_08_start AND timestamp_ms < hour_09_end`
will:
1. Read metadata from all 4 files (tiny: just footers)
2. Check row group statistics against the filter
3. Only read row groups from part-00003.parquet that overlap with 08:00–09:00
4. Skip part-00000, part-00001, part-00002 entirely

This is why sorting by timestamp within each file and writing
statistics is critical — it makes the hour-based streaming efficient.

---

## 4. Revised Pipeline Flow (Streaming)

```
for each product (sequential):

  ┌────────────────────────────────────────────────────────────────┐
  │  PHASE A: Lightweight full-day scan (3 columns only)          │
  │                                                                │
  │  A1. Stream entire day's CLOB via Dataset scanner              │
  │      Columns: ONLY [seq_num, bid_px1, ask_px1]                │
  │      → 3 columns × 50M rows × 8 bytes = 1.2 GB               │
  │                                                                │
  │  A2. Stream entire day's transactions                          │
  │      Columns: [seq_num, price, qty]                            │
  │      → 0.05 GB                                                 │
  │                                                                │
  │  A3. Interleave by seq_num (Numba)                             │
  │  A4. Walk interleaved → aggressor_side[], swept_flag[]         │
  │                                                                │
  │  Output kept in memory:                                        │
  │    trade_seq_nums[2M]  int64                                   │
  │    trade_aggressor[2M] int8                                    │
  │    trade_swept[2M]     int8                                    │
  │    trade_ms_index[2M]  int32  (precompute ms bucket per trade) │
  │  Total: ~20 MB                                                 │
  │                                                                │
  │  Free the 1.2 GB CLOB BBO arrays (no longer needed)           │
  └────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  PHASE B–D: Stream by hour (23 iterations)                    │
  │                                                                │
  │  For hour_index in 0..22:                                      │
  │    hour_cst = (17 + hour_index) % 24                           │
  │    hour_start_ms = hour_index * 3_600_000                      │
  │    hour_end_ms   = hour_start_ms + 3_600_000                   │
  │    overlap_start = max(0, hour_start_ms - 10)                  │
  │    overlap_end   = min(82_800_000, hour_end_ms + 10)           │
  │                                                                │
  │    B1. Stream CLOB for [overlap_start, overlap_end) via        │
  │        Dataset scanner with timestamp filter                   │
  │        ALL 43 columns this time (need full book depth)         │
  │        → ~3.6M rows × 43 cols × 8B ≈ 1.2 GB                  │
  │                                                                │
  │    B2. Extract transactions for this hour from precomputed     │
  │        trade_ms_index (simple mask, already in memory)         │
  │                                                                │
  │    B3. Build 3.6M-slot ms timeline for this hour               │
  │        - Map CLOB updates to ms slots (last per ms)            │
  │        - Forward-fill from carryover_state                     │
  │        - Bucket transactions into ms slots                     │
  │                                                                │
  │    B4. Compute all 11 metrics (Numba parallel)                 │
  │                                                                │
  │    B5. Find events (ms with transactions > 0)                  │
  │        Gather 14-offset windows                                │
  │        Handle boundary: events in last 8ms of hour need        │
  │        metrics from overlap buffer (next hour's first 8ms)     │
  │                                                                │
  │    B6. Compute hourly percentiles for this hour                │
  │                                                                │
  │    B7. Append Step 2 windows to output buffer                  │
  │        Update carryover_state for next hour                    │
  │        Free hour arrays                                        │
  │                                                                │
  │  End hour loop                                                 │
  └────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  WRITE OUTPUTS                                                 │
  │                                                                │
  │  Write Step 2 Parquet (accumulated windows)                    │
  │  Write Step 3 Parquet (23 hours × 14 offsets × 11 metrics)    │
  │  Free all product arrays                                       │
  │  gc.collect()                                                  │
  └────────────────────────────────────────────────────────────────┘
```

---

## 5. Handling Window Boundaries at Hour Edges

The trickiest part of hourly streaming: an event at the end of hour H
needs metric values from the start of hour H+1 (offsets +1 through +8).

### Solution: Overlap buffer + deferred window completion

```
 Option A: Overlap buffer (simple, recommended)
 ───────────────────────────────────────────────
 When loading hour H, also load the first 10ms of hour H+1.
 The overlap is tiny (10 rows out of 3.6M).
 Events near the hour boundary can look ahead into the buffer.

 The Dataset scanner filter:
   timestamp_ms >= hour_start - 5ms     (for events in first 5ms
                                         needing backward offsets
                                         from previous hour)
   timestamp_ms <  hour_end + 10ms      (for events in last 8ms
                                         needing forward offsets)

 Since we process hours sequentially, the -5ms overlap is handled
 by the carryover from the previous hour's last few ms.
 The +10ms lookahead is loaded from the next hour's partition.

 Worst case extra data: 10 rows × 43 cols × 8B = 3,440 bytes.
 Negligible.
```

### Forward-fill at hour boundaries:

```
 Hour H ends:
   Save carryover_state = book values at ms 3,599,999

 Hour H+1 starts:
   If ms slot 0 has no CLOB update:
     Fill ms slot 0 with carryover_state
   Forward-fill continues from there normally
```

---

## 6. Summary: Why Streaming Is Worth the Complexity

```
 Metric              Full Load (v3)    Streaming (v4)
 ──────────────────  ────────────────  ────────────────
 Peak RAM / product  ~32 GB            ~3.5 GB
 Products in 128 GB  3 (sequential)    30+ (parallel possible)
 Disk I/O per hour   Read entire day   Read 1/23 of data
 Forward-fill scope  82.8M slots       3.6M slots
 Complexity          Simple            Moderate (hour boundaries)
 Robustness          OOM risk on       Very safe on 128 GB
                     extreme products
```

The streaming approach trades a moderate increase in code complexity
(hour-boundary handling, carryover state) for a dramatic improvement in
memory efficiency and robustness. It also opens the door to future
parallelism (processing 4+ products simultaneously).

---

## 7. Updated Recommendation

Proceed with the **hybrid streaming architecture**:
- Phase A: full-day scan but only 3 columns (1.3 GB) for sweep detection
- Phases B–D: hourly streaming via PyArrow Dataset scanner (~1.2 GB/chunk)
- Phase E: in-memory percentiles (trivially small)

This keeps the Numba kernel design identical — they still operate on
flat NumPy arrays. Only the *orchestration* changes to feed them
hour-sized chunks instead of full-day arrays.
