# CME vs Hyperliquid Liquidity Comparison — Simple Pipeline
## Technical Design Document

| | |
|---|---|
| **Version** | 1.0 (draft for review) |
| **Date** | 2026-07-12 |
| **Scope** | A straight-line, notebook-executable pipeline. **Not** agentic — no LangGraph, no anomaly agent, no human-in-loop. |
| **Relationship to prior work** | This is the simpler predecessor to the agentic v2.2 design. It shares the normalization principles and the modular ingest boundary, but nothing else. |
| **Status** | Design only. No code until sign-off. |
| **v1.1 change** | Added §1.1, stating the normalization principle explicitly up front (it was already embedded in §3.4/§5; now surfaced as the organizing rule). |

---

## 1. Purpose

Compare **liquidity** between CME and Hyperliquid (HL) for a shared asset
(e.g. gold: CME `GC` vs a Trade.xyz per-ounce perpetual), using minute-level
order-book snapshots. Three liquidity families are produced, normalized so the
two venues are objectively comparable despite different contract sizes and
price increments:

1. **Book depth** (in USD notional) — bid, ask, and average.
2. **Bid-ask spread** — in price and in basis points.
3. **Cost to trade / slippage** — buy, sell, and average, for configurable
   trade sizes.

Plus a **CME↔HL book market-share** metric, at a chosen level or full-book.

All metrics are aggregated to minute / hour / trading-hour granularity, per day
and across days, and rendered as charts.

### 1.1 Normalization principle (the organizing rule)

Raw per-contract numbers are not comparable across venues, because the two
venues differ in **contract size** (CME GC = 100 troy oz/contract; HL gold = 1
oz/contract) and in **minimum price increment** (CME GC tick = $0.10/oz; HL
finer). Every metric in this design is built on one unifying move that removes
both differences:

> **`price × size × point_value → USD`, then express spread and cost as `bps of mid`.**
>
> - **`× point_value` (the contract multiplier) → USD** absorbs the
>   **contract-size** difference. A resting size becomes a dollar amount, so
>   "size" means the same economic thing on both venues.
> - **`bps of mid`** absorbs the **price-increment** difference. A spread or a
>   slippage expressed as a fraction of price is unit-free, so a $0.10 tick and
>   a finer HL tick are measured on the same scale.

Concretely, this rule instantiates as:

| Metric | Normalized form | Where |
|---|---|---|
| Depth | USD notional = `settlement × size × multiplier`, summed over levels | §3.4, §5.1 |
| Spread | `spread_bps = (ask − bid) / mid × 10,000` | §5.2 |
| Cost to trade | fill a fixed **USD** notional; `slippage_bps = (VWAP_fill − mid) / mid × 10,000` | §5.3 |
| Market share | ratio of USD depths (already normalized) | §5.4 |

`point_value` is the `cme_multiplier` / `hl_multiplier` from config (§4.1).
Because the rule is applied uniformly, contract-size and tick differences are
gone before any CME-vs-HL number is compared, and the same rule carries
unchanged to other assets (crude, etc.). Two residual caveats survive
normalization and are flagged where they arise: the **tick-size floor** on
spreads (§5.2) and **USDC settlement basis** if HL settles in USDC (§11).

---

## 2. Pipeline stages (straight-line)

```
1. Query        pull minute-level rows for [asset, date range] from the source
2. Forward-fill reconstruct the dense book from the sparse CLOB representation
3. Notional     value every level's size using previous-day settlement (fixed/day)
4. Metrics      depth, spread, cost-to-trade, market share (per minute)
5. Aggregate    mean + dispersion by minute / hour / trading_hour, per day & across days
6. Compare      CME vs HL side by side; market-share; normalized bps views
7. Visualize    charts at each granularity, intraday and across days
```

No branching, no cycles, no agent. Each stage is a pure function; the notebook
calls them in order. A run is fully determined by its config + inputs.

---

## 3. Data

### 3.1 Source schema (wide, one row per asset-minute)
One row per `(business_date, time_in_minute, asset)`. CME and HL for the same
asset **share the row**. Columns follow an identical suffix pattern with six
fields per level:

For CME levels `n = 1..10` and HL levels `n = 1..20`:

| Field pattern | Meaning |
|---|---|
| `cme_l{n}_bid_size`, `cme_l{n}_ask_size` | resting size (contracts) at level n |
| `cme_l{n}_bid_orders`, `cme_l{n}_ask_orders` | order count at level n |
| `cme_l{n}_bid_price`, `cme_l{n}_ask_price` | price at level n |
| `hl_l{n}_...` (same six) | Hyperliquid equivalents, n = 1..20 |

Row-level fields:

| Field | Meaning |
|---|---|
| `business_date` | trading date |
| `time_in_minute` | minute timestamp |
| `cme_asset`, `hl_asset` | asset identifiers per venue |
| `previous_day_settlement_price` | CME prior-day settlement; the **constant valuation price for the whole day** (§3.4) |
| `trading_hour` | session tag: `RTH` (regular), `ATH` (asian), `ETH` (europe) — **already stamped on the row**; the pipeline groups by it directly |

Prices are in the **same per-unit terms on both venues** (e.g. per troy ounce),
confirmed — so notional and bps comparisons are apples-to-apples once the
multiplier is applied.

### 3.2 Sparse representation → the central data problem
The book is a **sparse CLOB**: a row/field is written only when that state
*changes*. A minute with no change has **no entry**, and either the CME side,
the HL side, or both may be missing in any given minute. The dense book must be
reconstructed by forward-fill (§3.3) before any metric is computed.

### 3.3 Forward-fill rules (precise)
Fill the last known state forward across empty minutes, **per venue
independently** (CME and HL are filled separately, since their trading calendars
differ). Rules:

1. **Normal gaps:** if a venue has no entry at minute *t*, carry forward that
   venue's most recent prior state — even across multi-minute gaps (e.g. a
   missing 5:00am fills from 3:59pm). This extreme case is not expected for
   liquid products (ES, CL, GC) but is handled.
2. **CME daily maintenance break (16:00–16:59 CST):** **do not fill.** Minutes
   16:00–16:59 CST are left **empty/NULL** on the CME side. HL is unaffected.
3. **CME closed (weekends, holidays):** CME order-book state is **NULL — no
   forward-fill.** HL trades 24/7 and is filled normally through these periods.
4. **HL:** filled continuously; HL has no scheduled closures.

The output of this stage carries an explicit per-venue flag per minute:
`cme_state ∈ {live, filled, closed_null}` and `hl_state ∈ {live, filled}`, so
downstream metrics and charts can distinguish a real quote from a
forward-filled one from a legitimate closure. Metrics on a `closed_null` CME
minute are NULL (not zero, not carried).

> **Timezone note:** the 16:00–16:59 CST rule and the weekend/holiday calendar
> are evaluated in the venue's session timezone. The `trading_hour` column
> already encodes session membership; the closure rules are applied on top using
> a small calendar in config (holiday list + CST maintenance window).

### 3.4 Notional valuation — constant daily price
All size→USD conversions for a given `business_date` use that date's
`previous_day_settlement_price` as a **single fixed valuation price for the
whole day**. This is deliberate: holding the valuation price constant means
depth-in-USD changes reflect **pure size/liquidity changes**, not intraday price
drift. Formula per level, per venue:

```
level_notional_usd = previous_day_settlement_price × size_contracts × multiplier
```

- `multiplier` = `cme_multiplier` (e.g. 100 for GC) or `hl_multiplier` (e.g. 1).
- Same settlement price used for both venues on that date (they track the same
  underlying), so the comparison isn't contaminated by different reference
  prices.
- One settlement per asset per business date, taken from the row.

---

## 4. Configuration (JSON)

### 4.1 Product spec
```json
{
  "assets": {
    "GOLD": {
      "cme_symbol": "GC", "hl_symbol": "XYZ:GOLD-USD",
      "cme_multiplier": 100, "hl_multiplier": 1,
      "cme_tick_size": 0.10, "hl_tick_size": 0.01,
      "unit": "troy_ounce", "cme_levels": 10, "hl_levels": 20
    }
  }
}
```
Everything venue/asset-specific is read from here; no calculation hardcodes a
multiplier, tick, or level count. Adding crude or another asset is a config
entry.

### 4.2 Run parameters (user-set in the notebook)
| Parameter | Meaning | Example |
|---|---|---|
| `asset` | which asset to run | `"GOLD"` |
| `date_range` | business dates to include | `["2026-06-01", "2026-06-05"]` |
| `depth_level_n` | level cutoff for depth & market-share (1..10) | `5` |
| `use_full_book` | if true, market-share uses HL(1–20)/(CME(1–10)+HL(1–20)) instead of level-n | `false` |
| `cost_trade_sizes` | notional sweep for slippage, config-driven off observed CME sizes | see §6.3 |
| `agg_granularity` | `minute` \| `hour` \| `trading_hour` | `"trading_hour"` |

---

## 5. Metric definitions

All metrics computed **per minute** first (on the forward-filled, notional-
valued book), then aggregated (§7). CME uses levels 1..10; HL uses 1..20; the
depth/market-share cutoff is governed by `depth_level_n` or `use_full_book`.

### 5.1 Book depth (USD notional)
For a venue and a level cutoff `N`:
```
bid_depth_usd(N) = Σ_{i=1..N} settlement × bid_size_i × multiplier
ask_depth_usd(N) = Σ_{i=1..N} settlement × ask_size_i × multiplier
avg_depth_usd(N) = (bid_depth_usd + ask_depth_usd) / 2   # simple mean of sides
```
Produced for CME and HL. `avg` is the simple (unweighted) mean of the two sides,
per your confirmation.

### 5.2 Bid-ask spread
```
spread_price = ask_l1_price − bid_l1_price
mid          = (ask_l1_price + bid_l1_price) / 2
spread_bps   = spread_price / mid × 10,000
```
Both are reported. **bps is the cross-venue-comparable number**; `spread_price`
is shown alongside. The **tick size is reported next to the spread** so a
"tighter" HL spread that is partly just a finer minimum increment isn't
misread as deeper competition (tick floor: CME GC cannot quote inside $0.10/oz).

### 5.3 Cost to trade / slippage
Walk the book to fill a fixed **USD notional** `Q`, volume-weighted average
fill vs mid, per side:
```
buy_slippage_bps(Q)  = (VWAP_ask_fill(Q)  − mid) / mid × 10,000
sell_slippage_bps(Q) = (mid − VWAP_bid_fill(Q)) / mid × 10,000
avg_slippage_bps(Q)  = (buy_slippage_bps + sell_slippage_bps) / 2
```
Filling uses `size × multiplier × settlement` per level as the fillable notional
at that level's price, walking down levels until `Q` is met. If the book (to its
available levels) cannot fill `Q`, the result is NULL for that minute/venue and
flagged as "insufficient depth" (informative — it means the size exceeds
displayed liquidity). Sizes come from config (§6.3).

### 5.4 CME↔HL book market share
Two modes, chosen by parameter:

**Level-n mode** (`depth_level_n = N`, `use_full_book = false`) — like-for-like:
```
market_share_hl(N) = HL_depth_usd(N) / ( CME_depth_usd(N) + HL_depth_usd(N) )
```
Computed per side (bid, ask) and on the average. Uses the same `N` on both
venues so it's a fair comparison within a matched level count.

**Full-book mode** (`use_full_book = true`) — as specified, accepting that HL's
20 levels reach further from mid than CME's 10:
```
market_share_hl_full = HL_depth_usd(1..20) / ( CME_depth_usd(1..10) + HL_depth_usd(1..20) )
```

> **Interpretation caveat (surfaced, not hidden):** full-book market share
> structurally favors HL because its 20 levels span a wider price range than
> CME's 10. The like-for-like level-n mode is the fairer comparison; full-book
> answers "of all displayed resting notional, what fraction sits on HL." Both
> are legitimate; the charts label which is shown.

---

## 6. Cost-to-trade sizing (config-driven)

Per your direction, slippage trade sizes are derived from **observed CME level
sizes**, not fixed dollar amounts, so they scale with the contract and typical
book. Default sweep of three (all configurable):

| Size label | Definition | Rationale |
|---|---|---|
| `L1` | CME level-1 notional | cost to clear the top level |
| `1.5×L1` | 1.5 × CME level-1 notional | just beyond top-of-book |
| `L1+L2` | CME level-1 + level-2 notional | cost to sweep two levels |

`cost_trade_sizes` in config lets the user override these (absolute USD values
or level-derived expressions). The same USD sizes are applied to both venues so
the comparison is on equal footing.

---

## 7. Aggregation

Per-minute metrics are aggregated over the chosen window. `trading_hour` grouping
uses the source column directly (RTH/ATH/ETH); `hour` floors `time_in_minute`;
`minute` is the raw series.

- **Headline statistic:** mean of the per-minute metric over the window (minutes
  are uniform, so this is time-weighted).
- **Dispersion band:** min/max (and p10/p90 where useful) carried alongside the
  mean for the plots, so variability is visible.
- **Two axes of aggregation:**
  - *Within a day:* by minute, by hour, by trading_hour.
  - *Across days:* one point per day (total-day mean) and per (day × trading_hour),
    to show day-over-day evolution and session patterns.
- Forward-filled minutes are included in aggregation; `closed_null` CME minutes
  are excluded from CME aggregates (not counted as zero).

---

## 8. Comparison outputs

For each metric, the pipeline emits a tidy comparison table and the matching
chart data:

| Comparison | Content |
|---|---|
| Depth | CME vs HL `avg_depth_usd(N)` (and bid/ask), per granularity |
| Spread | CME vs HL `spread_bps` (and price), tick size annotated |
| Cost to trade | CME vs HL `avg_slippage_bps(Q)` for each size Q, buy/sell split available |
| Market share | HL share (level-n and/or full-book), per side and average |

All in normalized units (USD notional; bps), so contract-size and tick
differences are absorbed.

---

## 9. Visualization

Charts at each requested granularity. Proposed set (Plotly, notebook-inline):

1. **Depth over time** — CME vs HL `avg_depth_usd`, intraday (minute/hour), with
   dispersion band; shaded CME-closed spans shown as gaps (not zeros).
2. **Spread over time** — CME vs HL `spread_bps`, with tick-floor reference lines.
3. **Cost-to-trade** — slippage bps per venue, one series per trade size; and a
   small-multiples view across the three sizes.
4. **Market share** — HL share over time (level-n and full-book as separate
   traces), with a 50% reference line.
5. **Across-days** — per-day (and per day × trading_hour) bars/lines for each
   metric, to show day-over-day and session patterns.
6. **Trading-hour breakdown** — RTH vs ATH vs ETH grouped bars per metric.

Every chart labels units (USD / bps), the level cutoff `N` or full-book, and
whether values are live or include forward-filled minutes.

---

## 10. Architecture

### 10.1 Modular ingest (retained from prior design)
A single `DataSource` interface with `load_book(asset, date_range)` returning
the wide minute-level frame. Implementations:
- `MockSource` — synthetic wide frame for development/testing now.
- `SqlSource` / `BigQuerySource` — stub; the real query swaps in as a one-class
  change with no downstream impact, as long as it returns the §3.1 schema.

This is the *only* piece of "framework" carried over. There is **no LangGraph,
no agent, no orchestration layer** — the notebook calls stage functions directly.

### 10.2 Module layout
```
liq_compare/
  config/
    product_specs.json        assets, multipliers, ticks, level counts
    calendar.json             CME holidays + 16:00-16:59 CST maintenance window
  ingest/
    base.py                   DataSource interface
    mock.py                   MockSource (live now)
    sql.py                    SqlSource / BigQuerySource stub (retrofit point)
  fill.py                     sparse -> dense forward-fill with closure rules (sec 3.3)
  notional.py                 constant-daily-price valuation (sec 3.4)
  metrics.py                  depth, spread, cost-to-trade, market share (sec 5)
  aggregate.py                minute/hour/trading_hour, per-day & across-days (sec 7)
  compare.py                  CME vs HL comparison tables (sec 8)
  viz.py                      Plotly charts (sec 9)
liquidity_comparison.ipynb    straight-line notebook + parameter UI
```

### 10.3 Notebook flow
Config/params cell → query → fill → notional → metrics → aggregate → compare →
charts. A small ipywidgets parameter panel (asset, date range, `depth_level_n`,
`use_full_book`, cost sizes, granularity) re-runs the straight-line flow. No
review panels, no interrupts.

---

## 11. Edge cases

| Edge case | Handling |
|---|---|
| Sparse minute, one or both venues missing | Forward-fill per venue (sec 3.3) |
| CME 16:00–16:59 CST maintenance | Left NULL, not filled |
| CME weekend/holiday | NULL, not filled; HL continues |
| Multi-minute gap (e.g. missing 5:00am) | Fill from last known prior state (extreme case, handled) |
| Book cannot fill requested cost-to-trade size | Slippage = NULL, flagged "insufficient depth" |
| Both sides zero size (imbalance/derived ratios) | Guarded; NULL not divide-by-zero |
| Fewer than N levels populated after fill | Depth sums populated levels; if a level is genuinely absent it contributes 0 size, not NULL, *provided the venue is live* |
| Full-book market share favoring HL | Documented interpretation caveat (sec 5.4); like-for-like mode offered |
| Tick-size floor on spreads | Tick reported alongside spread (sec 5.2) |
| Settlement missing for a date | Day flagged; notional not computed for that date (fail loud, not silent) |
| Mixed timezone in closure vs trading_hour | trading_hour from column; closures evaluated in session TZ from calendar |
| USDC settlement basis (if HL settles in USDC) | Noted as a caveat: a USDC/USD deviation would appear in USD-denominated metrics as apparent liquidity difference; out of scope to correct here |

---

## 12. What this design deliberately excludes
- No LangGraph, no anomaly-investigation agent, no human-in-loop.
- No Mode-A/Mode-B basis modes, no carry/funding analysis, no correlation
  (those belong to the agentic v2.2 track).
- No BigQuery input/output persistence (mock source now; SQL/BQ stub for later).
- No roll/continuous-contract handling (single settlement per asset-day is used
  as given).

---

## 13. Open items for confirmation
1. **Cost-to-trade default sizes** — confirm `L1`, `1.5×L1`, `L1+L2` (derived
   from CME level-1/2 notional) as the default sweep, user-overridable.
2. **Market-share sides** — confirm it's reported per side (bid, ask) *and* on
   the average, in both level-n and full-book modes.
3. **Dispersion band** — confirm min/max (with optional p10/p90) is the desired
   band around the mean in charts.
4. **`hl_asset` vs `cme_asset`** — confirm these are just per-venue identifiers
   on the shared row and there is exactly one asset pairing per row (no
   many-to-one).
