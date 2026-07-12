# CME vs Hyperliquid Derivatives Comparison
## Requirements, Architecture & Technical Specification — v2.2 (Draft)

| | |
|---|---|
| **Version** | 2.2 (draft for review) |
| **Date** | 2026-07-11 |
| **Supersedes** | v1.0, v2.0, v2.1 |
| **Status** | Design-only. No code to be written from this version until sign-off. |
| **Resolved in 2.1** | Agent autonomy default accepted (§7.5); CME settlement table confirmed available (§4.1, §5.5); data-quality guards moved in-scope to build now against mock data (§2, §11.3–11.4). |
| **Resolved in 2.2** | Synthetic schema is canonical for now, live-BigQuery reconciliation deferred (§4.1); HL/CME contract semantics fixed — SP-USDC notional `size×price` (pv 1), ES `50×price`, both track SPX at the same level (§6.4); stakeholder findings-acceptance criteria defined, price/return **correlation promoted to a first-class deliverable** (§6.6, §9); Mode A carry treatment via settlement deferred to the with-spot redo (§5.5); BigQuery **output** persistence in scope, raw **input** load out of scope (§2, §7.3, §11.4); all notional/contract calcs parametrized for CL/GC extension (§6.4, §7.8). |
| **Built so far** | ES / MES vs SP-USDC pipeline on synthetic mock data (v1 scope). v2 introduces material design changes not yet implemented — see §0. |

---

## 0. What changed from v1 (change log)

v2 incorporates a design review. The following are **new or materially changed**
and are **not yet built**:

1. **Two basis-analysis modes** (§5.5): a lean *without-spot* mode runnable with
   today's data, and a *with-spot* mode (the planned redo) that carries the
   carry attribution, settlement anchoring, and funding-vs-carry analysis once
   spot/index data is available.
2. **Genuine anomaly-investigation agent** (§7.5): replaces the v1 "fully
   automated, review-only" model with a real agent that detects anomalies,
   classifies them, decides next steps, and escalates ambiguous cases to a
   human. This makes the "agentic + human-in-loop" requirement real and
   justifies the LangGraph choice.
3. **Funding rate promoted to a first-class metric** (§6) — previously
   captured but unused.
4. **Basis-divergence metric redefined** (§6) — rolling standard deviation of
   the *demeaned* basis is now the primary divergence-volatility measure; RMS
   about zero is kept only as a secondary magnitude measure, because RMS about
   zero is dominated by the carry level, not by divergence.
5. **Liquidity metric redefined** (§6) — cost-to-trade a fixed notional and
   band-limited depth are now primary; the raw 10-level USD sum is demoted to
   secondary.
6. **Roll continuity handling** (§5.4) — a ratio back-adjustment method is now
   specified for any continuous (multi-contract) series, because front-month
   selection alone leaves a level-shift artifact at each roll.
7. **Honest latency scope** (§6.2) — the 1-minute HL sampling makes sub-minute
   lead-lag unmeasurable; the document now states this as a hard constraint
   rather than implying latency analysis is fully deliverable.
8. **CME time-weighted vs snapshot metrics** (§6.3) — the asymmetry between
   CME's tick data and HL's snapshots is now made explicit, with both views
   produced.
9. Added **acceptance criteria** (§9), an expanded **edge-case** section
   covering the new agent and settlement modes (§11), and a **glossary that
   distinguishes futures carry from perpetual funding** (§3).

**v2.2 additions on top of the above:** contract semantics fixed (§6.4); price/
return **correlation promoted to the headline stakeholder deliverable** (§6.6,
§9.2); Mode A trimmed with carry work deferred to the with-spot redo (§5.5);
BigQuery **output** persistence brought into scope while raw input stays
synthetic (§7.3); and all contract calculations parametrized for CL/GC (§7.8).

---

## 1. Purpose and business objective

Compare derivative instruments on a decentralized perpetual exchange
(Hyperliquid) against economically equivalent futures on a centralized
exchange (CME), across three products traded on both venues:

| Pair | CME instruments | Hyperliquid instrument |
|---|---|---|
| Equity index | ES (standard), MES (micro) | SP-USDC (perpetual) |
| Crude oil | CL (WTI) | CL (perpetual) |
| Gold | GC | XYZ:GOLD-USD (perpetual) |

The objective is a structured, repeatable comparison of **liquidity**,
**price divergence**, and **the structural difference between a funding-pinned
perpetual and an expiring, carry-bearing future** — presented as clear metrics
and visuals, driven by an agentic pipeline that investigates its own anomalies
and escalates the ambiguous ones to a human reviewer.

### Why this is non-trivial
CME instruments are **futures with fixed expiries, nanosecond tick data, a
real weekly/holiday closure calendar, and a price that embeds cost-of-carry**.
Hyperliquid instruments are **perpetuals with no expiry, sampled once a minute,
trading 24/7, and pinned toward spot by a funding mechanism**. A credible
comparison must reconcile the carry/funding difference and the sampling
asymmetry before any single metric can be trusted — that reconciliation is the
core of this project.

---

## 2. In-scope vs out-of-scope

**In scope now (built — v1):**
- ES vs SP-USDC and MES vs SP-USDC on synthetic mock data matching the target
  production schema.
- Deterministic five-stage pipeline (Transform → Discover → Save → Visualize
  → Summarize) with an in-notebook UI.

**Specified in v2, not yet built:**
- The two basis-analysis modes (§5.5), with Mode A's settlement-based carry
  treatment deferred to the with-spot redo.
- The anomaly-investigation agent and human-in-loop escalation (§7.5).
- The redefined metrics (funding, demeaned-basis volatility, cost-to-trade
  depth, roll back-adjustment) and **price/return correlation as a first-class
  stakeholder deliverable** (§6).
- **Data-quality guards — crossed-book filter, price/size sanity bounds, and
  schema validation — to be built now against mock data (§11.3–11.4).**
- **BigQuery persistence of the processed output data and insights** (Save
  stage → BigQuery). In scope (§7.3, §11.4).
- CL vs CL, GC vs XYZ:GOLD-USD — design target only; no build until requested,
  but all notional/contract calculations are parametrized so the extension is
  a config exercise (§6.4, §7.8).

**Explicitly out of scope:**
- Trade execution, order routing, or any actionable trading signal.
- Real-time/streaming operation — this is a batch, historical-window tool.
- Any financial advice or recommendation output.
- **Loading raw exchange input data from BigQuery.** Input stays synthetic for
  now; the synthetic schema (§4.1) is treated as canonical and reconciled with
  live BigQuery tables in a later retrofit. Only *output* persistence to
  BigQuery is in scope.

---

## 3. Definitions and reference conventions

| Term | Definition |
|---|---|
| Mid price | (best bid + best ask) / 2 at top of book |
| Notional value | `size × price × point_value`, per leg with its own point value: **ES = 50, MES = 5, SP-USDC = 1** (so the perp is simply `size × price`). Parametrized per product (§6.4). |
| Point value | $ value of one full index-point move per contract (ES $50, MES $5) |
| **Futures carry** | The amount by which a CME future's price sits away from spot due to cost-of-carry (financing minus dividends for equity index; storage and convenience yield for crude; lease/financing for gold). **Converges to zero as the contract approaches expiry.** A moving target across the contract's life. |
| **Perpetual funding** | Periodic payment between perp longs and shorts that pins the perpetual price toward spot. The perpetual's structural analogue to futures carry. Captured as `funding_rate_1h`. |
| Raw basis | `CME_mid − HL_mid`, in index points and $. **Contains futures carry** — not pure cross-venue divergence. |
| Divergence volatility | Rolling standard deviation of the *demeaned* basis — the primary measure of how much the two venues move apart, with the slowly-varying carry level removed. |
| Front month | The CME contract month currently carrying the most open interest |
| Roll period | The window during which open interest shifts from front month to next |
| Continuous series | A single price/basis series spanning multiple contract months, requiring back-adjustment at each roll (§5.4) |
| Settlement price | CME's official daily/session mark per contract month; used as the daily anchor in the without-spot analysis mode (§5.5) |
| As-of alignment | For an HL timestamp T, use the most recent CME tick at or before T |
| Staleness | A CME tick is stale relative to T if older than the configured tolerance (default 90s) |
| Session bucket | One of `normal`, `nyse_open_window`, `nyse_close_window`, `roll_period`, `cme_closed` — mutually exclusive, in that priority order |

---

## 4. Data architecture

### 4.1 Source of truth
The **synthetic-data schema below is treated as canonical for this build.**
Raw input is generated synthetically (loading raw exchange data from BigQuery
is out of scope — §2); the schema is designed to mirror the intended production
tables so it can be **retrofitted to live BigQuery later** without changing the
downstream pipeline. Reconciliation against live tables is a deferred activity,
not a prerequisite for this build.

**CME raw table** (nanosecond, tick-by-tick, event-driven):

| Column | Type | Notes |
|---|---|---|
| `ts_ns` | INT64 | nanosecond epoch, exchange timestamp |
| `symbol` | STRING | `ES`, `MES`, `CL`, `GC` |
| `contract_month` | STRING | e.g. `2026U` — required for roll handling |
| `exchange` | STRING | constant `CME` |
| `bid_px_1..10`, `ask_px_1..10` | FLOAT64 | 10 levels each side |
| `bid_sz_1..10`, `ask_sz_1..10` | INT64 | contracts, each side |
| `volume_cum` | INT64 | cumulative session volume |
| `open_interest` | INT64 | daily OI snapshot, drives roll detection |

**CME settlement table** (available in BigQuery; note that Mode A's
settlement-based carry treatment is **deferred to the with-spot redo** — §5.5):

| Column | Type | Notes |
|---|---|---|
| `settle_date` | DATE | session date |
| `symbol`, `contract_month` | STRING | |
| `settlement_price` | FLOAT64 | official daily/session settlement |

**Hyperliquid raw table** (microsecond, 1-minute snapshot):

| Column | Type | Notes |
|---|---|---|
| `ts_us` | INT64 | microsecond epoch |
| `symbol` | STRING | `SP-USDC`, `CL`, `XYZ:GOLD-USD` |
| `exchange` | STRING | constant `Hyperliquid` |
| `bid_px_1..10`, `ask_px_1..10` | FLOAT64 | 10 levels each side |
| `bid_sz_1..10`, `ask_sz_1..10` | FLOAT64 | fractional sizes allowed |
| `funding_rate_1h` | FLOAT64 | hourly funding rate — now a first-class input |
| `open_interest` | FLOAT64 | |

**Spot / index reference table** (new in v2 — required only for the with-spot
mode; gated on availability):

| Column | Type | Notes |
|---|---|---|
| `ts` | TIMESTAMP | reference timestamp |
| `symbol` | STRING | underlying identifier |
| `spot_price` | FLOAT64 | index / spot reference level |

### 4.2 Processed / output schema
Written by the Save stage. Adds funding, settlement-anchored, and (when
available) spot-attribution columns to the v1 schema.

Core (as v1): `ts`, `hl_mid`, `cme_mid`, `hl_spread`, `cme_spread`,
`hl_imbalance`, `cme_imbalance`, `cme_market_open`, `cme_stale`,
`is_roll_period`, `is_open_window`, `is_close_window`, `session_bucket`.

**Changed / new in v2:**

| Column | Notes |
|---|---|
| `raw_basis_points`, `raw_basis_usd_cme_pv`, `raw_basis_usd_hl_pv` | raw CME−HL, carry-bearing (renamed from `basis_*` to make the carry content explicit) |
| `basis_demeaned` | basis minus its rolling mean (carry level removed) |
| `basis_divergence_vol` | rolling std of `basis_demeaned` — **primary divergence metric** |
| `basis_rms_zero` | rolling RMS about zero — secondary magnitude metric |
| `basis_delta` | first difference of raw basis — high-frequency divergence, carry-free |
| `funding_rate_1h`, `funding_cum`, `funding_annualized` | perp funding, cumulative and annualized |
| `cme_ret_1m`, `hl_ret_1m` | 1-minute log returns per venue (basis of correlation, §6.6) |
| `rolling_return_corr` | rolling-window return correlation between venues (§6.6) |
| `cme_cost_to_trade_1m`, `hl_cost_to_trade_1m` | slippage cost to fill a fixed notional (default $1M) — **primary liquidity metric** |
| `cme_depth_band_10bps`, `hl_depth_band_10bps` | depth within ±10bps of mid |
| `cme_depth_usd_l10`, `hl_depth_usd_l10` | raw 10-level sum — secondary |
| `cme_spread_twa`, `cme_depth_twa` | time-weighted-average CME metrics over the minute (§6.3) |
| `contract_month`, `days_to_expiry` | for roll analysis |
| `implied_carry_settlement` | **deferred to the with-spot redo** — settlement-to-settlement carry drift |
| `cme_carry_to_spot`, `hl_premium_to_spot` | **with-spot mode only** (the redo): each leg's distance from spot |
| `settlement_price` | loaded when the settlement table is used (carry work deferred, §5.5) |

### 4.3 Reference data (JSON, editable without code changes)
- `product_specs.json` — point value, tick size, currency, contract unit,
  expiry cycle, settlement type, roll rule
- `data_characteristics.json` — raw schema docs + alignment policy
- `calendar.json` — NYSE cash hours, CME Globex hours, maintenance break,
  holiday list, demo window
- **`anomaly_config.json` (new)** — anomaly detection thresholds and the
  agent's classification rules and guardrails (§7.5)

---

## 5. Transformation logic (Stage 1)

### 5.1 The core problem
HL is one row per minute; CME is one row per book event at nanosecond
resolution, absent entirely during closures. They cannot be joined on exact
timestamp equality.

### 5.2 Alignment method — as-of / last-tick
For every HL 1-minute timestamp `T`: take the most recent CME tick with
`ts_ns <= T`. If that tick is older than the staleness tolerance (default 90s),
null the CME fields and set `cme_stale = True`. If `T` falls in a known closure
(weekend, holiday, maintenance break), set `cme_market_open = False` and null
the CME fields **as an expected closure, not a data-quality failure**. The two
kinds of NULL are handled differently downstream: closure NULLs are silent;
staleness NULLs during open hours are candidate anomalies for the agent (§7.5).

Implemented with `pandas.merge_asof(direction="backward")`. Given the confirmed
"days at a time" volume and the decision to keep pandas in-notebook, this is the
default. **Caveat:** full-depth CME tick data can be tens of millions of rows
even for a few days; if a run's raw CME pull is large enough to strain memory,
the as-of alignment should be pushed into BigQuery SQL (window function) so only
the minute-grid result is pulled into pandas. This is a runtime fallback, not a
redesign (§7.4).

### 5.3 Notional value
`notional = size × price × point_value`, applied per leg with each product's
own point value (ES 50, MES 5, SP-USDC 1) — never a shared multiplier across
venues. Values are parametrized from `product_specs.json` (§6.4).

### 5.4 Roll handling and continuous-series back-adjustment
Front month = contract with highest open interest; a roll is flagged when OI
crosses from front to next quarterly contract, detected from
`open_interest`/`contract_month` in the feed (not a fixed calendar date).

**New in v2:** front-month selection alone leaves a **level-shift artifact** —
the next contract carries more carry (further from expiry), so a naive
continuous series steps at each roll even when nothing about cross-venue pricing
changed. For any continuous (multi-contract) series, apply **ratio
back-adjustment** (scale the pre-roll history by the ratio of the two contracts'
prices at the roll instant) so the joined series is continuous. Per-contract
analysis (default) does not need this; continuous analysis does. Both views are
produced, clearly labeled.

### 5.5 Two basis-analysis modes

Because we do not currently have spot prices, the runnable primary mode is the
without-spot mode below. The proper carry treatment (settlement anchoring,
funding-vs-carry decomposition) is **deferred to the with-spot redo (Mode B)** —
the stakeholder confirmed that the carry question can only be resolved with
spot, and the analysis will be redone once spot data is available. Mode A is
therefore kept deliberately lean and does not overclaim carry handling.

#### Mode A — Without spot. *Runnable now, lean.*
No spot reference. Uses CME instrument prices, HL perp prices, and HL funding.
Deliverables:
1. **Raw level basis** (`raw_basis_points`), reported with an explicit
   "contains carry" caveat. Both ES and SP-USDC track the same S&P 500 (SPX)
   level (§6.4), so the raw basis is directly meaningful in index points; no
   rescaling is needed. Useful for gross divergence and its dynamics.
2. **First-difference divergence** (`basis_delta`) — analyzing the *change* in
   basis strips the slowly-varying carry and isolates high-frequency
   cross-venue divergence and minute-level lead-lag. This is the primary
   no-spot divergence view.
3. **Funding rate reporting** (`funding_rate_1h`, `funding_cum`,
   `funding_annualized`) — reported as the perpetual's own carry mechanism.
   Comparing it against a settlement-implied futures carry is **deferred to
   Mode B** (it needs spot to be interpreted correctly).
4. **Price/return correlation** (§6.6) — the headline stakeholder deliverable,
   fully computable without spot.

> **Deferred to Mode B:** settlement-anchored multi-day de-trending and any
> funding-vs-implied-carry decomposition. The CME settlement table is available
> and can be loaded, but these carry analyses are only meaningful alongside
> spot, so they are parked until the with-spot redo rather than approximated now.

#### Mode B — With spot present. *Designed; gated on spot data (the planned redo).*
Adds a spot/index reference. **Spot does not change the raw cross-venue
difference (it cancels in `HL − CME`); it enables attribution and validation:**
1. **Decompose** the raw basis into each venue's distance from spot:
   `cme_carry_to_spot = CME_mid − spot` and
   `hl_premium_to_spot = HL_mid − spot`. This shows *which venue* is driving
   the divergence at any moment.
2. **Validate carry behavior** — regress `cme_carry_to_spot` on
   `days_to_expiry`; confirm it converges toward zero at expiry as carry theory
   predicts. A future that doesn't converge is itself an anomaly.
3. **Validate funding behavior** — correlate `hl_premium_to_spot` with the
   funding rate; the perp premium should oscillate around zero, pulled by
   funding.
4. If financing rates and dividends/storage become available, extend to a
   theoretical fair-value futures price and measure the future's richness/
   cheapness to fair value (further open item — needs those inputs).

---

## 6. Metrics (Stage 2)

### 6.1 Liquidity (revised)
| Metric | Method | Priority |
|---|---|---|
| **Cost-to-trade fixed notional** | Walk the book to fill a default $1M order; report average fill price slippage vs mid, per venue | **Primary** |
| **Band-limited depth** | Sum of USD notional within ±10bps of mid, per side | **Primary** |
| Top-of-book spread | `ask_px_1 − bid_px_1` | Supporting |
| Raw 10-level USD depth | Σ over 10 levels × point value | Secondary (context only — deep levels aren't executable at top price) |
| Order-book imbalance | `(bid_sz_1 − ask_sz_1)/(bid_sz_1 + ask_sz_1)`, NaN if both zero | Supporting, with the caveat in §6.2 |

### 6.2 Price divergence and the honest limits of latency analysis
- **Primary divergence metric:** `basis_divergence_vol` = rolling std of the
  demeaned basis. RMS-about-zero (`basis_rms_zero`) is kept only as a secondary
  magnitude measure — it is dominated by the carry level and is *not* a clean
  divergence measure.
- **Latency (honest scope):** true sub-minute lead-lag price discovery is
  **not measurable** because HL is sampled once a minute. What *is* deliverable
  is minute-level lead-lag via cross-correlation of one-minute returns (does
  CME's minute return predict HL's next-minute return, or vice versa), reported
  with an explicit caveat that anything finer than one minute is invisible on
  the HL side. The alignment `cme_age_seconds`/`cme_stale` fields quantify data
  freshness, which is a data-pipeline latency proxy, not a price-discovery
  latency measure — the document keeps these two ideas separate.
- **Imbalance caveat:** top-of-book imbalance is a microstructure signal that
  is informative tick-by-tick on CME but is a single instantaneous reading per
  minute on HL; it is reported for both but interpreted cautiously on the HL
  side.

### 6.3 CME time-weighted vs snapshot (new)
CME has full tick data within each minute; HL has one snapshot. Two views are
produced, clearly labeled:
- **Snapshot-match** (default for like-for-like cross-venue comparison): CME
  metrics taken from the last tick before the minute mark, matching HL's single
  snapshot.
- **Time-weighted CME** (supplementary, exploits CME's richer data):
  `cme_spread_twa`, `cme_depth_twa` averaged over the minute. Used to describe
  CME's within-minute behavior, **not** directly differenced against HL (that
  would be non-comparable).

### 6.4 Notional / point-value (confirmed) and parametrization for CL/GC
Notional is `size × price × point_value`, applied per leg with its own point
value, all sourced from `product_specs.json` — **nothing is hardcoded per
product**:

| Instrument | point_value | Notional |
|---|---|---|
| CME ES | 50 | `50 × price × size` |
| CME MES | 5 | `5 × price × size` |
| HL SP-USDC | 1 | `price × size` (perp is quoted/margined in USDC, size in index units) |

Both **CME ES and HL SP-USDC track the same underlying — the S&P 500 index
(SPX) — at the same level**, so the raw basis in index points is directly
meaningful; no rescaling or normalization is required.

**Parametrization for CL/GC (design target):** because point value, tick size,
and contract unit are all read from config, extending to WTI crude (CME `CL`,
1,000 bbl/contract) and gold (CME `GC`, 100 troy oz/contract) against their HL
perpetuals is a matter of adding config entries and a data loader — the
notional, depth, cost-to-trade, basis, and correlation calculations are all
written to consume `point_value`/`tick_size`/`contract_unit` as parameters and
need no per-product code changes (§7.8).

### 6.5 Temporal windowing
- **NYSE open/close windows:** first/last 15 minutes of the 09:30–16:00 ET cash
  session. **Per direction, the same NYSE definition is applied to CL and GC**
  for cross-pair consistency — a deliberate simplification, not a claim that
  NYSE hours are economically meaningful for commodities. Flagged so it is not
  mistaken for an oversight.
- **Roll period / CME closed:** per §5.4 and `calendar.json`.

### 6.6 Price and return correlation (headline stakeholder deliverable — new)
The stakeholder's primary interest is how closely the CME future and the HL
perpetual co-move. Reported as both metrics and visuals:
- **Level correlation** (Pearson) between `cme_mid` and `hl_mid`, reported *with
  a caveat*: two trending price-level series are almost always highly
  correlated, so a high level-correlation number is expected and not by itself
  informative.
- **Return correlation** (Pearson/Spearman of 1-minute log returns) — the
  **honest measure of co-movement**, since returns are (near-)stationary. This
  is the headline number.
- **Rolling correlation** — a rolling-window return correlation time series, to
  show how co-movement strengthens or breaks down over time.
- **Correlation by session bucket** — separately for `normal`,
  `nyse_open_window`, `nyse_close_window`, `roll_period`, so the stakeholder can
  see whether co-movement degrades in high-volatility windows.
- **Visuals:** a scatter of CME vs HL returns with a fitted regression line
  (slope ≈ beta between venues), and the rolling-correlation time series. Plus
  a compact summary-stats table (level corr, return corr, rolling corr
  min/mean/max, beta) so the two products can be compared at a glance.

---

## 7. Architecture

### 7.1 Two layers
1. A **deterministic pipeline substrate** — the five stages, each a pure,
   reproducible function. Given the same inputs and config, identical outputs.
2. An **anomaly-investigation agent** (§7.5) that sits above the substrate,
   calls its stages as tools, investigates flagged anomalies, decides next
   steps, and escalates ambiguous cases to a human. The agent's *routing* is
   non-deterministic; the *metrics it computes* remain deterministic (§7.6).

### 7.2 Deterministic substrate — LangGraph nodes
`transform_data → discover_insights → save_insights → visualize → summarize`.
State is a single `TypedDict` threaded through nodes. This layer alone is what
v1 built.

### 7.3 Data access layer and persistence
**Input (read):** `DataLoader` abstracts the raw-data source behind
`load_hl()`, `load_cme()`, `load_cme_settlement()`, and (Mode B) `load_spot()`.
Input is **synthetic for this build** (`data_mode="mock"`); reading raw exchange
data from BigQuery is out of scope and deferred to a later retrofit. The
`bigquery` read path remains a documented stub so the retrofit is a config
change, not a rewrite.

**Output (write) — in scope:** the Save stage persists the **processed output
data and the derived insights** (metrics table + agent findings/decision log)
to BigQuery. This is a real deliverable, not a stub. Because output rows can be
re-generated on re-runs, the output tables enforce an idempotency/write policy
(§11.4) so a re-run does not duplicate rows: either `WRITE_TRUNCATE` per
(pair, analysis-window) partition, or a merge keyed on `(symbol, ts)`.

### 7.4 Compute placement (confirmed: days-at-a-time, pandas)
Default: pandas in-notebook. Documented fallback: if a run's raw CME tick pull
is large enough to strain memory, push the as-of alignment into BigQuery SQL and
pull only the minute-grid result. Same logical result, different execution
location.

### 7.5 Anomaly-investigation agent (the core of the "agentic" requirement)

**Role.** After the Discover stage produces metrics, the agent examines them,
identifies anomalies, classifies each, decides an action, and loops until either
all anomalies are resolved/explained or it escalates to the human. It is a
genuine decision-making loop, not a scripted pass.

**Tool surface (what the agent can call).**
- Re-run any pipeline stage with adjusted parameters (e.g. tighter staleness
  tolerance, different RMS window).
- Slice/drill: recompute metrics on a narrower time window or a single
  session-bucket.
- Query reference data: product specs, calendar, roll status for a timestamp.
- Cross-check: compare a suspicious value against neighboring windows or the
  other CME symbol (ES vs MES sanity check).
- Escalate: surface a structured finding to the human (interrupt).

**Anomaly taxonomy (what triggers investigation), thresholds in
`anomaly_config.json`.**
- Basis outliers beyond N rolling std.
- Spread spikes / depth collapses beyond a threshold.
- Staleness clusters during open hours (distinct from closures).
- Crossed or locked books (best bid ≥ best ask).
- Zero/negative prices or sizes.
- Unexpected gaps in the HL minute grid (venue outage).
- Basis level-shifts not explained by a flagged roll.
- Funding/carry inconsistency (Mode B: carry not converging near expiry).

**Decision policy (how the agent classifies each anomaly).** Each anomaly is
classified into one of:
- **Known market-structure effect** (roll, open/close window, closure) → annotate
  and continue, no escalation.
- **Data-quality problem** (crossed book, stale cluster, bad tick, schema drift)
  → apply the configured remedy (flag/exclude) and record it; escalate only if
  the remedy materially changes results.
- **Genuine cross-venue dislocation** (divergence unexplained by carry, funding,
  or structure) → this is a *finding*, not an error; escalate to the human with
  the supporting slice and context.
- **Ambiguous** → escalate.

**Human-in-loop (the feedback the requirement asked for).** The agent runs
autonomously on known-effect and routine data-quality cases; it **pauses via a
LangGraph `interrupt()`** for ambiguous cases and genuine dislocations,
presenting a structured finding. The human approves, overrides, or supplies a
parameter change, and the graph resumes — a real loop, not review-only.

**Autonomy policy (confirmed).** Known market-structure effects auto-proceed
with annotation. Data-quality problems are auto-remedied per the configured
rule, and escalated only if the remedy materially changes results. Genuine
cross-venue dislocations and ambiguous cases always escalate to the human.
Classification thresholds are tuned to over-escalate rather than under-escalate.

**LangGraph structure (now non-trivial).** Conditional edges route from an
`investigate` node to either `act_and_recheck` (cycle back into investigate) or
`escalate_to_human` (interrupt) or `done`. The cycle + conditional routing +
interrupt is what genuinely justifies LangGraph over plain function composition
— a point v1 could not honestly make.

**Guardrails.**
- Max investigation iterations per run (config); on exceeding, escalate wholesale.
- Cost/token cap for the agent's LLM calls.
- Every agent decision (anomaly, classification, action, rationale) is logged to
  an audit trail persisted alongside the processed data.
- Agent LLM calls run at temperature 0.

### 7.6 Reproducibility with a non-deterministic agent
The tension between "reproducible NFR" and "LLM agent" is resolved by
separation: **all metrics are computed by deterministic substrate code**, so
the numbers are reproducible given the same data/config. The agent only decides
*which slices to investigate and what to escalate* — its routing may vary run to
run, but it cannot change a computed metric value, only flag, exclude (per
logged rule), or annotate it. The full decision log makes any run auditable
and explainable even though the routing path is not bit-identical.

### 7.7 Technology stack
| Layer | Choice | Rationale |
|---|---|---|
| Orchestration | LangGraph | now justified by conditional routing, the investigate cycle, and `interrupt()`-based human-in-loop — not just the linear DAG |
| Agent LLM | Anthropic API (temp 0) | classification and next-step decisions; deterministic-as-possible; full logging |
| Data processing | pandas (days-at-a-time) | matches confirmed volume; BigQuery-SQL fallback documented |
| Persistence | BigQuery for **output** data + insights (in scope); parquet locally during dev. Raw **input** stays synthetic (BigQuery input read deferred) | matches confirmed scope |
| Visualization | Plotly | interactive, notebook-native, handles session-bucket shading |
| UI | ipywidgets, in-notebook | per requirement |

### 7.8 Parametrization for multi-product extension (CL/GC)
The system is built product-agnostic so that adding WTI crude and gold is a
configuration exercise, not new code. Everything product-specific flows from
`product_specs.json`:

| Parameter | Source | Consumed by |
|---|---|---|
| `point_value` | product_specs | notional, depth (USD), cost-to-trade, basis-in-$ |
| `tick_size` | product_specs | spread quantization, min-increment handling |
| `contract_unit` | product_specs | notional interpretation (index pts / bbl / troy oz) |
| `expiry_cycle`, `roll_rule` | product_specs | roll detection, continuous back-adjustment |
| `underlying` | product_specs | correlation pairing, spot mapping (Mode B) |
| session hours | calendar.json | temporal windowing |

No calculation hardcodes ES/SP-USDC values. Adding `CL`/`GC` and their HL perps
requires: (1) config entries, (2) a data-loader/mock-generator call for the new
symbols, (3) invoking the pipeline with the new symbol pair. The one item
needing a human decision at that point is the session-window definition for
commodities (§6.5), which is currently defaulted to NYSE hours by direction.

---

## 8. Non-functional requirements
- **Reproducibility:** metrics deterministic given data + config (§7.6).
- **Auditability:** explicit `cme_market_open`/`cme_stale` flags; full agent
  decision log; no silent interpolation.
- **Config-over-code:** specs, calendar, alignment policy, and anomaly rules in
  JSON.
- **Cost control (BigQuery):** scope queries by date partition and symbol
  cluster; confirm partitioning before pointing at production tables.
- **Credential handling:** no credentials in notebook/config; BigQuery via
  application-default credentials or a user-supplied service account outside
  version control; Anthropic key via environment variable.

---

## 9. Acceptance criteria

### 9.1 Pipeline correctness (engineering sign-off)
1. **Alignment correctness:** count of `cme_closed` minutes matches the calendar
   arithmetic exactly (verified in v1 for the S&P pair).
2. **NULL semantics:** every CME NULL is attributable to either a closure or a
   logged staleness event; no unexplained NULLs.
3. **Carry sanity (Mode B):** `cme_carry_to_spot` trends toward zero as
   `days_to_expiry → 0`; any exception is surfaced as a finding.
4. **Roll continuity:** the back-adjusted continuous basis series has no
   step-discontinuity at flagged rolls beyond a configured tolerance.
5. **Agent audit:** every escalation has a logged rationale and supporting
   slice; every auto-resolved anomaly has a logged classification and remedy.
6. **Reproducibility:** re-running on identical data/config reproduces all
   metric values bit-for-bit (agent routing may differ; metrics may not).
7. **Output persistence:** processed data and insights land in BigQuery with no
   duplicate rows across re-runs (§7.3, §11.4).

### 9.2 Stakeholder findings acceptance (analytical sign-off)
A CME stakeholder signs off on the *findings* when the deliverables answer the
following, each as a metric plus a supporting visual and summary stat:
1. **Liquidity comparison** between CME and HL across **bid-ask spread, book
   depth, and cost-to-trade** — side by side per venue.
2. **Liquidity quality** via **order-book imbalance**, so the stakeholder can
   judge whether displayed liquidity is balanced or one-sided.
3. **Liquidity under stress** — the above metrics **during intraday
   high-volatility windows (NYSE open/close) and during high-volatility
   regimes (whole volatile days)**, shown against normal conditions so
   degradation is visible.
4. **Price co-movement** — the correlation between the **CME futures mid price
   and the HL SP-USDC mid price** (§6.6), delivered as both a **visualization**
   (returns scatter with regression line; rolling-correlation time series) and
   a **summary-stats table** (level corr, return corr, rolling corr
   min/mean/max, cross-venue beta), so the product can be compared across the
   two exchanges at a glance.

These four are the explicit stakeholder acceptance targets; the automated
end-of-run summary (§7.2 Summarize stage) must surface all four.

---

## 10. (reserved)

---

## 11. Edge cases

### 11.1 Timestamp and alignment
| Edge case | Risk | Handling |
|---|---|---|
| CME tick older than staleness tolerance | Stale price overstates freshness/liquidity | `cme_stale` flag + NULL beyond tolerance; stale *clusters* during open hours become agent anomalies |
| HL minute missing (venue outage) | Silent gap read as "no divergence" | **New:** explicit `hl_missing` flag; gap becomes an agent anomaly (venue-outage class) |
| CME data during maintenance break | Confused with a pipeline failure | Modeled as expected closure, identical to weekend/holiday |
| Out-of-order / late tick arrival | Dropped or double-counted | Sort by `ts_ns` before `merge_asof`; production ingestion must be idempotent (open item) |
| DST transition | NYSE-hours flags off by one hour half the year | **Fix required before real data:** timezone-aware `America/New_York` conversion per calendar day, not a fixed UTC offset |
| ns-vs-µs dtype mismatch on join | Silent all-NULL join | Both timestamps cast to common `datetime64[ns, UTC]` before join |

### 11.2 Market structure / calendar
| Edge case | Risk | Handling |
|---|---|---|
| CME closed, HL trading | Basis against a null/stale CME leg | `cme_market_open=False`, CME nulled, bucket `cme_closed` |
| Roll contract ambiguity (two months active) | Basis against wrong contract | Highest-OI front-month selection; `is_roll_period` flagged; continuous series ratio back-adjusted (§5.4) |
| Roll level-shift in continuous basis | Artifact mistaken for divergence | Ratio back-adjustment; per-contract analysis default; agent flags unexplained level-shifts |
| Commodity session hours (CL/GC lack NYSE open) | Quiet periods mislabeled high-volatility | Documented deliberate simplification (§6.5) |
| Holiday-calendar drift | Post-coverage sessions treated as open | `calendar.json` versioned/dated; refresh annually or source live |

### 11.3 Price / quote data quality
| Edge case | Risk | Handling |
|---|---|---|
| Crossed/locked book (bid ≥ ask) | Negative spread breaks metrics | **Build now (mock):** validity check flags/excludes crossed rows; becomes an agent anomaly |
| Zero/negative price or size | Corrupts mid and all $ metrics | **Build now (mock):** sanity-bound check vs short rolling median before accepting a tick |
| Fewer than 10 populated levels | Depth undercounts, read as thin | Depth sums only populated levels; NULL levels propagate as NULL, not zero — explicit `fillna` policy required for real data (open item) |
| Imbalance with both sides zero | Divide-by-zero | Denominator NaN'd → NaN imbalance, no error |
| Basis with one leg NULL | NULL treated as zero → fake "zero basis" | NULL-aware arithmetic; NULL propagates to NULL (verified in v1) |
| Rolling window spanning closed→open | NULLs as zero deflate volatility | `min_periods` guard; NULLs excluded, not counted as zero |
| Tick-size quantization (CME $0.25 vs HL $0.10) | Basis "steps" mistaken for divergence | Documented expected artifact; informative about relative granularity; called out in interpretation |
| ES vs MES point-value mix | "$1 basis" means 10× different exposure | Basis reported per product with its own point value; cross-symbol comparison in $ notional only |
| USDC de-peg | $ basis assumes USDC≈USD; de-peg adds a hidden currency-basis component | Not modeled; flagged simplification; a de-peg-period analysis needs a USDC/USD reference leg |

### 11.4 Reference data / configuration
| Edge case | Risk | Handling |
|---|---|---|
| Spec JSON out of sync with exchange terms | Misstated notional/tick value | Specs isolated in JSON for independent review; ES/MES/SP-USDC values now confirmed (§6.4) |
| Schema evolution upstream | Silent column ignore or hard error | **Build now (mock):** schema-validation step at start of `transform_data` fails loudly on unexpected column set; schema drift is an agent anomaly |
| Duplicate rows on **output** re-run | Re-running a window double-writes insights | **In scope:** output tables use `WRITE_TRUNCATE` per (pair, window) partition or a merge on `(symbol, ts)`; raw input idempotency is out of scope (input is synthetic) |
| Settlement missing for a contract-day | Anchor fails silently | Settlement-based carry is **deferred to the with-spot redo** (§5.5); when built, missing settlement becomes an agent anomaly (day flagged, not anchored to a stale prior) |
| Spot series gaps (Mode B) | Attribution silently drops rows | Spot gaps flagged; affected windows excluded from attribution with a logged note |

### 11.5 Agent-specific (new in v2)
| Edge case | Risk | Handling |
|---|---|---|
| Agent loops without converging | Runaway cost / no result | Max-iteration cap → wholesale escalation to human |
| Agent misclassifies a genuine dislocation as a known effect | Real finding buried | Conservative policy: ambiguous → escalate; classification thresholds tuned to over-escalate rather than under-escalate; all classifications logged for review |
| Agent LLM cost blowout | Budget overrun | Per-run token/cost cap; agent halts and escalates on cap |
| Non-deterministic routing undermines reproducibility | Results not reproducible | Metrics are deterministic substrate code; only routing varies; full decision log (§7.6) |
| Agent acts on a hallucinated anomaly | Wasted work / spurious finding | Agent may only act via the defined tool surface on real flagged values; it cannot invent data; every action ties to a logged, data-derived trigger |
| Human unavailable at an interrupt | Pipeline stalls | Configurable timeout → park the finding in the audit log and continue with the anomaly marked "unreviewed," rather than blocking indefinitely |

### 11.6 Operational (BigQuery phase)
| Edge case | Risk | Handling |
|---|---|---|
| Large/unpartitioned query scan | Cost/latency | Confirm date-partition + symbol-cluster before use |
| Transient BigQuery API failure | Ungraceful mid-run failure | **Recommended:** retry/backoff around `DataLoader` BigQuery calls |
| Extended venue outage (hours/days) | Read as a very quiet market | Explicit gap-length alert threshold, separate from routine closure/staleness flags |

---

## 12. Open items / decisions log

**Resolved in v2.1:** agent autonomy default (accepted); CME settlement table
availability (available); data-quality guards timing (build now against mock
data).

**Resolved in v2.2:**
- **Live BigQuery schema** — deferred; synthetic schema is canonical for now
  and will be retrofitted to live tables later (§4.1).
- **Hyperliquid contract semantics** — SP-USDC notional is `size × price`
  (point value 1); ES is `50 × price`; both track the S&P 500 (SPX) at the same
  level, so raw basis needs no rescaling (§6.4).
- **Stakeholder findings acceptance** — defined as four targets (liquidity
  across spread/depth/cost-to-trade; imbalance for liquidity quality; liquidity
  under intraday and regime volatility; CME-vs-HL price correlation with
  visuals and summary stats). Price/return correlation promoted to a first-class
  deliverable (§6.6, §9.2).
- **Settlement usage in Mode A** — deferred; the carry treatment will be redone
  properly with spot prices (Mode B). Mode A stays lean (§5.5).
- **Output persistence / idempotency** — BigQuery persistence of processed
  output + insights is in scope with a no-duplicate write policy; raw input load
  is out of scope (§7.3, §11.4).
- **CL/GC** — design only; all calculations parametrized so extension is a
  config exercise (§6.4, §7.8).

**Remaining forward-looking items (not blocking the next build):**
1. **The with-spot redo (Mode B)** — execute once spot/index data is available;
   this is where carry attribution, settlement anchoring, and funding-vs-carry
   are properly done.
2. **Commodity session-window definition** — when CL/GC are built, decide
   whether to keep NYSE hours (current default) or switch to NYMEX/COMEX Globex
   session hours (§6.5).
3. **DST handling** — before any live-data retrofit, replace the fixed-offset
   NYSE-hours logic with timezone-aware per-day conversion (§11.1).
4. **Live BigQuery retrofit** — reconcile the synthetic schema with production
   tables (§4.1) when input-loading comes into scope.
