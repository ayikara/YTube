# CME Group Messaging Efficiency Program (MEP) — Implementation Prompt

## Context

You are a senior engineer on the CME Group technology team. Your task is to design and implement the **Messaging Efficiency Program (MEP)** compliance engine — a T+1 batch system that identifies CME Globex market participants whose messaging-to-volume ratios exceed established benchmarks, making them liable for surcharges.

Two source documents are attached as PDFs:

1. **MEP Specification** (`2024-revised-mep.pdf`) — the full program rules (v11.8, May 2026), covering scoring, tiers, exemptions, EMT thresholds, mass quotes, and partner exchanges.
2. **MEP Benchmarks** (`upcoming-mep-benchmarks.pdf`) — the current quarterly product group benchmark ratios by tier.

Read both documents completely before proceeding. All implementation decisions must trace back to specific sections of these documents.

---

## Business Rules Summary

### What the program measures

The MEP computes a **volume ratio** per CME Globex Firm ID (GFID) per product group per trade date:

```
Volume Ratio = Total Messaging Score / Total Traded Volume (contracts)
```

If this ratio exceeds the **tier-adjusted product group benchmark**, the firm is non-compliant and subject to a $1,000/day/product-group surcharge.

### GFID identification

A GFID is the 3-character alpha-numeric value in positions 4–6 of iLink tag 49-SenderCompID. All scoring is at GFID level unless the exchange approves aggregation of multiple GFIDs for a single entity.

### Message scoring (Section 3 of spec)

Each message is weighted based on three dimensions:

| Dimension | Categories |
|---|---|
| **Session** | RTH (07:00–16:00 CT) vs ETH (all other times) |
| **Month type** | Front month outright vs Back month / Spread |
| **Message type** | New Order (D), Modification (G), Cancellation (F), Mass Action (CA), Fill-and-Kill (Tag59=3), MinQty (Tag110) |

Weighting factors from the spec:

| Message Type | RTH Front Month | RTH Back Month / Spread | ETH (all) |
|---|---|---|---|
| New Order (D) | 0 | 0 | 0 |
| Modification (G) | 1 | 0.1 | 0.1 |
| Cancellation (F) | 3 | 0.3 | 0.3 |
| Mass Action (CA) | 3 × N orders cancelled | 0.3 × N orders cancelled | 0.3 × N orders cancelled |
| Fill & Kill | 3 | 0.3 | 0.3 |
| MinQty elimination | 3 | 0.3 | 0.3 |

**Critical:** New orders always score 0. Mass Action (MsgType=CA) raw message count = number of individual orders cancelled (not 1 per CA message). This applies to both raw count for tier determination AND weighted score.

### Raw message count and exclusions

- **Raw message count** = Orders + Modifications + Cancellations + Eliminations (ETH + RTH combined)
- Firms with **≤ 50,000 raw daily messages** per product group are **excluded** from MEP scoring entirely
- Mass quote messages (MsgType=i) are excluded from standard MEP — handled by a separate mass quote pipeline

### Messaging tiers (Section 6 of spec)

| Raw Daily Messages (ETH+RTH) | Effective Benchmark |
|---|---|
| ≤ 50,000 | Excluded — not applicable |
| 50,001 – 100,000 | 3× product group benchmark |
| 100,001 – 150,000 | 2× product group benchmark |
| > 150,000 | 1× product group benchmark (base) |

Tier is determined by the **unweighted** raw message count.

### Volume calculation

- Volume = total contracts traded, both sides
- **Spread volume** is multiplied by 3×: spread instrument + Leg A + Leg B each counted separately
- Zero volume = no breach determination (not a breach)

### Non-compliance

A GFID is non-compliant on a given trade date if its volume ratio in **any** product group exceeds the tier-adjusted benchmark.

For product groups evaluated at **second-level granularity**, any single second breaching the threshold constitutes a daily non-compliance.

### Exemptions and waivers (Section 11 of spec)

Applied in strict priority order (waterfall):

1. **Not a breach** → PASS
2. **Market maker exempt** → AUTO_WAIVED (still subject to EMT)
3. **6× breach** (ratio > 6× effective benchmark) → SURCHARGE_APPLIED (no waivers possible)
4. **Monthly aggregate ratio ≤ benchmark** → PASS_ON_MONTHLY
5. **One-per-month automatic daily waiver** available → AUTO_WAIVED
6. **Month not yet closed** → POTENTIAL (status pending)
7. **All exhausted** → SURCHARGE_APPLIED

Holiday calendar: standard MEP is not enforced on U.S. holidays per the MEP Holiday Calendar. EMT does NOT observe the holiday calendar.

### EMT (Excessive Messaging Threshold — Sections 15–16)

Separate from standard MEP. Two tracks:

| Track | Raw Msgs Threshold | Ratio Threshold | Surcharge |
|---|---|---|---|
| iLink Session ID | > 1 million | > 500:1 | $10,000 + $1,000 port closure |
| GFID (general) | > 10 million | > 500:1 | $10,000 + $1,000 port closure |
| GFID (SS-SOFR, eff. Jan 2026) | > 5 million | > 50:1 | $10,000 + $1,000 port closure |
| GFID MPS (SS-SOFR, eff. Apr 2026) | > 2,000 modifications/second | per-second | $5,000 per breach |

### GFID aggregation (Section 2)

By default each GFID is independent. The exchange may approve combining multiple GFIDs for a single entity. Stored in a lookup table with effective dates. Aggregation fails (reverts to individual) when daily ratio > 6× benchmark. Aggregation does NOT apply to Mass Quote MEP, EMT iLink Session, or EMT GFID programs.

### Status values reported

`PASS`, `POTENTIAL`, `AUTO_WAIVED`, `PASS_ON_MONTHLY`, `PASS_ON_AGGREGATION`, `SURCHARGE_APPLIED`

---

## Input Data Available

Day-level aggregated order entry data is available in BigQuery with this schema:

| Column | Description |
|---|---|
| `account` | Trading account identifier |
| `sessionid` | iLink session identifier |
| `symbol` | Product group (maps to MEP product group) |
| `securitydescription` | Instrument-level identifier (contract/expiry) |
| `firm` | CME Globex Firm ID (GFID, 3-char) |
| `trade` | Count of trade messages |
| `elimination` | Count of elimination messages (FAK/FOK/MinQty) |
| `orders` | Count of new order messages |
| `cancels` | Count of cancel messages |
| `trade_volume` | Total contracts traded |

**Note:** This data is already aggregated to day level grouped by (account, sessionid, symbol, securitydescription, firm). The session (RTH vs ETH) and month type (front vs back) classification must be resolved by joining reference data tables.

---

## Reference Data Tables (in BigQuery)

| Table | Purpose |
|---|---|
| `ref.product_spec` | Product taxonomy: product group, subgroup, asset class, options/futures flag, expiry dates |
| `ref.front_month_flag` | Daily flag per instrument: is_front_month (Y/N) from CME Reference Data API |
| `ref.trading_schedule` | RTH open/close times by product group, holiday calendar, abbreviated sessions |
| `ref.mep_rules` | Versioned benchmark ratios per product group with effective dates |
| `ref.mep_message_weights` | Weighting factors by session × month type × message type with effective dates |
| `ref.messaging_tiers` | Tier thresholds and benchmark multipliers with effective dates |
| `ref.gfid_aggregation_groups` | Exchange-approved GFID combinations with effective dates |
| `ref.market_maker_exemptions` | Registered market maker GFIDs exempt from standard benchmarks |

---

## Architecture Requirements

### Technology stack
- **Google BigQuery** — all raw data storage and SQL aggregation
- **Python** — rule engine for conditional logic, exemptions, EMT checks
- **Looker** — T+1 compliance dashboards and email notification triggers

### Design principle: split SQL and Python by operation type

**BigQuery SQL handles:** set operations at scale — scanning 100M+ order event rows, joining weight/reference tables, grouping by GFID/product group per second and per day. Data stays where it lives; no network transfer.

**Python handles:** conditional branching that SQL does poorly — the exemption waterfall, GFID aggregation graph traversal, 6× override logic, EMT branching, monthly waiver sequencing. Python processes only the ~10,000 aggregated rows output by SQL.

### Rule parameter management

**Rule parameters** (benchmark ratios, tier thresholds, weighting factors, effective dates) → stored in BigQuery tables. They change quarterly, need audit trails, and compliance teams update them without engineering.

**Rule logic** (exemption waterfall, GFID aggregation, EMT branching) → versioned Python code with unit tests. Changes only when the conditional logic itself changes.

### JSON as intermediary

When new benchmarks or rules are published:
1. Extract from source PDF into JSON files (human-readable, reviewable)
2. Business reviews and signs off on JSON
3. Generate BigQuery INSERT SQL programmatically from JSON (JSON is single source of truth)
4. Load to BigQuery; run validation queries
5. Python engine picks up new parameters via effective-date JOINs — no code changes needed

### Rule traceability — connecting PDFs to code

The MEP spec PDF is the legal source of truth. It contains two fundamentally different kinds of content mixed together: **parameter values** (benchmark = 15:1) and **conditional logic** (if ratio > 6× benchmark then no waivers apply). When a new revision drops, either or both may change. Every rule artifact must formally trace back to a specific section of the source document.

The two source documents have different change patterns and need different traceability mechanisms:

**MEP Spec PDF (rules)** — changes incrementally (v11.7 adds MPS rule, v11.8 modifies 6× exclusion). Needs a **rule catalog** (`mep_logic_rules.json`) that maps every implemented rule to its spec section, the Python function that implements it, and whether the rule is a PARAMETER change (update JSON/BQ only) or a LOGIC change (requires Python code modification).

**Benchmark PDF (limits)** — changes wholesale every quarter (entire file replaced). Does NOT need a separate catalog. The existing `mep_benchmarks.json` IS the catalog — it just needs `benchmark_id` and `source_page` fields per entry, plus a **diff tool** (`diff_benchmarks.py`) that mechanically compares two quarterly JSON files and reports which product groups changed.

**Traceability chain:**
- Every entry in `mep_benchmarks.json` has a `benchmark_id` and `source_page` tracing to the benchmark PDF
- Every entry in `mep_rules.json` has a `rule_id` and `spec_section_ref` tracing to the MEP spec PDF
- Every entry in `mep_logic_rules.json` maps a `rule_id` to a specific Python function and spec section
- Every BigQuery row carries `rule_id` and `spec_section_ref` for audit queries
- Every Python function documents which `rule_id`s it implements in its docstring

**When a new spec revision drops (e.g. v11.9):**
1. Check the Appendix C revision table — it states which sections changed
2. Look up affected `rule_id`s in `mep_logic_rules.json` by `spec_section`
3. Check `change_type`: PARAMETER → update JSON only; LOGIC → update Python + tests
4. Run driver scripts to validate; cross-check Script 1 vs Script 2 output

**When new quarterly benchmarks drop:**
1. Extract to new `mep_benchmarks_q4_2026.json`
2. Run `diff_benchmarks.py` against prior quarter JSON → produces change report
3. Business reviews diff report, signs off
4. `generate_bq_sql.py` produces INSERT scripts from the new JSON
5. Load to BQ; Python engine picks up new values via effective-date JOINs

---

## Deliverables Requested

### 1. High-level architecture document
- Four-layer pipeline: Ingest → SQL Aggregate → Python Rules Engine → Looker Report
- Data flow diagram showing what each layer processes and the volume reduction at each stage
- Architecture decision records: why BigQuery + Python (not Java), why the SQL/Python split, why JSON intermediary for rule lifecycle

### 2. BigQuery data model
- Full table schemas (column, type, description) for all reference data tables, order entry tables, and intermediate aggregation tables
- Partitioning and clustering strategy
- `mep_calc.msg_score_by_second` — per-second aggregation table
- `mep_calc.daily_scores` — daily roll-up table (the input to Python)

### 3. SQL aggregation layer
- Step 1: Session classification view (RTH vs ETH, front vs back month, outright vs spread)
- Step 2: Per-event weighted score view (join weight lookup)
- Step 3: Per-second aggregation (for second-level products and EMT MPS)
- Step 4: Daily roll-up with second-level breach flag

### 4. JSON extraction (three files)
- `mep_benchmarks.json` — all product group benchmarks from the benchmark PDF, with `_metadata` (source, quarter, status, effective dates), messaging tiers, mass quote benchmarks. Each product group entry includes `benchmark_id` (e.g. `BM-ES-Q3-2026`) and `source_page` for traceability back to the PDF.
- `mep_rules.json` — all 24 message weight entries, EMT thresholds, exclusion rules, exemption parameters, session definitions, surcharge schedule, status values. Each entry includes `rule_id` and `spec_section_ref` tracing to the MEP spec PDF.
- `mep_logic_rules.json` — **rule catalog** mapping every conditional rule in the Python engine to its source spec section. Each entry contains: `rule_id`, `spec_section`, `spec_text_summary` (plain English from the PDF), `implements_function` (Python function name), `change_type` (PARAMETER, LOGIC, or PARAMETER+LOGIC), and `parameter_key` (if applicable, the JSON path to the parameter). This file is the bridge between the spec PDF and the Python code — when a new revision drops, you look up affected `rule_id`s here to know exactly what to update.

### 5. SQL generation script and benchmark diff tool
- `generate_bq_sql.py` — reads benchmark and rules JSON files, generates BigQuery INSERT statements. Includes: expire-previous-quarter UPDATE, INSERT for new quarter, validation queries with expected values. Every generated row includes `rule_id`/`benchmark_id` and `spec_section_ref` for audit trail.
- `diff_benchmarks.py` — compares two quarterly benchmark JSON files, reports added/removed/changed product groups with old vs new values. Produces a `change_report.json` for audit trail. Used during quarterly benchmark updates to mechanically detect what changed.

### 6. Python rules engine (`mep_rule_logic.py`)
- Loads all parameters from JSON at startup (zero hardcoded business values)
- Pure functions for each rule: `is_excluded()`, `assign_tier()`, `calculate_volume_ratio()`, `is_daily_breach()`, `is_6x_breach()`, `check_emt_gfid()`, `check_emt_mps()`, `apply_exemption_waterfall()`
- Orchestrator `run()` method that sequences calls with no business logic of its own
- GFID aggregation using Union-Find (disjoint set union) for transitive group resolution

### 7. Simulated test data
- 11+ scenarios covering every significant branch: excluded, each tier passing/failing, 6× breach, second-level breach, EMT GFID breach, EMT MPS breach, market maker exempt, zero volume, monthly waiver save, surcharge with waivers exhausted
- Expected results dict for automated validation

### 8. Driver scripts
- **Script 1:** Loads parameters from JSON files (pre-BQ validation path). Runs engine on simulated data. Validates against expected results. Exports CSV.
- **Script 2:** Loads parameters from BigQuery tables (production path, with simulated BQ reads as in-memory stubs). Runs same engine on same data. Cross-validates against Script 1 output to confirm parameter parity.

### 9. Unit tests
- Independent tests for each rule function with boundary conditions
- Tests for GFID aggregation: direct pairs, transitive chains, effective-date filtering, isolated GFIDs
- Tests for exemption waterfall priority ordering

---

## Critical Implementation Notes

Flag these explicitly in the design:

1. **Order Mass Action double-counting risk:** MsgType=CA must expand to N orders for both raw count and weighted score
2. **Spread volume 3× expansion:** verify whether source data generates three rows per spread or one row needing multiplication
3. **RTH/ETH boundary and CME trade date:** an order at 23:30 CT Monday belongs to Tuesday's CME trade date
4. **Per-second products:** a single breaching second = full daily non-compliance; monthly waiver does NOT override
5. **GFID aggregation timing:** takes effect first of month following approval; do not backfill
6. **EMT holiday independence:** EMT calculates on MEP holidays; implement as separate pipeline branches
7. **Monthly waiver sequencing:** daily results first, then monthly aggregate, then auto-waiver; 6× breaches excluded from all waivers
8. **Mass quote = entirely separate pipeline:** different ratio (entries/volume), monthly RTH only, no GFID aggregation
9. **Rule traceability is mandatory:** every BigQuery row must carry `rule_id`/`benchmark_id` + `spec_section_ref`. Every Python function must document which `rule_id`s it implements. The `mep_logic_rules.json` catalog must cover every conditional branch in the rules engine.
10. **Deleted rules:** when a rule is removed in a new spec version, set `effective_to` on affected entries in JSON and BQ. Python function stays for historical replay but is never applied to new dates. Catalog entry gets `archived: true`.

---

## Input Data Consideration

The provided aggregated data groups by (account, session, sessionid, symbol, securitydescription, firm) at day level. The `session` column already contains the RTH/ETH flag — no timestamp-based session classification is needed. The SQL layer must:

- **Session type:** already resolved — the `session` column contains `RTH` or `ETH`. Confirm this maps exactly to MEP's definition (RTH = 07:00–16:00 CT).
- **Resolve month type:** join `ref.front_month_flag` on securitydescription + trade_date to determine front vs back month
- **Resolve outright vs spread:** join `ref.product_spec` to determine if the instrument is a defined spread
- **Map symbol to MEP product group:** the `symbol` field should map to the MEP product group used in benchmarks — confirm this mapping with product taxonomy tables
- **Aggregate to GFID level:** sum across accounts and sessions within the same GFID and product group

The `orders` column maps to New Order (weight 0), `cancels` to Cancellation, `elimination` to FAK/FOK/MinQty, and `trade` to trade messages. **Modification counts are not explicitly in the schema** — clarify whether modifications are included in one of the existing columns or need to be added.

---

## Constraints

- Python 3.12+, no external dependencies beyond standard library + google-cloud-bigquery + pandas (for production BQ reads)
- All business parameters must come from JSON or BQ tables — zero hardcoded values in Python
- Every rule function must be independently unit-testable with no infrastructure
- SQL must be partitioned on trade_date and clustered on gfid + product_group
