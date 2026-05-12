# README — GEMINI.md

## What is GEMINI.md?

`GEMINI.md` is an AI context file, not application documentation. It is read automatically by **Gemini CLI** and **Gemini Code Assist** (VS Code extension) at the start of every session to load project-wide standards, conventions, and guardrails into the model's working context.

Think of it as a standing briefing document written *for the AI*, not for developers. You should not need to read or reference it during normal development work. Its purpose is to ensure that every AI-assisted interaction in this repository produces output that is consistent with the team's architecture, coding standards, data governance requirements, and SDLC process — without the developer having to re-explain them every time.

---

## Why does this file exist?

Without a context file like this, AI coding assistants generate generic output: they pick arbitrary libraries, ignore team conventions, miss governance requirements, and produce code that needs significant rework before it meets the team's standards.

`GEMINI.md` solves this by giving the AI a stable, versioned understanding of:

- The platform stack (BigQuery, GCS, Python, Argo, Looker, etc.)
- The data domains and their sensitivity levels
- The required SDLC stages and which skill files govern each one
- Library selection rules (pandas by default; polars and PyArrow only when justified)
- Code quality, logging, and error-handling expectations
- Security and data governance rules that apply to all code in this repo
- Folder and naming conventions
- Testing standards and coverage targets
- Deployment patterns (Docker, Kubernetes, Argo Workflows)

---

## How Gemini uses this file

| Tool | How GEMINI.md is loaded |
|---|---|
| **Gemini CLI** | Automatically detected and injected when the CLI is run from any directory in the repository tree |
| **Gemini Code Assist (VS Code)** | Loaded as project context when the extension is active in a workspace that contains this file |

The AI does not show the file's content to the user. It simply behaves as if everything in the file is already understood. Developers interact with Gemini normally; the file works silently in the background.

---

## What GEMINI.md does NOT do

- It does not replace the skill files in the individual task folders. Those contain the detailed, task-specific instructions Gemini follows when assisting with a specific type of work (e.g. writing a BigQuery query, refactoring a notebook, authoring an Argo workflow). `GEMINI.md` sets the project-wide context; the skill files provide the step-by-step guidance.
- It does not enforce anything at runtime. It is a prompt-layer control, not a linter, gate, or policy engine.
- It is not read by humans during development — see the skill folder `README.md` files for that purpose.
- It is not a substitute for code review, testing, or governance processes.

---

## Structure of GEMINI.md

The file is organised into 14 numbered sections. Each section corresponds to an area where consistent AI behaviour matters.

| Section | What it covers |
|---|---|
| 1. Team and Platform Context | Stack, data domains, environment |
| 2. Standard Development Lifecycle | SDLC stages mapped to skill files |
| 3. Language and Library Defaults | Pandas-first policy, SQL standards, notebook rules |
| 4. Code Quality Expectations | Linting, typing, logging, credentials policy |
| 5. Security and Data Governance | Data minimisation, PII handling, auditability |
| 6. Repository and Folder Structure | Top-level layout, naming conventions |
| 7. Notebook-to-Production Principles | Cell design, refactoring progression |
| 8. BigQuery and GCS Patterns | Client usage, parameterisation, I/O patterns |
| 9. Testing Standards | Unit, functional, and performance test requirements |
| 10. Logging, Validation, and Error Handling | Logger setup, schema validation, exception policy |
| 11. Docker and Deployment Standards | Image pinning, secrets policy, Argo integration |
| 12. Looker and Connected Sheets Standards | Semantic layer and governed export requirements |
| 13. Code Review | PR process, Gemini first-pass review, review criteria |
| 14. Skill File Reference | Index of all skill folders and their purposes |

---

## How to maintain this file

`GEMINI.md` should be treated as a living document owned by the Data Science team lead or platform engineer. Update it when:

- The primary stack or a major dependency changes (e.g. a new orchestration platform replaces Argo, a new BI tool replaces Looker).
- A new standing policy is adopted that should apply to all AI-generated code (e.g. a new data sensitivity classification, a new mandatory logging field, a new linting rule).
- A new skill folder is added to the repository — add it to the skill file reference table in Section 14.
- An existing standard is deprecated or replaced — update the relevant section and remove obsolete guidance.
- A library default changes (e.g. if polars is promoted from optional to default) — update Section 3.

When updating, write as if addressing the AI directly. Be specific and prescriptive. Avoid vague guidance like "follow best practices" — instead state the exact expectation (e.g. "Always include a partition filter on the `trade_date` column").

---

## Relationship to the skill files

`GEMINI.md` and the skill files form a two-level instruction hierarchy:

```
GEMINI.md                        ← project-wide context, always active
└── skill files (SKILL.md)       ← task-specific instructions, invoked per task
    ├── sql-bigquery/SKILL.md
    ├── python-pandas/SKILL.md
    ├── unit-tests/SKILL.md
    └── ... (23 skill folders total)
```

When a developer asks Gemini to help with a specific task, Gemini uses the project context from `GEMINI.md` alongside the detailed instructions in the relevant skill file. The skill file tells Gemini *how* to do the task; `GEMINI.md` tells it *what rules and constraints apply* across all tasks.

---

## File location

`GEMINI.md` must remain at the **repository root**. Moving it will prevent Gemini CLI and Gemini Code Assist from detecting it automatically.

```
/
├── GEMINI.md           ← must stay here
├── README.md
├── src/
├── notebooks/
├── sql/
├── tests/
├── docker/
├── argo/
└── ...
```

---

## Questions

For questions about the standards described in `GEMINI.md`, refer to the relevant skill folder `README.md`, which is written for developers and explains the same standards in human-readable form. For questions about the AI tooling itself, refer to the [Gemini Code Assist documentation](https://cloud.google.com/gemini/docs/codeassist/overview) and [Gemini CLI documentation](https://cloud.google.com/gemini/docs/cli/overview).
