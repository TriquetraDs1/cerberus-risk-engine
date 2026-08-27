# Contributing

This is a solo submission project, but the workflow below is what a reviewer running
the repo, or a future contributor, should expect.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv\Scripts\Activate.ps1 on PowerShell
pip install -r requirements.txt
pip install -e .
pip install ruff pre-commit
pre-commit install
```

## Running checks locally (same as CI)

```bash
ruff check src/ scripts/ tests/
pytest tests/ -v
```

## Adding a new pipeline stage

Each stage lives in its own `src/cerberus/<stage>/` module and is invoked by a thin
`scripts/<verb>_<noun>.py` CLI entry point — see `scripts/generate_data.py` and
`scripts/detect_rings.py` for the pattern. Keep stages independently runnable and
independently testable; the whole point of the architecture (see
`docs/ARCHITECTURE.md`, "Detection layer split is deliberate") is that no stage should
require another to be re-run to be tested in isolation.

## Commit conventions

Conventional-commit-flavored prefixes (`feat:`, `fix:`, `docs:`, `test:`) are
appreciated but not enforced. Reference the day number from `docs/ARCHITECTURE.md`'s
build plan when the commit corresponds to one, e.g. `feat(day3): Louvain ring detector`.
