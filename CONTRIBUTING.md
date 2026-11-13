# Contributing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e ".[dev]"
```

Download the NLTK data once (the app degrades gracefully without it, but the
silver tier's accuracy depends on lemmatization):

```python
import nltk
for pkg in ["stopwords", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
```

## Running locally

```bash
# 1. train the tiers you need
python -m newsclass.train --tier bronze      # ~20 seconds
python -m newsclass.train --tier silver      # ~7 minutes on CPU
python -m newsclass.train --tier gold        # ~45 minutes on CPU

# 2. serve
uvicorn newsclass.app:app --reload --port 8000
# interactive docs at http://localhost:8000/docs

# 3. verify
python scripts/smoke_test.py
```

`PYTHONPATH` must include `src` when running without an editable install:

```bash
$env:PYTHONPATH="src"          # PowerShell
export PYTHONPATH=src          # bash
```

## Quality gates

Run these before every commit. CI enforces the same set.

```bash
python scripts/run_tests.py           # 73 tests (self-contained temp dirs)
ruff check src tests scripts          # lint
python scripts/evaluate.py --gate 0.85
```

## Code conventions

- **Type hints everywhere.** The codebase is checked as if it were typed.
- **Public functions have docstrings** explaining *why*, not what the code obviously does.
- **No hard-coded paths or secrets.** Configuration goes through `Settings`.
- **Preprocessing lives in `newsclass/text.py`** and nowhere else. If training and
  serving ever disagree, that module is where the bug belongs.
- **New API fields are additive.** Removing or renaming a response field is a
  breaking change and requires a major version bump.

## Adding a new tier

1. Subclass `TierModel` in `src/newsclass/tiers.py` and implement `ready` and
   `predict_proba`.
2. Register it in `TIER_CLASSES`, `TIER_DISPLAY` and `TIER_DESCRIPTION`.
3. Add a training function to `src/newsclass/train.py` and register it in `BUILDERS`.
4. Add its artifacts marker to `ModelRegistry.available_tiers()`.
5. Extend the tests: contract, fallback ordering, and a real evaluation run.

## Reporting a model issue

Include the `x-request-id` from the response header. It maps the report to the
exact log line without exposing the request body.
