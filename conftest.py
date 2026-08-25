# Ensures the repository root is on sys.path so `pytest` (not just
# `python -m pytest`) can import the `bot` package.

# Hermetic tests: cost observations written by run_step go to a scratch dir,
# so the production tape (cost_observations.jsonl) is only ever appended by
# the scheduled CI runner.
import os as _os
import tempfile as _tempfile

_os.environ.setdefault(
    "COST_OBSERVATIONS_LOG",
    _os.path.join(_tempfile.gettempdir(), "trading-bot-test-obs", "cost_observations.jsonl"),
)
