# Paper-trading bot — containerized environment.
# The default command runs the full quality gate (lint + types + tests).
# Override the command for paper runs, e.g.:
#   docker run trading-bot python -m bot trade --symbol BTCUSDT --once
FROM python:3.12-slim

WORKDIR /app

# No runtime dependencies beyond the stdlib; dev tools are installed for the gate.
RUN pip install --no-cache-dir pytest ruff mypy pytest-cov

COPY pyproject.toml conftest.py ./
COPY bot/ ./bot/
COPY tests/ ./tests/

# Non-root: the paper loop only ever writes state/ledger/report files in its workdir
RUN useradd --create-home trader && chown -R trader:trader /app
USER trader

CMD ["sh", "-c", "ruff check . && mypy && pytest --cov=bot"]
