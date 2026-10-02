FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.8.17
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
# usuario sin privilegios
RUN useradd -m -u 1000 bot && mkdir -p /app/var /app/strategy && chown -R bot /app/var /app/strategy
USER bot
CMD ["tbot", "run", "--auto"]
