FROM agrigorev/zoomcamp-model:2025

# Copy uv files
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /code

# Add virtual environment to PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --locked

# Copy the application code
COPY predict.py ./

# Expose the port
EXPOSE 9696

# Run the application
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
